"""SYNTHETIC scenario generator (SIMULATED data - not real-world evidence).

Builds deterministic detection streams with known ground truth so the rule engine can be
tested and demonstrated offline. Ground truth is derived from the noise-free simulated foot
positions using an explicit annotation policy: a stay in the zone of >= gt_min_duration_s is
a 'true intrusion'. Because the policy is ours, results on this data validate engine LOGIC
only; they say nothing about real-world accuracy.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .detector import Detection
from .zone_manager import Zone

W, H, FPS = 640, 360, 10.0
PERSON_W, PERSON_H = 44.0, 110.0
DEFAULT_POLYGON = [[0.55, 0.35], [0.95, 0.35], [0.95, 0.95], [0.55, 0.95]]


@dataclass
class Actor:
    name: str
    waypoints: List[Tuple[float, float, float]]            # (t_s, foot_x_norm, foot_y_norm)
    base_conf: float = 0.85
    dropouts: List[Tuple[float, float]] = field(default_factory=list)   # (start_s, end_s) no detection

    def foot_at(self, t: float):
        wp = self.waypoints
        if t < wp[0][0] or t > wp[-1][0]:
            return None
        for (t0, x0, y0), (t1, x1, y1) in zip(wp, wp[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
        return wp[-1][1], wp[-1][2]


@dataclass
class Scenario:
    name: str
    description: str
    fps: float
    duration_s: float
    frames: List[List[Detection]]
    ground_truth: List[Dict[str, float]]
    width: int = W
    height: int = H
    synthetic: bool = True


def _gt_from_actors(actors: Sequence[Actor], zone: Zone, duration: float, min_dur: float):
    gt, step = [], 0.01
    for a in actors:
        start = None
        t = 0.0
        while t <= duration + 1e-9:
            p = a.foot_at(t)
            inside = p is not None and zone.contains_normalized(*p)
            if inside and start is None:
                start = t
            if (not inside or t + step > duration) and start is not None:
                end = t if not inside else t
                if end - start >= min_dur:
                    gt.append({"start": round(start, 2), "end": round(end, 2), "actor": a.name})
                start = None
            t += step
    return sorted(gt, key=lambda g: g["start"])


def build_scenario(name: str, actors: Sequence[Actor], duration: float, description: str,
                   seed: int = 7, fps: float = FPS, gt_min_duration_s: float = 2.0,
                   polygon=DEFAULT_POLYGON) -> Scenario:
    rng = random.Random(seed)
    zone = Zone("Restricted Zone", polygon)
    n = int(duration * fps)
    frames: List[List[Detection]] = []
    for i in range(n):
        t = i / fps
        dets = []
        for a in actors:
            p = a.foot_at(t)
            if p is None or any(s <= t < e for s, e in a.dropouts):
                continue
            fx, fy = p[0] * W, p[1] * H
            conf = max(0.01, min(0.99, a.base_conf + rng.uniform(-0.05, 0.05)))
            dets.append(Detection((fx - PERSON_W / 2, fy - PERSON_H, fx + PERSON_W / 2, fy), conf))
        frames.append(dets)
    gt = _gt_from_actors(actors, zone, duration, gt_min_duration_s)
    return Scenario(name, description, fps, duration, frames, gt)


def mixed_scenario(seed: int = 7) -> Scenario:
    """60 s timeline combining the cases the engine must handle."""
    out_x, out_y = 0.25, 0.80
    in_x, in_y = 0.75, 0.80
    actors = [
        # 1. Genuine sustained intrusion (true positive)
        Actor("A_sustained", [(2, out_x, out_y), (4, in_x, in_y), (10, in_x, in_y), (12, out_x, out_y)]),
        # 2. Brief graze of the zone edge (annotation policy: NOT a true intrusion)
        Actor("B_graze", [(14, 0.40, 0.80), (15.0, 0.62, 0.80), (15.6, 0.62, 0.80), (16.6, 0.40, 0.80)]),
        # 3. Genuine intrusion with detector dropouts every 2 s (flicker): baseline fragments it
        Actor("C_flicker", [(20, in_x, in_y), (30, in_x, in_y)],
              dropouts=[(22, 22.4), (24, 24.4), (26, 26.4), (28, 28.4)]),
        # 4. Genuine intrusion detected only with LOW confidence (both methods will miss it)
        Actor("D_lowconf", [(34, in_x, in_y), (40, in_x, in_y)], base_conf=0.32),
        # 5. Person pacing outside the zone the whole time (no intrusion)
        Actor("E_outside", [(44, 0.15, 0.7), (50, 0.35, 0.7), (56, 0.15, 0.7)]),
    ]
    return build_scenario("mixed_synthetic", actors, 60.0,
                          "SYNTHETIC: sustained intrusion, edge graze, flickering intrusion, "
                          "low-confidence intrusion, outside pacing.", seed=seed)


def dataset_to_dict(sc: Scenario) -> dict:
    return {"label": "SYNTHETIC - simulated data, not real-world evidence",
            "name": sc.name, "description": sc.description, "fps": sc.fps,
            "duration_s": sc.duration_s, "frame_size": [sc.width, sc.height],
            "ground_truth_policy": "intrusion = noise-free foot-point inside zone for >= 2.0 s",
            "ground_truth": sc.ground_truth,
            "frames": [[{"bbox": list(d.bbox), "confidence": round(d.confidence, 4)} for d in f]
                       for f in sc.frames]}


def save_dataset(sc: Scenario, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset_to_dict(sc), f)


def load_dataset(path: str) -> Scenario:
    with open(path) as f:
        d = json.load(f)
    frames = [[Detection(tuple(x["bbox"]), x["confidence"]) for x in fr] for fr in d["frames"]]
    return Scenario(d["name"], d["description"], d["fps"], d["duration_s"], frames,
                    d["ground_truth"], d["frame_size"][0], d["frame_size"][1])


class SyntheticSource:
    """Renders blank 'SIMULATED' frames so the same pipeline/UI code path is exercised."""
    kind = "synthetic"

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.fps, self.width, self.height = scenario.fps, scenario.width, scenario.height
        self.total_frames = len(scenario.frames)

    def frames(self):
        import cv2
        for i in range(self.total_frames):
            img = np.full((self.height, self.width, 3), (30, 26, 22), np.uint8)
            for x in range(0, self.width, 40):
                cv2.line(img, (x, 0), (x, self.height), (40, 36, 32), 1)
            for y in range(0, self.height, 40):
                cv2.line(img, (0, y), (self.width, y), (40, 36, 32), 1)
            yield img, i / self.fps, i

    def release(self):
        pass
