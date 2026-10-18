"""Evaluation of baseline vs proposed rule engines. Never fabricates numbers:
precision/recall/false-alerts-per-minute are only computed when ground truth is supplied."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .event_engine import CAT_ALERT, CAT_LOWCONF, CAT_TRANSIENT, EngineConfig, HazardEventEngine, Observation
from .synthetic import Scenario
from .tracking import IoUTracker
from .zone_manager import Zone

NO_GT_NOTE = ("Precision, recall and false alerts/min need ground-truth annotations; none were "
              "supplied, so they are NOT computed.")


def load_ground_truth(path_or_text: str) -> List[Dict[str, float]]:
    """GT file: JSON list of {"start": s, "end": s} (seconds of video time)."""
    try:
        text = open(path_or_text).read() if not path_or_text.lstrip().startswith(("[", "{")) else path_or_text
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid ground-truth input: {e}") from e
    if isinstance(data, dict):
        data = data.get("ground_truth", data.get("events"))
    if not isinstance(data, list):
        raise ValueError("Ground truth must be a JSON list of {start, end} objects.")
    out = []
    for g in data:
        try:
            s, e = float(g["start"]), float(g["end"])
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(f"Bad ground-truth entry {g!r}: need numeric start/end") from ex
        if e < s:
            raise ValueError(f"Ground-truth entry has end < start: {g!r}")
        out.append({"start": s, "end": e})
    return out


def _overlaps(a_start, a_end, g, tol):
    return a_start <= g["end"] + tol and a_end >= g["start"] - tol


def summarize(incidents: Sequence[Dict], ground_truth: Optional[Sequence[Dict]] = None,
              stream_duration_s: Optional[float] = None, tolerance_s: float = 0.0) -> Dict:
    alerts = [i for i in incidents if i["category"] == CAT_ALERT]
    transient = [i for i in incidents if i["category"] == CAT_TRANSIENT]
    lowconf = [i for i in incidents if i["category"] == CAT_LOWCONF]
    merged = sum(int(i.get("merged_reentries") or 0) for i in alerts)
    s: Dict = {"candidate_events": len(alerts) + merged + len(transient),
               "emitted_alerts": len(alerts), "merged_reentries": merged,
               "suppressed_transient": len(transient), "suppressed_low_confidence": len(lowconf),
               "ground_truth_supplied": bool(ground_truth)}
    if not ground_truth:
        s["note"] = NO_GT_NOTE
        return s
    matched_gt = set()
    tp = 0
    lat = []
    for a in alerts:
        hit = [k for k, g in enumerate(ground_truth) if _overlaps(a["start_ts"], a["end_ts"], g, tolerance_s)]
        if hit:
            tp += 1
            for k in hit:
                if k not in matched_gt:
                    lat.append(a["alert_ts"] - ground_truth[k]["start"])
                matched_gt.add(k)
    fp = len(alerts) - tp
    s.update({"gt_events": len(ground_truth), "true_positive_alerts": tp, "false_alerts": fp,
              "gt_events_detected": len(matched_gt),
              "precision": (tp / len(alerts)) if alerts else None,
              "recall": len(matched_gt) / len(ground_truth),
              "alerts_per_detected_gt_event": (tp / len(matched_gt)) if matched_gt else None,
              "median_alert_delay_s": float(np.median(lat)) if lat else None})
    if stream_duration_s and stream_duration_s > 0:
        s["false_alerts_per_min"] = fp / (stream_duration_s / 60.0)
    else:
        s["false_alerts_per_min"] = None
        s["note"] = "stream duration unknown: false alerts/min not computed"
    return s


def run_scenario(scenario: Scenario, cfg: dict, run_id: str = "eval") -> Dict:
    """Headless run of BOTH engines on a detection stream (tracker + zone + engine, no images).
    Latency here is rule-engine-only (no detector), reported as such."""
    zone = Zone(cfg["zone"].get("name", "Zone"), cfg["zone"]["polygon"])
    engines = {m: HazardEventEngine(EngineConfig.from_app_config(cfg, m, "synthetic"), f"{run_id}")
               for m in ("baseline", "proposed")}
    tracker = IoUTracker(cfg["tracker"]["iou_threshold"], cfg["tracker"]["max_missed"])
    lat = []
    for i, dets in enumerate(scenario.frames):
        t0 = time.perf_counter()
        ts = i / scenario.fps
        tracked = tracker.update(dets)
        obs = [Observation(t.track_id, t.detection.confidence,
                           zone.contains_foot(t.detection.bbox, scenario.width, scenario.height),
                           t.detection.bbox) for t in tracked]
        for e in engines.values():
            e.process(ts, obs)
        lat.append((time.perf_counter() - t0) * 1000.0)
    end_ts = len(scenario.frames) / scenario.fps
    for e in engines.values():
        e.flush(end_ts)
    total_s = sum(lat) / 1000.0
    out = {"scenario": scenario.name, "synthetic": scenario.synthetic, "duration_s": scenario.duration_s,
           "frames": len(scenario.frames),
           "performance": {"scope": "tracker + zone test + both rule engines (no detector, no drawing)",
                           "avg_latency_ms": float(np.mean(lat)) if lat else None,
                           "p95_latency_ms": float(np.percentile(lat, 95)) if lat else None,
                           "throughput_fps": (len(lat) / total_s) if total_s > 0 else None},
           "ground_truth": scenario.ground_truth, "methods": {}}
    for m, e in engines.items():
        out["methods"][m] = {"metrics": summarize(e.incidents, scenario.ground_truth, scenario.duration_s),
                             "incidents": e.incidents}
    out["label"] = ("SYNTHETIC data with ground truth defined by our own annotation policy: validates "
                    "rule-engine logic only; NOT evidence of real-world performance.")
    return out
