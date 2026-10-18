"""Minimal greedy IoU tracker. Keeps IDs stable across short gaps; not a substitute
for ByteTrack/DeepSORT (documented limitation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .detector import Detection


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[TrackedDetection]:
        pairs = []
        for tid, t in self._tracks.items():
            for di, d in enumerate(detections):
                s = iou(t["bbox"], d.bbox)
                if s >= self.iou_threshold:
                    pairs.append((s, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d, out = set(), set(), []
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            self._tracks[tid].update(bbox=detections[di].bbox, missed=0)
            out.append(TrackedDetection(tid, detections[di]))
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"bbox": d.bbox, "missed": 0}
            used_t.add(tid)
            out.append(TrackedDetection(tid, d))
        for tid in list(self._tracks):
            if tid not in used_t:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_missed:
                    del self._tracks[tid]
        return out
