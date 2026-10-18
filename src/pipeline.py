"""Glue: source -> detector -> tracker -> zone test -> BOTH engines -> annotated frame.
Baseline and proposed engines consume the identical observation stream, so the live
comparison is fair. Contains no Streamlit code (testable headless)."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import numpy as np

from .event_engine import CAT_ALERT, EngineConfig, EngineEvent, HazardEventEngine, Observation
from .incident_store import IncidentStore
from .tracking import IoUTracker, TrackedDetection
from .zone_manager import Zone

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class FrameResult:
    __slots__ = ("frame_idx", "ts", "annotated", "tracked", "events", "latency_ms", "detect_ms",
                 "state", "n_active", "pending")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


class MonitoringSession:
    def __init__(self, cfg: dict, detector, source_type: str, store: Optional[IncidentStore] = None,
                 source_name: str = "", run_id: Optional[str] = None):
        self.cfg, self.detector, self.store = cfg, detector, store
        self.source_type, self.source_name = source_type, source_name
        self.run_id = run_id or new_run_id()
        self.zone = Zone(cfg["zone"].get("name", "Restricted Zone"), cfg["zone"]["polygon"])
        self.tracker = IoUTracker(cfg["tracker"]["iou_threshold"], cfg["tracker"]["max_missed"])
        self.engines: Dict[str, HazardEventEngine] = {
            m: HazardEventEngine(EngineConfig.from_app_config(cfg, m, source_type), self.run_id)
            for m in ("baseline", "proposed")}
        self.latencies: List[float] = []
        self.detect_times: List[float] = []
        self._t_start = time.perf_counter()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_ts = 0.0
        self._persisted = set()

    # ------------------------------------------------------------------
    def process_frame(self, frame, ts: float, frame_idx: int) -> FrameResult:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        dets = self.detector.detect(frame, frame_idx)
        t1 = time.perf_counter()
        tracked = self.tracker.update(dets)
        obs = [Observation(t.track_id, t.detection.confidence,
                           self.zone.contains_foot(t.detection.bbox, w, h), t.detection.bbox)
               for t in tracked]
        events: List[EngineEvent] = []
        for e in self.engines.values():
            events += e.process(ts, obs)
        t2 = time.perf_counter()
        self._last_ts = ts
        self.detect_times.append((t1 - t0) * 1000.0)
        self.latencies.append((t2 - t0) * 1000.0)
        self._persist()
        r = FrameResult()
        r.frame_idx, r.ts, r.tracked, r.events = frame_idx, ts, tracked, events
        r.latency_ms, r.detect_ms = self.latencies[-1], self.detect_times[-1]
        prop = self.engines["proposed"]
        r.n_active = len(prop.active_alerts())
        r.pending = prop.pending_candidates(ts)
        r.state = ("ALERT" if r.n_active else
                   "CANDIDATE" if r.pending else "CLEAR")
        r.annotated = self._annotate(frame, r, obs)
        return r

    def _annotate(self, frame, r: FrameResult, obs: List[Observation]):
        img = frame.copy()
        h, w = img.shape[:2]
        pts = np.array(self.zone.to_pixels(w, h), np.int32)
        col = {"ALERT": (77, 77, 229), "CANDIDATE": (36, 165, 245), "CLEAR": (127, 178, 63)}[r.state]
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], True, col, 2)
        thr = self.cfg["confidence_threshold"]
        for o, t in zip(obs, r.tracked):
            x1, y1, x2, y2 = [int(v) for v in o.bbox]
            if o.confidence < thr:
                c = (150, 150, 150)
            else:
                c = (77, 77, 229) if o.in_zone else (127, 178, 63)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.circle(img, (int((x1 + x2) / 2), y2), 4, (0, 220, 255), -1)
            cv2.putText(img, f"ID{o.track_id} {o.confidence:.2f}", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        label = {"ALERT": f"RULE ALERT x{r.n_active} (not a confirmed incident)",
                 "CANDIDATE": "CANDIDATE: validating persistence",
                 "CLEAR": "CLEAR"}[r.state]
        cv2.rectangle(img, (0, 0), (w, 22), (26, 21, 18), -1)
        cv2.putText(img, f"{label}   t={r.ts:6.1f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
        if self.source_type == "synthetic":
            cv2.putText(img, "SIMULATED INPUT", (w - 150, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
        return img

    def _persist(self):
        if not self.store:
            return
        recs = []
        for e in self.engines.values():
            recs += e.incidents
        self.store.upsert_many(recs)

    def finish(self) -> Dict:
        for e in self.engines.values():
            e.flush(self._last_ts)
        self._persist()
        n = len(self.latencies)
        wall = time.perf_counter() - self._t_start
        summary = {"run_id": self.run_id, "started_at": self._started_at,
                   "source_type": self.source_type, "source_name": self.source_name,
                   "detector": getattr(self.detector, "name", "unknown"),
                   "frames_processed": n,
                   "avg_latency_ms": float(np.mean(self.latencies)) if n else None,
                   "p95_latency_ms": float(np.percentile(self.latencies, 95)) if n else None,
                   "avg_detect_ms": float(np.mean(self.detect_times)) if n else None,
                   "fps": (n / wall) if wall > 0 and n else None,
                   "duration_s": self._last_ts,
                   "config_json": __import__("json").dumps(self.cfg)}
        if self.store:
            self.store.add_run(summary)
            ret = self.cfg.get("retention", {})
            self.store.purge(ret.get("max_days", 30), ret.get("max_records", 5000))
        summary["counters"] = {m: dict(e.counters) for m, e in self.engines.items()}
        return summary


def run_source(source, session: MonitoringSession, frame_stride: int = 1,
               max_frames: Optional[int] = None, max_seconds: Optional[float] = None) -> Iterator[FrameResult]:
    """Iterate a FrameSource through the session. Always releases the source."""
    t0 = time.monotonic()
    done = 0
    try:
        for frame, ts, idx in source.frames():
            if idx % max(1, frame_stride):
                continue
            yield session.process_frame(frame, ts, idx)
            done += 1
            if max_frames and done >= max_frames:
                break
            if max_seconds and time.monotonic() - t0 > max_seconds:
                break
    finally:
        source.release()
