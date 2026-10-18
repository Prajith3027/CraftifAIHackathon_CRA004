"""Configurable, temporally validated, explainable hazard-event engine.

Independent of UI, camera and GPU: it consumes per-frame Observations (track id,
confidence, in-zone flag) with timestamps in seconds and emits explainable events.

Two behaviours from ONE implementation (so the comparison is apples-to-apples):
  * BASELINE : alert on the first confident in-zone observation (persistence 0,
               no hysteresis, no merging).
  * PROPOSED : alert only after the intrusion persists `persistence_s`; short gaps up
               to `exit_grace_s` (missing / low-confidence / momentarily outside)
               do not break the episode (hysteresis); re-entries within
               `dedup_cooldown_s` of a resolved alert are merged into that incident.

Every alert AND every suppression is recorded with a plain-language reason.
An alert is RULE-TRIGGERED, never a confirmed real-world safety incident.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .config import ConfigError

EPS = 1e-6
DISCLAIMER = ("Rule-triggered alert from a software prototype; NOT a confirmed real-world "
              "safety incident and not safety-certified.")
CAT_ALERT = "rule_alert"
CAT_TRANSIENT = "suppressed_transient"
CAT_LOWCONF = "suppressed_low_confidence"


@dataclass(frozen=True)
class Observation:
    track_id: int
    confidence: float
    in_zone: bool                      # foot-point inside the restricted zone
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class EngineConfig:
    method: str = "proposed"           # "proposed" | "baseline"
    confidence_threshold: float = 0.5
    persistence_s: float = 2.0
    exit_grace_s: float = 0.8
    dedup_cooldown_s: float = 3.0
    low_conf_min_obs: int = 3          # shorter low-confidence streaks are counted, not logged
    zone_name: str = "Restricted Zone"
    source_type: str = "video_file"

    def validate(self) -> "EngineConfig":
        if self.method not in ("proposed", "baseline"):
            raise ConfigError(f"method must be 'proposed' or 'baseline' (got {self.method!r})")
        for name, lo, hi in (("confidence_threshold", 0, 1), ("persistence_s", 0, None),
                             ("exit_grace_s", 0, None), ("dedup_cooldown_s", 0, None),
                             ("low_conf_min_obs", 1, None)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{name} must be a number (got {v!r})")
            if v < lo or (hi is not None and v > hi):
                raise ConfigError(f"{name}={v} out of range [{lo}, {hi if hi is not None else 'inf'}]")
        return self

    @classmethod
    def from_app_config(cls, cfg: dict, method: str, source_type: str) -> "EngineConfig":
        base = cls(method="proposed",
                   confidence_threshold=cfg["confidence_threshold"],
                   persistence_s=cfg["persistence_s"],
                   exit_grace_s=cfg["exit_grace_s"],
                   dedup_cooldown_s=cfg["dedup_cooldown_s"],
                   low_conf_min_obs=cfg.get("low_conf_min_obs", 3),
                   zone_name=cfg["zone"].get("name", "Restricted Zone"),
                   source_type=source_type)
        return base.as_baseline().validate() if method == "baseline" else base.validate()

    def as_baseline(self) -> "EngineConfig":
        """Naive rule: same confidence threshold, but no persistence/hysteresis/merging."""
        return replace(self, method="baseline", persistence_s=0.0, exit_grace_s=0.0,
                       dedup_cooldown_s=0.0)


@dataclass(frozen=True)
class EngineEvent:
    kind: str          # CANDIDATE_STARTED | ALERT_RAISED | ALERT_MERGED | ALERT_RESOLVED | SUPPRESSED
    ts: float
    incident_id: Optional[str]
    track_id: int
    message: str


class _Episode:
    __slots__ = ("track_id", "start", "last", "conf_sum", "conf_n", "max_conf",
                 "uncertain_obs", "missing_obs", "incident_id")

    def __init__(self, track_id: int, ts: float):
        self.track_id, self.start, self.last = track_id, ts, ts
        self.conf_sum = 0.0
        self.conf_n = 0
        self.max_conf = 0.0
        self.uncertain_obs = 0
        self.missing_obs = 0
        self.incident_id: Optional[str] = None


class _Uncertain:
    __slots__ = ("track_id", "start", "last", "conf_sum", "n", "max_conf")

    def __init__(self, track_id: int, ts: float, conf: float):
        self.track_id, self.start, self.last = track_id, ts, ts
        self.conf_sum, self.n, self.max_conf = conf, 1, conf


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HazardEventEngine:
    def __init__(self, config: EngineConfig, run_id: str = "run",
                 clock: Callable[[], datetime] = _utcnow):
        self.cfg = config.validate()
        self.run_id = run_id
        self._clock = clock
        self._seq = 0
        self._last_ts: Optional[float] = None
        self._episodes: Dict[int, _Episode] = {}
        self._uncertain: Dict[int, _Uncertain] = {}
        self._incidents: Dict[str, dict] = {}
        self._meta: Dict[str, dict] = {}          # per-incident running confidence sums
        self._last_resolved: Optional[str] = None
        self.counters = {"candidate_events": 0, "emitted_alerts": 0, "merged_reentries": 0,
                         "suppressed_transient": 0, "suppressed_low_confidence": 0,
                         "ignored_low_conf_blips": 0}

    # ------------------------------------------------------------------ public
    @property
    def incidents(self) -> List[dict]:
        return [dict(v) for v in self._incidents.values()]

    def active_alerts(self) -> List[dict]:
        return [dict(v) for v in self._incidents.values() if v["status"] == "active"]

    def pending_candidates(self, ts: float) -> List[dict]:
        """Episodes in progress that have not (yet) satisfied persistence."""
        return [{"track_id": e.track_id, "elapsed_s": round(e.last - e.start, 3),
                 "needed_s": self.cfg.persistence_s}
                for e in self._episodes.values() if e.incident_id is None]

    def process(self, ts: float, observations: List[Observation]) -> List[EngineEvent]:
        if self._last_ts is not None and ts < self._last_ts - EPS:
            raise ValueError(f"timestamps must be non-decreasing ({ts} < {self._last_ts})")
        self._last_ts = ts
        events: List[EngineEvent] = []
        seen = set()
        for o in observations:
            if o.track_id in seen:
                continue
            seen.add(o.track_id)
            self._observe(ts, o, events)
        for tid, ep in self._episodes.items():
            if tid not in seen:
                ep.missing_obs += 1          # explicit: detection missing this frame
        self._expire(ts, events, force=False, flushing=False)
        self._sync_active()
        return events

    def flush(self, ts: Optional[float] = None) -> List[EngineEvent]:
        """End of stream: close everything still open."""
        events: List[EngineEvent] = []
        self._expire(self._last_ts if ts is None else ts, events, force=True, flushing=True)
        return events

    # ----------------------------------------------------------------- internals
    def _observe(self, ts: float, o: Observation, events: List[EngineEvent]) -> None:
        cfg = self.cfg
        confirmed = o.in_zone and o.confidence >= cfg.confidence_threshold
        if confirmed:
            self._uncertain.pop(o.track_id, None)
            ep = self._episodes.get(o.track_id)
            if ep is None:
                ep = _Episode(o.track_id, ts)
                self._episodes[o.track_id] = ep
                self.counters["candidate_events"] += 1
                events.append(EngineEvent("CANDIDATE_STARTED", ts, None, o.track_id,
                                          f"Track {o.track_id} foot-point entered '{cfg.zone_name}' "
                                          f"(conf {o.confidence:.2f}); validating persistence."))
            ep.last = ts
            ep.conf_sum += o.confidence
            ep.conf_n += 1
            ep.max_conf = max(ep.max_conf, o.confidence)
            if ep.incident_id is None and ep.last - ep.start + EPS >= cfg.persistence_s:
                self._raise(ts, ep, events)
        elif o.in_zone:                       # uncertain: in zone but below confidence threshold
            ep = self._episodes.get(o.track_id)
            if ep is not None:
                ep.uncertain_obs += 1
            else:
                u = self._uncertain.get(o.track_id)
                if u is None:
                    self._uncertain[o.track_id] = _Uncertain(o.track_id, ts, o.confidence)
                else:
                    u.last, u.n = ts, u.n + 1
                    u.conf_sum += o.confidence
                    u.max_conf = max(u.max_conf, o.confidence)
        # present but outside zone: nothing to do; the episode ages out via exit_grace_s

    def _new_id(self) -> str:
        self._seq += 1
        tag = "P" if self.cfg.method == "proposed" else "B"
        return f"{self.run_id}-{tag}{self._seq:04d}"

    def _record(self, **kw) -> dict:
        iid = self._new_id()
        rec = {"incident_id": iid, "run_id": self.run_id, "method": self.cfg.method,
               "source_type": self.cfg.source_type, "zone_name": self.cfg.zone_name,
               "merged_reentries": 0, "detected_at": self._clock().isoformat(),
               "disclaimer": DISCLAIMER}
        rec.update(kw)
        self._incidents[iid] = rec
        return rec

    def _raise(self, ts: float, ep: _Episode, events: List[EngineEvent]) -> None:
        cfg = self.cfg
        dur = ep.last - ep.start
        prev = self._incidents.get(self._last_resolved) if self._last_resolved else None
        if (prev is not None and prev["status"] == "resolved" and cfg.dedup_cooldown_s > 0
                and 0 <= ep.start - prev["end_ts"] <= cfg.dedup_cooldown_s + EPS):
            # Deduplicate: re-entry shortly after resolution is the same incident.
            prev["status"] = "active"
            prev["merged_reentries"] += 1
            prev["reason"] += (f" | Re-entry of track {ep.track_id} {ep.start - prev['end_ts']:.1f} s "
                               f"after resolution (<= dedup cooldown {cfg.dedup_cooldown_s} s) "
                               f"merged into this incident instead of raising a new alert.")
            ep.incident_id = prev["incident_id"]
            self.counters["merged_reentries"] += 1
            events.append(EngineEvent("ALERT_MERGED", ts, prev["incident_id"], ep.track_id,
                                      f"Re-entry merged into {prev['incident_id']} (dedup)."))
            return
        mean = ep.conf_sum / max(ep.conf_n, 1)
        if cfg.method == "baseline":
            why = (f"BASELINE rule: person (track {ep.track_id}) foot-point inside "
                   f"'{cfg.zone_name}' with confidence {mean:.2f} >= {cfg.confidence_threshold:.2f}; "
                   f"alert raised immediately (no persistence check).")
        else:
            why = (f"Person (track {ep.track_id}) foot-point stayed inside '{cfg.zone_name}' for "
                   f"{dur:.1f} s >= persistence threshold {cfg.persistence_s:.1f} s; mean confidence "
                   f"{mean:.2f} >= {cfg.confidence_threshold:.2f}.")
        rec = self._record(category=CAT_ALERT, status="active", track_id=ep.track_id,
                           start_ts=ep.start, end_ts=ep.last, duration_s=dur, alert_ts=ts,
                           confidence=mean, max_confidence=ep.max_conf, reason=why)
        ep.incident_id = rec["incident_id"]
        self._meta[rec["incident_id"]] = {"sum": 0.0, "n": 0}
        self.counters["emitted_alerts"] += 1
        events.append(EngineEvent("ALERT_RAISED", ts, rec["incident_id"], ep.track_id, why))

    def _expire(self, ts: float, events: List[EngineEvent], force: bool, flushing: bool) -> None:
        cfg = self.cfg
        for tid in list(self._episodes):
            ep = self._episodes[tid]
            gap = ts - ep.last
            if force or gap > cfg.exit_grace_s + EPS:
                self._close(ts, ep, gap, events, flushing)
                del self._episodes[tid]
        for tid in list(self._uncertain):
            u = self._uncertain[tid]
            if force or ts - u.last > cfg.exit_grace_s + EPS:
                self._finalize_uncertain(ts, u, events)
                del self._uncertain[tid]

    def _close(self, ts, ep: _Episode, gap: float, events, flushing: bool) -> None:
        cfg = self.cfg
        mean = ep.conf_sum / max(ep.conf_n, 1)
        if ep.incident_id is not None:
            rec = self._incidents[ep.incident_id]
            meta = self._meta.setdefault(rec["incident_id"], {"sum": 0.0, "n": 0})
            meta["sum"] += ep.conf_sum
            meta["n"] += ep.conf_n
            rec.update(status="resolved", end_ts=ep.last, duration_s=ep.last - rec["start_ts"],
                       confidence=meta["sum"] / max(meta["n"], 1),
                       max_confidence=max(rec["max_confidence"], ep.max_conf))
            tail = ("stream ended" if flushing else
                    f"no confirmed in-zone detection for {gap:.1f} s > exit grace {cfg.exit_grace_s:.1f} s")
            rec["reason"] += (f" | Resolved: {tail}; total duration {rec['duration_s']:.1f} s; gap observations: "
                              f"{ep.missing_obs} missing, {ep.uncertain_obs} low-confidence.")
            self._last_resolved = rec["incident_id"]
            events.append(EngineEvent("ALERT_RESOLVED", ts, rec["incident_id"], ep.track_id,
                                      f"Alert {rec['incident_id']} resolved after {rec['duration_s']:.1f} s."))
            return
        dur = ep.last - ep.start
        why = (f"Intrusion candidate (track {ep.track_id}) lasted {dur:.1f} s < persistence threshold "
               f"{cfg.persistence_s:.1f} s ({ep.conf_n} confirmed in-zone observations, mean conf "
               f"{mean:.2f}); suppressed as transient"
               f"{' (stream ended)' if flushing else ''}.")
        rec = self._record(category=CAT_TRANSIENT, status="suppressed", track_id=ep.track_id,
                           start_ts=ep.start, end_ts=ep.last, duration_s=dur, alert_ts=None,
                           confidence=mean, max_confidence=ep.max_conf, reason=why)
        self.counters["suppressed_transient"] += 1
        events.append(EngineEvent("SUPPRESSED", ts, rec["incident_id"], ep.track_id, why))

    def _finalize_uncertain(self, ts, u: _Uncertain, events) -> None:
        cfg = self.cfg
        if u.n < cfg.low_conf_min_obs:
            self.counters["ignored_low_conf_blips"] += 1
            return
        mean = u.conf_sum / u.n
        why = (f"Person (track {u.track_id}) appeared in '{cfg.zone_name}' for {u.last - u.start:.1f} s "
               f"but detection confidence stayed below {cfg.confidence_threshold:.2f} (max {u.max_conf:.2f}); "
               f"no alert raised because the observation was too uncertain. Possible missed intrusion - "
               f"review manually.")
        rec = self._record(category=CAT_LOWCONF, status="suppressed", track_id=u.track_id,
                           start_ts=u.start, end_ts=u.last, duration_s=u.last - u.start, alert_ts=None,
                           confidence=mean, max_confidence=u.max_conf, reason=why)
        self.counters["suppressed_low_confidence"] += 1
        events.append(EngineEvent("SUPPRESSED", ts, rec["incident_id"], u.track_id, why))

    def _sync_active(self) -> None:
        """Keep live duration/confidence of ongoing alerts up to date."""
        for ep in self._episodes.values():
            if ep.incident_id is None:
                continue
            rec = self._incidents[ep.incident_id]
            meta = self._meta.get(rec["incident_id"], {"sum": 0.0, "n": 0})
            n = meta["n"] + ep.conf_n
            rec["end_ts"] = ep.last
            rec["duration_s"] = ep.last - rec["start_ts"]
            rec["confidence"] = (meta["sum"] + ep.conf_sum) / max(n, 1)
            rec["max_confidence"] = max(rec["max_confidence"], ep.max_conf)
