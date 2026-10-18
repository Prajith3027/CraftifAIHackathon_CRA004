"""Maps ContextGuard incidents to a PROPOSED actionable-event JSON schema.

STATUS: this is OUR proposed schema. It has NOT been validated against CraftifAI PipeGen
(the portal was not accessible while building). It performs no PipeGen execution.
"""
from __future__ import annotations

from typing import Dict, List

SCHEMA_ID = "contextguard.event.v1-proposed"
REQUIRED = ["schema", "event_id", "event_type", "timestamp_utc", "stream_time_s", "confidence",
            "duration_s", "zone", "reason", "is_confirmed_incident", "source_type"]


def to_actionable_event(incident: Dict) -> Dict:
    ev = {"schema": SCHEMA_ID,
          "event_id": incident["incident_id"],
          "event_type": incident["category"],
          "timestamp_utc": incident.get("detected_at"),
          "stream_time_s": incident.get("start_ts"),
          "confidence": incident.get("confidence"),
          "duration_s": incident.get("duration_s"),
          "zone": incident.get("zone_name"),
          "track_id": incident.get("track_id"),
          "reason": incident.get("reason"),
          "is_confirmed_incident": False,   # rule-triggered only; never a verified safety incident
          "source_type": incident.get("source_type")}
    problems = validate_event(ev)
    if problems:
        raise ValueError("; ".join(problems))
    return ev


def validate_event(ev: Dict) -> List[str]:
    probs = [f"missing field '{k}'" for k in REQUIRED if k not in ev]
    if ev.get("is_confirmed_incident") is not False:
        probs.append("is_confirmed_incident must be False for rule-triggered events")
    return probs
