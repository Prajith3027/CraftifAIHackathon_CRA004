"""CSV / JSON exporters for incident records and evaluation summaries."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .incident_store import COLUMNS

SCHEMA_VERSION = "1.0"
NOTE = ("Records are rule-triggered alerts/suppressions from a software prototype. They are NOT "
        "confirmed real-world safety incidents. source_type='synthetic' means simulated data.")


def incidents_to_csv(records: List[Dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in records:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
    return buf.getvalue()


def incidents_to_json(records: List[Dict]) -> str:
    return json.dumps({"schema_version": SCHEMA_VERSION, "note": NOTE,
                       "exported_at": datetime.now(timezone.utc).isoformat(),
                       "count": len(records), "incidents": records}, indent=2, default=str)


def evaluation_to_json(summary: Dict) -> str:
    return json.dumps({"schema_version": SCHEMA_VERSION,
                       "exported_at": datetime.now(timezone.utc).isoformat(),
                       "evaluation": summary}, indent=2, default=str)


def write_text(path: Path | str, text: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)
