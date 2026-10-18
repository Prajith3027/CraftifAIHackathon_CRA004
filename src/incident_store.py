"""SQLite-backed persistent incident + run store (stdlib only)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

COLUMNS = ["incident_id", "run_id", "method", "category", "status", "track_id", "source_type",
           "zone_name", "start_ts", "end_ts", "duration_s", "alert_ts", "confidence",
           "max_confidence", "merged_reentries", "detected_at", "reason", "disclaimer"]
RUN_COLUMNS = ["run_id", "started_at", "source_type", "source_name", "detector", "frames_processed",
               "avg_latency_ms", "p95_latency_ms", "avg_detect_ms", "fps", "duration_s", "config_json"]

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "contextguard.db"


class IncidentStore:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS incidents (incident_id TEXT PRIMARY KEY, "
                      + ", ".join(f"{k} TEXT" if k in ("incident_id", "run_id", "method", "category",
                                  "status", "source_type", "zone_name", "detected_at", "reason",
                                  "disclaimer") else f"{k} REAL"
                                  for k in COLUMNS[1:]) + ")")
            c.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT, "
                      "source_type TEXT, source_name TEXT, detector TEXT, frames_processed REAL, "
                      "avg_latency_ms REAL, p95_latency_ms REAL, avg_detect_ms REAL, fps REAL, "
                      "duration_s REAL, config_json TEXT)")

    def _conn(self):
        if self._mem is not None:
            return _Keep(self._mem)
        return _Close(sqlite3.connect(self.path))

    # --- incidents
    def upsert(self, rec: Dict) -> None:
        self.upsert_many([rec])

    def upsert_many(self, recs: List[Dict]) -> None:
        if not recs:
            return
        q = f"INSERT OR REPLACE INTO incidents ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})"
        with self._conn() as c:
            c.executemany(q, [[r.get(k) for k in COLUMNS] for r in recs])

    def list(self, method: Optional[str] = None, run_id: Optional[str] = None,
             category: Optional[str] = None) -> List[Dict]:
        where, args = [], []
        for col, val in (("method", method), ("run_id", run_id), ("category", category)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        q = "SELECT " + ",".join(COLUMNS) + " FROM incidents"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY detected_at, incident_id"
        with self._conn() as c:
            return [dict(zip(COLUMNS, row)) for row in c.execute(q, args).fetchall()]

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM incidents")
            c.execute("DELETE FROM runs")

    def purge(self, max_days: int, max_records: int) -> int:
        """Apply retention: drop records older than max_days, then keep newest max_records."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_days)).isoformat()
        with self._conn() as c:
            n1 = c.execute("DELETE FROM incidents WHERE detected_at < ?", (cutoff,)).rowcount
            n2 = c.execute("DELETE FROM incidents WHERE incident_id NOT IN (SELECT incident_id FROM "
                           "incidents ORDER BY detected_at DESC, incident_id DESC LIMIT ?)",
                           (max_records,)).rowcount
        return n1 + n2

    # --- runs
    def add_run(self, run: Dict) -> None:
        q = f"INSERT OR REPLACE INTO runs ({','.join(RUN_COLUMNS)}) VALUES ({','.join('?' * len(RUN_COLUMNS))})"
        with self._conn() as c:
            c.execute(q, [run.get(k) for k in RUN_COLUMNS])

    def runs(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT " + ",".join(RUN_COLUMNS) + " FROM runs ORDER BY started_at").fetchall()
        return [dict(zip(RUN_COLUMNS, r)) for r in rows]


class _Close:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self.conn
    def __exit__(self, et, ev, tb):
        try:
            self.conn.commit() if et is None else self.conn.rollback()
        finally:
            self.conn.close()


class _Keep:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self.conn
    def __exit__(self, et, ev, tb):
        self.conn.commit() if et is None else self.conn.rollback()
