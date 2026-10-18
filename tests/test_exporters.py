import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv, io, json, tempfile, os
from datetime import datetime, timezone
from src.exporters import incidents_to_csv, incidents_to_json, evaluation_to_json, write_text
from src.incident_store import COLUMNS, IncidentStore
from src.event_engine import EngineConfig, HazardEventEngine, Observation
from src.evaluation import summarize, run_scenario, load_ground_truth
from src.synthetic import mixed_scenario
from src.config import load_config
from src.pipegen_adapter import to_actionable_event, validate_event


def sample_incidents():
    e = HazardEventEngine(EngineConfig(), "X", lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    for i in range(60):
        e.process(i / 10, [Observation(1, 0.8, True)])
    e.flush()
    return e.incidents


class TestExport(unittest.TestCase):
    def test_json_export(self):                             # (7)
        recs = sample_incidents()
        d = json.loads(incidents_to_json(recs))
        self.assertEqual(d["count"], 1)
        self.assertIn("NOT confirmed", d["note"])
        self.assertEqual(d["incidents"][0]["incident_id"], recs[0]["incident_id"])

    def test_csv_export(self):                              # (8)
        recs = sample_incidents()
        rows = list(csv.DictReader(io.StringIO(incidents_to_csv(recs))))
        self.assertEqual(list(rows[0].keys()), COLUMNS)
        self.assertEqual(rows[0]["category"], "rule_alert")
        self.assertIn("persistence", rows[0]["reason"])

    def test_empty_export(self):
        self.assertEqual(incidents_to_csv([]).strip(), ",".join(COLUMNS))
        self.assertEqual(json.loads(incidents_to_json([]))["count"], 0)

    def test_store_roundtrip_and_update(self):
        s = IncidentStore(":memory:")
        recs = sample_incidents()
        s.upsert_many(recs)
        s.upsert_many(recs)                                  # idempotent
        got = s.list()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["reason"], recs[0]["reason"])
        self.assertEqual(len(s.list(method="baseline")), 0)

    def test_file_store_persists(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            IncidentStore(p).upsert_many(sample_incidents())
            self.assertEqual(len(IncidentStore(p).list()), 1)

    def test_retention(self):
        s = IncidentStore(":memory:")
        recs = []
        for k in range(30):
            r = dict(sample_incidents()[0]); r["incident_id"] = f"I{k:03d}"
            r["detected_at"] = f"2026-01-01T00:00:{k:02d}+00:00"
            recs.append(r)
        s.upsert_many(recs)
        s.purge(max_days=36500, max_records=10)
        self.assertEqual(len(s.list()), 10)

    def test_write_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_text(os.path.join(d, "sub", "a.json"), evaluation_to_json({"x": 1}))
            self.assertEqual(json.load(open(p))["evaluation"]["x"], 1)


class TestEvaluation(unittest.TestCase):
    def test_no_ground_truth_does_not_fabricate(self):
        s = summarize(sample_incidents())
        self.assertNotIn("precision", s)
        self.assertNotIn("recall", s)
        self.assertIn("NOT computed", s["note"])

    def test_precision_recall_on_known_case(self):
        inc = sample_incidents()                              # one alert ~0.0-5.9 s
        s = summarize(inc, [{"start": 1, "end": 4}, {"start": 20, "end": 25}], 60)
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual(s["recall"], 0.5)
        s2 = summarize(inc, [{"start": 20, "end": 25}], 60)
        self.assertEqual(s2["precision"], 0.0)
        self.assertEqual(s2["false_alerts_per_min"], 1.0)

    def test_ground_truth_loading_errors(self):
        with self.assertRaises(ValueError):
            load_ground_truth("[{\"start\": 5}]")
        with self.assertRaises(ValueError):
            load_ground_truth("[{\"start\": 5, \"end\": 1}]")
        with self.assertRaises(ValueError):
            load_ground_truth("/no/such/file.json")

    def test_synthetic_comparison_is_consistent(self):
        r = run_scenario(mixed_scenario(), load_config(use_user=False))
        b, p = r["methods"]["baseline"]["metrics"], r["methods"]["proposed"]["metrics"]
        self.assertLess(p["emitted_alerts"], b["emitted_alerts"])
        self.assertEqual(p["false_alerts"], 0)
        self.assertIn("SYNTHETIC", r["label"])


class TestAdapter(unittest.TestCase):
    def test_actionable_event_schema(self):
        ev = to_actionable_event(sample_incidents()[0])
        self.assertEqual(validate_event(ev), [])
        self.assertFalse(ev["is_confirmed_incident"])


if __name__ == "__main__":
    unittest.main()
