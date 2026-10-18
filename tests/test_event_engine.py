import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
from src.config import ConfigError
from src.event_engine import (CAT_ALERT, CAT_LOWCONF, CAT_TRANSIENT, EngineConfig,
                              HazardEventEngine, Observation)

FPS = 10.0
FIXED = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def run(engine, plan, fps=FPS, n=None):
    """plan: function(frame_index, t) -> list[Observation]. Returns all events."""
    ev = []
    n = n or 0
    for i in range(n):
        ev += engine.process(i / fps, plan(i, i / fps))
    return ev


def eng(**kw):
    base = dict(confidence_threshold=0.5, persistence_s=2.0, exit_grace_s=0.8, dedup_cooldown_s=3.0)
    base.update(kw)
    return HazardEventEngine(EngineConfig(**base), "T", FIXED)


def inzone(conf=0.9, tid=1):
    return [Observation(tid, conf, True)]


def cats(e):
    return [i["category"] for i in e.incidents]


class TestPersistence(unittest.TestCase):
    def test_alert_only_after_persistence(self):           # (3)
        e = eng()
        ev = run(e, lambda i, t: inzone(), n=19)            # t = 0 .. 1.8 s
        self.assertFalse([x for x in ev if x.kind == "ALERT_RAISED"])
        ev = run_more(e, 19, 5)
        raised = [x for x in ev if x.kind == "ALERT_RAISED"]
        self.assertEqual(len(raised), 1)
        self.assertAlmostEqual(raised[0].ts, 2.0, places=6)
        self.assertIn("persistence threshold 2.0", raised[0].message)

    def test_transient_suppressed_with_reason(self):
        e = eng()
        run(e, lambda i, t: inzone() if i < 5 else [], n=40)
        e.flush()
        self.assertEqual(cats(e), [CAT_TRANSIENT])
        self.assertIn("suppressed as transient", e.incidents[0]["reason"])
        self.assertEqual(e.counters["emitted_alerts"], 0)

    def test_enter_leave_resolves(self):                    # (2)
        e = eng()
        run(e, lambda i, t: inzone() if 5 <= i < 45 else [Observation(1, 0.9, False)], n=80)
        inc = e.incidents[0]
        self.assertEqual((inc["category"], inc["status"]), (CAT_ALERT, "resolved"))
        self.assertAlmostEqual(inc["start_ts"], 0.5, places=6)
        self.assertAlmostEqual(inc["end_ts"], 4.4, places=6)
        self.assertIn("Resolved", inc["reason"])
        self.assertEqual(e.active_alerts(), [])


def run_more(e, start, n, plan=None):
    ev = []
    for i in range(start, start + n):
        ev += e.process(i / FPS, (plan or (lambda i, t: inzone()))(i, i / FPS))
    return ev


class TestConfidence(unittest.TestCase):
    def test_below_threshold_never_alerts(self):           # (4)
        e = eng()
        run(e, lambda i, t: inzone(0.4), n=80)
        e.flush()
        self.assertEqual(cats(e), [CAT_LOWCONF])
        self.assertEqual(e.counters["emitted_alerts"], 0)
        self.assertIn("too uncertain", e.incidents[0]["reason"])

    def test_threshold_boundary_inclusive_and_configurable(self):
        e = eng(confidence_threshold=0.6)
        run(e, lambda i, t: inzone(0.6), n=30)
        self.assertEqual(e.counters["emitted_alerts"], 1)
        e2 = eng(confidence_threshold=0.7)
        run(e2, lambda i, t: inzone(0.6), n=30)
        self.assertEqual(e2.counters["emitted_alerts"], 0)

    def test_short_low_conf_blip_counted_not_logged(self):
        e = eng()
        run(e, lambda i, t: inzone(0.3) if i < 2 else [], n=30)
        e.flush()
        self.assertEqual(e.incidents, [])
        self.assertEqual(e.counters["ignored_low_conf_blips"], 1)


class TestMissingDetections(unittest.TestCase):
    def test_short_dropout_does_not_split_alert(self):     # (5)
        e = eng()
        run(e, lambda i, t: [] if 40 <= i < 46 else inzone(), n=100)   # 0.6 s gap < 0.8 grace
        e.flush()
        self.assertEqual(e.counters["emitted_alerts"], 1)
        self.assertIn("6 missing", e.incidents[0]["reason"])

    def test_long_dropout_ends_alert(self):
        e = eng(dedup_cooldown_s=0.0)
        run(e, lambda i, t: [] if 40 <= i < 60 else inzone(), n=120)   # 2 s gap
        e.flush()
        self.assertEqual(e.counters["emitted_alerts"], 2)

    def test_low_conf_frames_inside_episode_do_not_break_it(self):
        e = eng()
        run(e, lambda i, t: inzone(0.3) if i % 10 == 5 else inzone(), n=80)
        e.flush()
        self.assertEqual(e.counters["emitted_alerts"], 1)
        self.assertIn("low-confidence", e.incidents[0]["reason"])


class TestDedup(unittest.TestCase):
    def test_reentry_within_cooldown_merged(self):         # (6)
        e = eng()
        # in 0-3 s, out 3-4.5 s (1.5 s > grace, < cooldown), in again 4.5-8 s
        run(e, lambda i, t: inzone() if (t < 3 or t >= 4.5) and t < 8 else [], n=100)
        e.flush()
        self.assertEqual(e.counters["emitted_alerts"], 1)
        self.assertEqual(e.counters["merged_reentries"], 1)
        self.assertEqual(e.incidents[0]["merged_reentries"], 1)
        self.assertIn("merged into this incident", e.incidents[0]["reason"])

    def test_reentry_after_cooldown_new_alert(self):
        e = eng(dedup_cooldown_s=1.0)
        run(e, lambda i, t: inzone() if (t < 3 or t >= 6) and t < 10 else [], n=130)
        e.flush()
        self.assertEqual(e.counters["emitted_alerts"], 2)

    def test_two_people_get_two_alerts(self):
        e = eng()
        run(e, lambda i, t: inzone(tid=1) + inzone(tid=2), n=40)
        self.assertEqual(len(e.active_alerts()), 2)


class TestBaselineVsProposed(unittest.TestCase):
    def test_baseline_alerts_immediately_proposed_waits(self):   # (10)
        b = HazardEventEngine(EngineConfig().as_baseline(), "B", FIXED)
        p = eng()
        for i in range(5):
            b.process(i / FPS, inzone()); p.process(i / FPS, inzone())
        self.assertEqual(b.counters["emitted_alerts"], 1)
        self.assertEqual(p.counters["emitted_alerts"], 0)

    def test_flicker_fragments_baseline_not_proposed(self):
        plan = lambda i, t: [] if i % 20 in (10, 11, 12, 13) else inzone()   # 0.4 s dropouts
        b = HazardEventEngine(EngineConfig().as_baseline(), "B", FIXED)
        p = eng()
        for i in range(100):
            b.process(i / FPS, plan(i, 0)); p.process(i / FPS, plan(i, 0))
        b.flush(); p.flush()
        self.assertGreaterEqual(b.counters["emitted_alerts"], 4)
        self.assertEqual(p.counters["emitted_alerts"], 1)

    def test_transient_graze_baseline_alerts_proposed_suppresses(self):
        b = HazardEventEngine(EngineConfig().as_baseline(), "B", FIXED)
        p = eng()
        for i in range(30):
            o = inzone() if 5 <= i < 11 else []
            b.process(i / FPS, o); p.process(i / FPS, o)
        b.flush(); p.flush()
        self.assertEqual((b.counters["emitted_alerts"], p.counters["emitted_alerts"]), (1, 0))
        self.assertEqual(p.counters["suppressed_transient"], 1)


class TestEngineValidation(unittest.TestCase):
    def test_invalid_config(self):                         # (9)
        for kw in (dict(confidence_threshold=1.2), dict(persistence_s=-1), dict(method="x"),
                   dict(exit_grace_s="a")):
            with self.assertRaises(ConfigError):
                HazardEventEngine(EngineConfig(**kw))

    def test_non_monotonic_time_rejected(self):
        e = eng(); e.process(1.0, [])
        with self.assertRaises(ValueError):
            e.process(0.5, [])

    def test_deterministic(self):
        def go():
            e = eng(); run(e, lambda i, t: inzone() if i % 17 else [], n=100); e.flush()
            return [(i["category"], round(i["start_ts"], 3), round(i["end_ts"], 3)) for i in e.incidents]
        self.assertEqual(go(), go())


if __name__ == "__main__":
    unittest.main()
