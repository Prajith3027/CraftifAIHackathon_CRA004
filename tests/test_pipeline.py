import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, tempfile
import numpy as np
import cv2
from src.config import load_config
from src.detector import Detection, HOGPersonDetector, SyntheticDetector, build_detector
from src.incident_store import IncidentStore
from src.pipeline import MonitoringSession, run_source
from src.synthetic import SyntheticSource, mixed_scenario, save_dataset, load_dataset
from src.video_source import SourceError, VideoFileSource, WebcamSource
from src.tracking import IoUTracker
from src.system_status import get_system_status


class TestPipeline(unittest.TestCase):
    def test_synthetic_pipeline_end_to_end(self):
        cfg = load_config(use_user=False)
        sc = mixed_scenario()
        store = IncidentStore(":memory:")
        sess = MonitoringSession(cfg, SyntheticDetector(sc.frames), "synthetic", store, "mixed")
        results = list(run_source(SyntheticSource(sc), sess, frame_stride=1))
        summ = sess.finish()
        self.assertEqual(len(results), len(sc.frames))
        self.assertEqual(results[0].annotated.shape, (sc.height, sc.width, 3))
        self.assertTrue(any(r.state == "ALERT" for r in results))
        self.assertEqual(summ["frames_processed"], len(sc.frames))
        self.assertEqual(len(store.list(method="proposed", category="rule_alert")), 2)
        self.assertEqual(len(store.runs()), 1)
        self.assertTrue(all(i["source_type"] == "synthetic" for i in store.list()))

    def test_video_file_source_and_hog_runs(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.avi")
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            for _ in range(20):
                vw.write(np.zeros((240, 320, 3), np.uint8))
            vw.release()
            src = VideoFileSource(p)
            self.assertEqual((src.width, src.height), (320, 240))
            det, msg = build_detector("hog")
            self.assertIsNotNone(det, msg)
            sess = MonitoringSession(load_config(use_user=False), det, "video_file")
            res = list(run_source(src, sess, frame_stride=2))
            self.assertEqual(len(res), 10)
            self.assertEqual(sess.finish()["frames_processed"], 10)

    def test_error_handling(self):
        with self.assertRaises(SourceError):
            VideoFileSource("/no/such/video.mp4")
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.mp4"); open(bad, "w").write("not a video")
            with self.assertRaises(SourceError):
                VideoFileSource(bad)
        det, msg = build_detector("yolo", "models/missing.pt")
        self.assertIsNone(det)
        self.assertIn("YOLO unavailable", msg)

    def test_auto_detector_falls_back(self):
        det, msg = build_detector("auto")
        self.assertIsNotNone(det)

    def test_dataset_roundtrip(self):
        sc = mixed_scenario()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json"); save_dataset(sc, p)
            sc2 = load_dataset(p)
        self.assertEqual(len(sc2.frames), len(sc.frames))
        self.assertEqual(sc2.ground_truth, sc.ground_truth)

    def test_tracker_keeps_id(self):
        t = IoUTracker()
        a = t.update([Detection((0, 0, 10, 20), 0.9)])[0].track_id
        b = t.update([Detection((1, 0, 11, 20), 0.9)])[0].track_id
        self.assertEqual(a, b)
        c = t.update([Detection((200, 200, 210, 220), 0.9)])[0].track_id
        self.assertNotEqual(a, c)

    def test_system_status_is_honest(self):
        rows = {r["component"]: r for r in get_system_status(None, "none")}
        self.assertEqual(rows["CraftifAI PipeGen"]["state"], "NOT COMPLETED")
        self.assertEqual(rows["Synthetic scenario generator"]["state"], "SIMULATED")


if __name__ == "__main__":
    unittest.main()
