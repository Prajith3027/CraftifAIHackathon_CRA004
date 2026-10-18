import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.zone_manager import Zone, ZoneError, point_in_polygon
from src.config import ConfigError, load_config, validate_config, save_user_config
import copy, tempfile, os

SQ = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]


class TestZone(unittest.TestCase):
    def test_point_in_polygon(self):                       # (1)
        self.assertTrue(point_in_polygon(0.75, 0.75, SQ))
        self.assertFalse(point_in_polygon(0.25, 0.75, SQ))
        self.assertTrue(point_in_polygon(0.5, 0.75, SQ))   # boundary counts as inside
        tri = [(0, 0), (1, 0), (0, 1)]
        self.assertTrue(point_in_polygon(0.2, 0.2, tri))
        self.assertFalse(point_in_polygon(0.8, 0.8, tri))

    def test_foot_point_uses_bottom_centre(self):
        z = Zone("z", SQ)
        # box whose centre is outside the zone but whose feet are inside
        bbox = (400, 100, 440, 360 * 0.75 + 0)  # frame 640x360
        self.assertEqual(Zone.foot_point((10, 20, 30, 80)), (20.0, 80))
        self.assertTrue(z.contains_foot((400, 120, 480, 290), 640, 360))   # foot (440,290)->(.69,.81)
        self.assertFalse(z.contains_foot((100, 120, 180, 290), 640, 360))

    def test_enter_and_leave(self):                         # (2)
        z = Zone("z", SQ)
        path = [(0.2, 0.8), (0.45, 0.8), (0.55, 0.8), (0.8, 0.8), (0.4, 0.8)]
        self.assertEqual([z.contains_normalized(*p) for p in path], [False, False, True, True, False])

    def test_invalid_zones(self):                          # (9)
        for bad in ([(0, 0), (1, 1)], [(0, 0), (2, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
                    [(0, 0), (1, 1), (1, 0), (0, 1)], "abc", [("a", 1), (2, 3), (4, 5)]):
            with self.assertRaises(ZoneError, msg=str(bad)):
                Zone("bad", bad)


class TestConfig(unittest.TestCase):
    def test_default_valid(self):
        self.assertEqual(validate_config(load_config(use_user=False)), [])

    def test_invalid_values_reported(self):                # (9)
        cfg = load_config(use_user=False)
        for key, val in (("confidence_threshold", 1.5), ("persistence_s", -1), ("detector", "magic"),
                         ("frame_stride", "x")):
            c = copy.deepcopy(cfg); c[key] = val
            self.assertTrue(validate_config(c), key)
        c = copy.deepcopy(cfg); c["zone"]["polygon"] = [[0, 0], [1, 1]]
        self.assertTrue(any("zone" in e for e in validate_config(c)))

    def test_malformed_file_and_refuse_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json"); open(p, "w").write("{not json")
            with self.assertRaises(ConfigError):
                load_config(p)
            c = load_config(use_user=False); c["persistence_s"] = -5
            with self.assertRaises(ConfigError):
                save_user_config(c, os.path.join(d, "u.json"))


if __name__ == "__main__":
    unittest.main()
