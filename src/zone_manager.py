"""Restricted-zone geometry. Polygons are stored in normalised (0-1) coordinates so
they are independent of video resolution."""
from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
EPS = 1e-9


class ZoneError(ValueError):
    """Raised for an invalid zone definition."""


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    """Ray casting. Points on the boundary count as inside."""
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) <= EPS and min(x1, x2) - EPS <= x <= max(x1, x2) + EPS \
                and min(y1, y2) - EPS <= y <= max(y1, y2) + EPS:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _polygon_area(poly: Sequence[Point]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _seg_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < EPS else (1 if v > 0 else -1)
    return orient(p1, p2, p3) != orient(p1, p2, p4) and orient(p3, p4, p1) != orient(p3, p4, p2)


class Zone:
    def __init__(self, name: str, polygon: Sequence[Sequence[float]]):
        try:
            poly = [(float(p[0]), float(p[1])) for p in polygon]
        except (TypeError, ValueError, IndexError) as e:
            raise ZoneError("polygon must be a list of [x, y] number pairs") from e
        if len(poly) < 3:
            raise ZoneError("polygon needs at least 3 vertices")
        for x, y in poly:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ZoneError(f"vertex ({x}, {y}) outside normalised range [0, 1]")
        if _polygon_area(poly) < 1e-4:
            raise ZoneError("polygon area is (almost) zero")
        n = len(poly)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue  # adjacent edges
                if _seg_intersect(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                    raise ZoneError("polygon edges cross each other (self-intersecting)")
        self.name = name
        self.polygon: List[Point] = poly

    @staticmethod
    def foot_point(bbox: BBox) -> Point:
        """Approximate ground contact: bottom-centre of the bounding box (pixels)."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, max(y1, y2))

    def contains_normalized(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def contains_foot(self, bbox: BBox, frame_w: int, frame_h: int) -> bool:
        if frame_w <= 0 or frame_h <= 0:
            raise ZoneError("frame size must be positive")
        fx, fy = self.foot_point(bbox)
        return self.contains_normalized(fx / frame_w, fy / frame_h)

    def to_pixels(self, w: int, h: int) -> List[Tuple[int, int]]:
        return [(int(round(x * w)), int(round(y * h))) for x, y in self.polygon]
