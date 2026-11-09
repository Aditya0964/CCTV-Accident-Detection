"""Vehicle-pair proximity and IoU using current tracked boxes."""
from __future__ import annotations

from itertools import combinations
from math import hypot
from .trajectory import Observation


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def nearby_pairs(observations: list[Observation], cfg: dict):
    for a, b in combinations(observations, 2):
        distance = hypot(a.cx-b.cx, a.cy-b.cy)
        size = (a.diagonal + b.diagonal) / 2
        normalized = distance / max(size, 1e-9)
        overlap = iou(a.box, b.box)
        close = normalized < cfg["proximity_threshold"]
        overlapping = overlap > cfg["iou_threshold"]
        if close or overlapping:
            region = [min(a.x1,b.x1), min(a.y1,b.y1), max(a.x2,b.x2), max(a.y2,b.y2)]
            yield (min(a.track_id,b.track_id),max(a.track_id,b.track_id)), close, overlapping, region, normalized, overlap
