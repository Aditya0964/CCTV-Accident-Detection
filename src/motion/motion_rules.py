"""Individual-vehicle motion indicators; these are not accident decisions."""
from __future__ import annotations

from math import acos, degrees, hypot
from .trajectory import Observation, window_speed


def speed_drop(history: list[Observation], now: float, cfg: dict) -> tuple[bool, float | None, float | None, float | None]:
    recent = cfg["recent_window_seconds"]
    previous = cfg["previous_window_seconds"]
    common = (cfg["max_sample_gap_seconds"], cfg["min_samples_per_window"], cfg["speed_smoothing_samples"])
    before = window_speed(history, now-recent-previous, now-recent, *common)
    after = window_speed(history, now-recent, now + 1e-9, *common)
    if before is None or after is None or before < cfg["min_moving_speed_px_s"]:
        return False, None, before, after
    ratio = after / (before + 1e-9)
    return ratio < cfg["speed_drop_ratio"], ratio, before, after


def _nearest(history: list[Observation], target: float, tolerance: float) -> Observation | None:
    if not history:
        return None
    result = min(history, key=lambda o: abs(o.timestamp - target))
    return result if abs(result.timestamp - target) <= tolerance else None


def direction_change(history: list[Observation], now: float, cfg: dict) -> tuple[bool, float | None]:
    interval = cfg["direction_window_seconds"]
    # Sample positions around fixed times to make angle independent of source FPS.
    tolerance = min(cfg["max_sample_gap_seconds"] / 2, interval / 2)
    a = _nearest(history, now-2*interval, tolerance)
    b = _nearest(history, now-interval, tolerance)
    c = _nearest(history, now, tolerance)
    if a is None or b is None or c is None or len({a.timestamp, b.timestamp, c.timestamp}) < 3:
        return False, None
    u = (b.cx-a.cx, b.cy-a.cy)
    v = (c.cx-b.cx, c.cy-b.cy)
    nu, nv = hypot(*u), hypot(*v)
    if min(nu, nv) < cfg["min_direction_displacement_px"]:
        return False, None
    cosine = max(-1.0, min(1.0, (u[0]*v[0]+u[1]*v[1])/(nu*nv)))
    angle = degrees(acos(cosine))
    return angle > cfg["direction_threshold_deg"], angle


def stationary_after_interaction(history: list[Observation], interaction_at: float,
                                 now: float, cfg: dict) -> bool:
    duration = cfg["stationary_duration_seconds"]
    if now < interaction_at + duration:
        return False
    min_samples = cfg["min_samples_per_window"]
    speed = window_speed(history, now-duration, now+1e-9,
                         cfg["max_sample_gap_seconds"], min_samples,
                         cfg["speed_smoothing_samples"])
    # Must be moving before the encounter to avoid parked-car false signals.
    earlier = window_speed(history, interaction_at-cfg["previous_window_seconds"], interaction_at+1e-9,
                           cfg["max_sample_gap_seconds"], min_samples,
                           cfg["speed_smoothing_samples"])
    return speed is not None and earlier is not None and (speed < cfg["stationary_speed_px_s"]
                                                         and earlier >= cfg["min_moving_speed_px_s"])
