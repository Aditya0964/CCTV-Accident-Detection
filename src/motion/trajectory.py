"""Track history and time-aware speed estimation (coordinates are resized pixels)."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import hypot
from typing import Deque


@dataclass(frozen=True)
class Observation:
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float

    @property
    def diagonal(self) -> float:
        return hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class TrackHistory:
    def __init__(self, history_seconds: float):
        self.history_seconds = history_seconds
        self.tracks: dict[int, Deque[Observation]] = defaultdict(deque)

    def add(self, obs: Observation) -> None:
        trail = self.tracks[obs.track_id]
        if trail and obs.timestamp <= trail[-1].timestamp:
            raise ValueError(f"Non-increasing timestamps for track {obs.track_id}")
        trail.append(obs)
        self.expire(obs.timestamp)

    def expire(self, now: float) -> None:
        for track_id in list(self.tracks):
            trail = self.tracks[track_id]
            while trail and trail[0].timestamp < now - self.history_seconds:
                trail.popleft()
            if not trail:
                del self.tracks[track_id]

    def get(self, track_id: int) -> list[Observation]:
        return list(self.tracks.get(track_id, ()))


def speed_samples(history: list[Observation], max_gap: float) -> list[tuple[float, float]]:
    """Return (interval midpoint time, pixel speed/sec) for reliable consecutive observations."""
    out = []
    for a, b in zip(history, history[1:]):
        dt = b.timestamp - a.timestamp
        if 0 < dt <= max_gap:
            speed = hypot(b.cx - a.cx, b.cy - a.cy) / dt
            out.append(((a.timestamp + b.timestamp) / 2, speed))
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def window_speed(history: list[Observation], start: float, end: float, max_gap: float,
                 min_samples: int, smooth_samples: int = 1) -> float | None:
    """Mean of per-interval speeds with interval midpoints within [start, end)."""
    samples = speed_samples(history, max_gap)
    speeds = [s for t, s in samples if start <= t < end]
    if len(speeds) < min_samples:
        return None
    # Median filter for local tracking jitter; timestamps are not altered.
    if smooth_samples > 1 and len(speeds) >= smooth_samples:
        radius = smooth_samples // 2
        filtered = []
        for i in range(len(speeds)):
            neighborhood = sorted(speeds[max(0, i-radius): min(len(speeds), i+radius+1)])
            filtered.append(neighborhood[len(neighborhood)//2])
        speeds = filtered
    return mean(speeds)
