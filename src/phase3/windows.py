"""Plan overlapping windows without depending on vehicle tracks or OpenCV."""
from __future__ import annotations

import math


def build_windows(
    frame_count: int,
    fps: float,
    candidates: list[dict] | None = None,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> list[dict]:
    """Cover every frame, then add windows centred on motion candidate times.

    Coverage windows are always present, including when ByteTrack finds no
    vehicles or Phase 2 emits no candidates. This guarantees input coverage,
    not accident-classification recall.
    """
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("A video needs positive frame count and FPS")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not math.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    width = min(frame_count, max(1, round(window_seconds * fps)))
    stride = max(1, round(stride_seconds * fps))
    if stride > width and width < frame_count:
        raise ValueError("stride_seconds must not leave gaps between windows")

    max_start = frame_count - width
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)

    planned: dict[int, dict] = {}

    def add(start: int, reason: str, candidate_index: int | None = None) -> None:
        window = planned.setdefault(
            start,
            {
                "start_frame": start,
                "end_frame_exclusive": start + width,
                "reasons": [],
                "candidate_indices": [],
            },
        )
        if reason not in window["reasons"]:
            window["reasons"].append(reason)
        if candidate_index is not None and candidate_index not in window["candidate_indices"]:
            window["candidate_indices"].append(candidate_index)

    for start in starts:
        add(start, "full_frame_coverage")

    duration = frame_count / fps
    for index, candidate in enumerate(candidates or []):
        for key in ("candidate_timestamp_sec", "decision_timestamp_sec"):
            if key not in candidate:
                continue
            timestamp = float(candidate[key])
            if not math.isfinite(timestamp) or not 0 <= timestamp <= duration:
                raise ValueError(f"Candidate {index} has an invalid {key}: {timestamp}")
            centre = min(frame_count - 1, round(timestamp * fps))
            start = min(max_start, max(0, centre - width // 2))
            add(start, "motion_candidate", index)

    windows = []
    for number, window in enumerate(sorted(planned.values(), key=lambda item: item["start_frame"]), 1):
        window["id"] = f"window_{number:04d}"
        window["start_sec"] = round(window["start_frame"] / fps, 6)
        window["end_sec"] = round(window["end_frame_exclusive"] / fps, 6)
        windows.append(window)
    return windows
