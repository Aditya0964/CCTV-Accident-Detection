"""Source-video grouping shared by dataset preparation and model training."""

from __future__ import annotations

import re
from pathlib import Path


def infer_video_group(path: Path) -> str:
    """Group adjacent clips from the same original recording."""
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    if "_clip_" in stem:
        return stem.split("_clip_", 1)[0]
    if "_seg_" in stem:
        return stem.split("_seg_", 1)[0]
    numbered_clip = re.fullmatch(
        r"(v\d+|\d+|RoadAccidents\d+_x264)_\d+", stem, flags=re.IGNORECASE
    )
    if numbered_clip:
        return numbered_clip.group(1).lower()
    return stem
