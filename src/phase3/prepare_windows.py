"""Export full-frame video windows for a later accident classifier."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from .windows import build_windows


def read_video_info(video: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid video metadata: {video}")
        count = 0
        while capture.read()[0]:
            count += 1
        if count == 0:
            raise ValueError(f"Video contains no decodable frames: {video}")
        return count, fps, width, height
    finally:
        capture.release()


def write_clips(video: Path, output_dir: Path, windows: list[dict], fps: float, width: int, height: int) -> None:
    starts: dict[int, list[dict]] = defaultdict(list)
    for window in windows:
        starts[window["start_frame"]].append(window)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot reopen video: {video}")
    active: dict[str, tuple[dict, cv2.VideoWriter]] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for window in starts.get(frame_index, []):
                path = output_dir / f"{window['id']}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create clip: {path}")
                window["clip_path"] = str(path.resolve())
                active[window["id"]] = (window, writer)
            for window, writer in active.values():
                writer.write(frame)
            frame_index += 1
            for key, (window, writer) in list(active.items()):
                if frame_index >= window["end_frame_exclusive"]:
                    writer.release()
                    del active[key]
        expected = max(window["end_frame_exclusive"] for window in windows)
        if frame_index < expected:
            raise RuntimeError(f"Video ended after {frame_index} frames, expected {expected}")
    finally:
        capture.release()
        for _, writer in active.values():
            writer.release()


def prepare(
    video: Path,
    output_dir: Path,
    candidates_path: Path | None = None,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    candidates = []
    if candidates_path is not None:
        with candidates_path.open(encoding="utf-8") as stream:
            candidates = json.load(stream)
        if not isinstance(candidates, list):
            raise ValueError("Candidates JSON must be a list")

    frame_count, fps, width, height = read_video_info(video)
    windows = build_windows(frame_count, fps, candidates, window_seconds, stride_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_clips(video, output_dir, windows, fps, width, height)
    manifest = {
        "source_video": str(video.resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": round(frame_count / fps, 6),
        "coverage": "every decodable frame occurs in at least one full-frame window",
        "motion_candidates": candidates,
        "windows": windows,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original, unannotated source video")
    parser.add_argument("--candidates", type=Path, help="Optional Phase 2 candidates.json")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/phase3/<video name>")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("outputs/phase3") / args.input.stem
    manifest = prepare(args.input, output_dir, args.candidates, args.window_seconds, args.stride_seconds)
    print(f"Exported {len(manifest['windows'])} windows to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print("All decodable frames are covered; accident classification is not implemented here.")


if __name__ == "__main__":
    main()
