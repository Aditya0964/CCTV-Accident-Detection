"""Stage 1 & 8: Rolling frame buffer, candidate clip extraction, and anomaly preview rendering."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.motion_anomaly import CandidateEvent, FrameTrackScore


@dataclass(frozen=True)
class BufferedFrame:
    """Single resized video frame stored in the rolling temporal buffer."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


class RollingFrameBuffer:
    """Fixed-horizon rolling buffer for extracting [t - before, t + after] clips."""

    def __init__(self, max_duration_sec: float, fps: float) -> None:
        if max_duration_sec <= 0 or fps <= 0:
            raise ValueError("max_duration_sec and fps must be positive")
        self.max_duration_sec = max_duration_sec
        self.fps = fps
        capacity = max(2, math.ceil(max_duration_sec * fps) + 2)
        self._buffer: deque[BufferedFrame] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, frame_index: int, timestamp_sec: float, image: np.ndarray) -> None:
        self._buffer.append(
            BufferedFrame(frame_index=frame_index, timestamp_sec=timestamp_sec, image=image)
        )

    def slice_window(self, start_sec: float, end_sec: float) -> list[BufferedFrame]:
        """Return buffered frames within [start_sec, end_sec] in chronological order."""
        return [
            item
            for item in self._buffer
            if start_sec - 1e-6 <= item.timestamp_sec <= end_sec + 1e-6
        ]


def compute_resized_dimensions(
    input_width: int, input_height: int, target_width: int
) -> tuple[int, int]:
    """Match Phase 1 resize calculation in src/track_vehicles.py."""
    if input_width <= 0 or input_height <= 0 or target_width <= 0:
        raise ValueError("Video dimensions and target_width must be positive")
    height = max(2, round(input_height * target_width / input_width))
    height += height % 2
    return target_width, height


def _open_video_metadata(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata in {video_path}: fps={fps}, {width}x{height}")
    return cap, fps, width, height


def extract_candidate_clips(
    video_path: Path,
    candidates: list[CandidateEvent],
    output_dir: Path,
    target_width: int = 960,
    before_sec: float = 2.0,
    after_sec: float = 2.0,
    draw_region: bool = True,
) -> list[CandidateEvent]:
    """Extract [t - before_sec, t + after_sec] clips and peak keyframes for each candidate."""
    if not candidates:
        return candidates

    cap, fps, in_w, in_h = _open_video_metadata(video_path)
    out_w, out_h = compute_resized_dimensions(in_w, in_h, target_width)
    clips_dir = output_dir / "candidates"
    clips_dir.mkdir(parents=True, exist_ok=True)

    buffer = RollingFrameBuffer(max_duration_sec=before_sec + after_sec + 1.0, fps=fps)
    pending = sorted(candidates, key=lambda c: c.timestamp_sec)
    written_ids: set[int] = set()
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (out_w, out_h))
            ts = frame_index / fps
            buffer.push(frame_index=frame_index, timestamp_sec=ts, image=resized)

            for cand in pending:
                if cand.candidate_id in written_ids:
                    continue
                target_end = cand.timestamp_sec + after_sec
                if ts >= target_end - (0.5 / fps):
                    _flush_candidate_clip(
                        cand, buffer, clips_dir, fps, (out_w, out_h), before_sec, after_sec, draw_region
                    )
                    written_ids.add(cand.candidate_id)

            frame_index += 1
            if len(written_ids) == len(pending):
                break

        # Flush any candidates whose post-window extends past the end of the video.
        for cand in pending:
            if cand.candidate_id not in written_ids:
                _flush_candidate_clip(
                    cand, buffer, clips_dir, fps, (out_w, out_h), before_sec, after_sec, draw_region
                )
                written_ids.add(cand.candidate_id)
    finally:
        cap.release()

    return candidates


def _flush_candidate_clip(
    candidate: CandidateEvent,
    buffer: RollingFrameBuffer,
    clips_dir: Path,
    fps: float,
    frame_size: tuple[int, int],
    before_sec: float,
    after_sec: float,
    draw_region: bool,
) -> None:
    start_sec = max(0.0, candidate.timestamp_sec - before_sec)
    end_sec = candidate.timestamp_sec + after_sec
    frames = buffer.slice_window(start_sec, end_sec)
    if not frames:
        return

    clip_path = clips_dir / f"candidate_{candidate.candidate_id:03d}.mp4"
    keyframe_path = clips_dir / f"candidate_{candidate.candidate_id:03d}.jpg"

    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create candidate clip: {clip_path}")

    best_frame: Optional[BufferedFrame] = None
    best_dist = float("inf")
    try:
        for item in frames:
            canvas = item.image.copy() if draw_region else item.image
            if draw_region:
                _draw_candidate_overlay(canvas, candidate, frame_size)
            writer.write(canvas)
            dist = abs(item.frame_index - candidate.frame_index)
            if dist < best_dist:
                best_dist = dist
                best_frame = BufferedFrame(item.frame_index, item.timestamp_sec, canvas)
    finally:
        writer.release()

    if best_frame is not None:
        cv2.imwrite(str(keyframe_path), best_frame.image)
        candidate.keyframe_path = str(keyframe_path)
    candidate.clip_path = str(clip_path)
    candidate.clip_start_sec = round(frames[0].timestamp_sec, 3)
    candidate.clip_end_sec = round(frames[-1].timestamp_sec, 3)


def _draw_candidate_overlay(
    image: np.ndarray,
    candidate: CandidateEvent,
    frame_size: tuple[int, int],
) -> None:
    w, h = frame_size
    x1, y1, x2, y2 = candidate.candidate_region
    px1 = max(0, min(w - 1, int(round(x1))))
    py1 = max(0, min(h - 1, int(round(y1))))
    px2 = max(0, min(w - 1, int(round(x2))))
    py2 = max(0, min(h - 1, int(round(y2))))
    cv2.rectangle(image, (px1, py1), (px2, py2), (0, 0, 255), 2)
    label = (
        f"Candidate #{candidate.candidate_id} | Score {candidate.motion_score}/{candidate.max_score}"
        f" | t={candidate.timestamp_sec:.2f}s"
    )
    cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def render_anomaly_video(
    video_path: Path,
    frame_scores: list[FrameTrackScore],
    candidates: list[CandidateEvent],
    output_path: Path,
    target_width: int = 960,
) -> Path:
    """Render an annotated debug video with per-vehicle scores, interaction links, and banners."""
    cap, fps, in_w, in_h = _open_video_metadata(video_path)
    out_w, out_h = compute_resized_dimensions(in_w, in_h, target_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create anomaly preview video: {output_path}")

    scores_by_frame: dict[int, list[FrameTrackScore]] = defaultdict(list)
    for fs in frame_scores:
        scores_by_frame[fs.frame].append(fs)

    status_colors = {
        "normal": (0, 200, 0),
        "suspicious": (0, 165, 255),
        "accident_candidate": (0, 0, 255),
    }

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (out_w, out_h))
            ts = frame_index / fps
            entries = scores_by_frame.get(frame_index, [])

            peak_score = 0
            for fs in entries:
                peak_score = max(peak_score, fs.motion_score)
                color = status_colors.get(fs.status, (0, 200, 0))
                x1, y1, x2, y2 = (int(round(v)) for v in fs.box)
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)

                flags = []
                if fs.sudden_speed_drop:
                    flags.append("DROP")
                if fs.sharp_direction_change:
                    flags.append("TURN")
                if fs.very_close_interaction:
                    flags.append("PROX")
                if fs.bounding_box_overlap:
                    flags.append("IOU")
                if fs.post_interaction_stopping:
                    flags.append("STOP")
                flag_txt = f" [{'+'.join(flags)}]" if flags else ""
                text = f"ID {fs.track_id} {fs.speed_px_s:.0f}px/s S:{fs.motion_score}{flag_txt}"
                cv2.putText(
                    resized,
                    text,
                    (max(0, x1), max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                )

                if fs.partner_box is not None and (
                    fs.very_close_interaction or fs.bounding_box_overlap
                ):
                    cx1 = int(round((fs.box[0] + fs.box[2]) * 0.5))
                    cy1 = int(round((fs.box[1] + fs.box[3]) * 0.5))
                    cx2 = int(round((fs.partner_box[0] + fs.partner_box[2]) * 0.5))
                    cy2 = int(round((fs.partner_box[1] + fs.partner_box[3]) * 0.5))
                    cv2.line(resized, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

            active_candidates = [
                c for c in candidates if c.timestamp_sec - 0.5 <= ts <= c.timestamp_sec + 1.5
            ]
            for cand in active_candidates:
                rx1, ry1, rx2, ry2 = (int(round(v)) for v in cand.candidate_region)
                cv2.rectangle(
                    resized,
                    (max(0, rx1), max(0, ry1)),
                    (min(out_w - 1, rx2), min(out_h - 1, ry2)),
                    (0, 0, 255),
                    3,
                )

            hud = f"Time {ts:.2f}s | Peak Score {peak_score}/9"
            if active_candidates:
                ids_str = ",".join(str(c.candidate_id) for c in active_candidates)
                hud += f" | ACCIDENT CANDIDATE #{ids_str}"
            hud_color = (0, 0, 255) if active_candidates else (255, 255, 255)
            cv2.putText(resized, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, hud_color, 2)

            writer.write(resized)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return output_path
