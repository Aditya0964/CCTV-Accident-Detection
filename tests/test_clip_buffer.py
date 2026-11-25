"""Unit tests for rolling frame buffer, clip extraction, and CLI integration."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.clip_buffer import RollingFrameBuffer, compute_resized_dimensions, extract_candidate_clips
from src.detect_anomalies import parse_args, run
from src.motion_anomaly import CandidateEvent


def _create_synthetic_video(path: Path, width: int = 320, height: int = 240, fps: float = 10.0, num_frames: int = 60) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create synthetic video at {path}")
    try:
        for idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (30, 30, 30)
            cv2.putText(frame, f"F{idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


class TestClipBufferAndCLI(unittest.TestCase):
    def test_rolling_buffer_capacity_and_slice(self) -> None:
        buf = RollingFrameBuffer(max_duration_sec=2.0, fps=10.0)
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        for idx in range(50):
            buf.push(idx, idx / 10.0, dummy)

        self.assertLessEqual(len(buf), 22)
        window = buf.slice_window(3.5, 4.9)
        self.assertEqual([f.frame_index for f in window], list(range(35, 50)))

    def test_resize_dimensions_matches_phase1_even_height(self) -> None:
        w, h = compute_resized_dimensions(1920, 1080, 960)
        self.assertEqual((w, h), (960, 540))
        w2, h2 = compute_resized_dimensions(1000, 503, 960)
        self.assertEqual(w2, 960)
        self.assertEqual(h2 % 2, 0)

    def test_extract_candidate_clips_with_boundary_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "sample.mp4"
            _create_synthetic_video(video_path, width=320, height=240, fps=10.0, num_frames=50)

            cand_early = CandidateEvent(
                candidate_id=1,
                timestamp_sec=0.5,
                frame_index=5,
                clip_start_sec=0.0,
                clip_end_sec=2.5,
                involved_track_ids=[1, 2],
                involved_classes=["car", "car"],
                candidate_region=[80.0, 60.0, 200.0, 140.0],
                motion_score=7,
                max_score=9,
                evidence={"sudden_speed_drop": True},
                metrics={},
            )
            cand_late = CandidateEvent(
                candidate_id=2,
                timestamp_sec=4.5,
                frame_index=45,
                clip_start_sec=2.5,
                clip_end_sec=6.5,
                involved_track_ids=[3],
                involved_classes=["truck"],
                candidate_region=[100.0, 80.0, 220.0, 160.0],
                motion_score=6,
                max_score=9,
                evidence={"sudden_speed_drop": True},
                metrics={},
            )

            extract_candidate_clips(
                video_path=video_path,
                candidates=[cand_early, cand_late],
                output_dir=tmp_dir,
                target_width=320,
                before_sec=2.0,
                after_sec=2.0,
            )

            self.assertTrue(Path(cand_early.clip_path).is_file())
            self.assertTrue(Path(cand_early.keyframe_path).is_file())
            self.assertAlmostEqual(cand_early.clip_start_sec, 0.0, places=2)
            self.assertAlmostEqual(cand_early.clip_end_sec, 2.5, places=2)

            self.assertTrue(Path(cand_late.clip_path).is_file())
            self.assertTrue(Path(cand_late.keyframe_path).is_file())
            self.assertAlmostEqual(cand_late.clip_start_sec, 2.5, places=2)
            self.assertAlmostEqual(cand_late.clip_end_sec, 4.9, places=2)

            cap = cv2.VideoCapture(cand_early.clip_path)
            self.assertTrue(cap.isOpened())
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(count, 26)  # frames 0..25 inclusive

    def test_detect_anomalies_cli_end_to_end(self) -> None:
        fps = 10.0
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "traffic.mp4"
            tracks_csv = tmp_dir / "tracks.csv"
            out_dir = tmp_dir / "out"
            _create_synthetic_video(video_path, width=320, height=240, fps=fps, num_frames=50)

            with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame",
                        "timestamp_sec",
                        "track_id",
                        "class_id",
                        "class_name",
                        "confidence",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "center_x",
                        "center_y",
                    ]
                )
                for f in range(20):
                    t = f / fps
                    cx1 = 60.0 + f * 8.0
                    cy1 = 120.0
                    cx2 = 220.0
                    cy2 = 20.0 + f * 5.0
                    writer.writerow([f, f"{t:.3f}", 4, 2, "car", "0.9", cx1 - 25, cy1 - 15, cx1 + 25, cy1 + 15, cx1, cy1])
                    writer.writerow([f, f"{t:.3f}", 9, 7, "truck", "0.9", cx2 - 30, cy2 - 20, cx2 + 30, cy2 + 20, cx2, cy2])
                for f in range(20, 50):
                    t = f / fps
                    cx1, cy1 = 220.0, 130.0
                    cx2, cy2 = 222.0, 122.0
                    writer.writerow([f, f"{t:.3f}", 4, 2, "car", "0.9", cx1 - 25, cy1 - 15, cx1 + 25, cy1 + 15, cx1, cy1])
                    writer.writerow([f, f"{t:.3f}", 9, 7, "truck", "0.9", cx2 - 30, cy2 - 20, cx2 + 30, cy2 + 20, cx2, cy2])

            args = parse_args(
                [
                    "--tracks",
                    str(tracks_csv),
                    "--output-dir",
                    str(out_dir),
                    "--video",
                    str(video_path),
                    "--width",
                    "320",
                    "--extract-clips",
                    "--render-video",
                ]
            )
            run(args)

            candidates = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(candidates), 1)
            self.assertGreaterEqual(candidates[0]["motion_score"], 5)
            self.assertTrue((out_dir / "motion_scores.csv").is_file())
            self.assertTrue((out_dir / "anomaly_preview.mp4").is_file())
            self.assertTrue(Path(candidates[0]["clip_path"]).is_file())
            self.assertTrue(Path(candidates[0]["keyframe_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
