import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from src.hazard_detector import ScanSettings, scan_video
from src.hazard_preview import render_hazard_preview


class TestHazardScan(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.video = Path(folder.name) / "frames.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 100))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(5):
                writer.write(np.zeros((100, 160, 3), dtype=np.uint8))
        finally:
            writer.release()

    def test_scaled_region_excludes_other_hazards_and_counts_once_per_frame(self):
        fire = ("fire", 0.90, (15, 15, 30, 30))
        smoke_elsewhere = ("smoke", 0.85, (110, 65, 140, 90))
        detector = Mock()
        detector.detect.side_effect = [
            [fire, fire, smoke_elsewhere], [fire, smoke_elsewhere],
            [fire, smoke_elsewhere], [smoke_elsewhere], [smoke_elsewhere],
        ]
        incident = {"incident_id": "incident_01", "timestamp_sec": 0.0,
                    "candidate_region": [20, 20, 80, 80]}
        [result] = scan_video(self.video, [incident], detector,
                              ScanSettings(tracking_width=320, region_padding=0))
        self.assertTrue(result["fire_detected"])
        self.assertFalse(result["smoke_detected"])
        self.assertEqual(result["visual_risk"], "High")
        self.assertEqual(result["hazard_scan"]["hits"], {"fire": 3, "smoke": 0})
        self.assertEqual(result["hazard_scan"]["region_xyxy_source_pixels"], (10, 10, 40, 40))
        self.assertEqual(result["incident_id"], incident["incident_id"])

    def test_one_frame_flash_does_not_pass_persistence(self):
        detector = Mock()
        detector.detect.side_effect = [[("fire", 0.9, (10, 10, 50, 50))]] + [[]] * 4
        [result] = scan_video(self.video, [{"timestamp_sec": 0.0}], detector, ScanSettings(), manual=True)
        self.assertFalse(result["fire_detected"])
        self.assertEqual(result["hazard_scan"]["status"], "scanned")

    def test_source_space_boxes_and_explicit_scene_wide_scan_can_share_a_report(self):
        detector = Mock()
        detector.detect.return_value = [("smoke", 0.9, (10, 10, 30, 30))]
        results = scan_video(self.video, [
            {"timestamp_sec": 0, "candidate_region": [10, 10, 50, 50], "region_space": "source"},
            {"timestamp_sec": 0, "candidate_region": None, "scan_full_frame": True},
        ], detector, ScanSettings(tracking_width=320, region_padding=0))
        self.assertEqual(results[0]["hazard_scan"]["region_xyxy_source_pixels"], (10, 10, 50, 50))
        self.assertEqual(results[1]["hazard_scan"]["scope"], "full_frame")
        self.assertTrue(all(row["smoke_detected"] for row in results))
        self.assertTrue(all(row["visual_risk"] == "Elevated" for row in results))

    def test_missing_region_or_too_few_frames_is_unknown(self):
        detector = Mock()
        detector.detect.return_value = []
        [missing] = scan_video(self.video, [{"timestamp_sec": 0.0}], detector, ScanSettings())
        detector.detect.assert_not_called()
        self.assertEqual(missing["hazard_scan"]["status"], "missing_or_invalid_region")
        self.assertIsNone(missing["fire_detected"])
        self.assertEqual(missing["visual_risk"], "Unknown")
        [short] = scan_video(self.video, [{"timestamp_sec": 0.8}], detector, ScanSettings(), manual=True)
        self.assertEqual(short["hazard_scan"]["status"], "insufficient_frames")
        self.assertIsNone(short["smoke_detected"])

    def test_preview_preserves_frames_and_limits_overlay_to_scan_window(self):
        detector = Mock()
        detector.detect.return_value = [("smoke", 0.9, (10, 10, 50, 50))]
        results = scan_video(self.video, [{"timestamp_sec": 0.2}], detector,
                             ScanSettings(window_sec=0.6), manual=True)
        self.assertEqual(results[0]["visual_risk"], "Elevated")
        preview = self.video.with_name("preview.mp4")
        render_hazard_preview(self.video, results, preview)
        capture = cv2.VideoCapture(str(preview))
        overlay_frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                overlay_frames.append(bool(np.any(frame > 120)))
        finally:
            capture.release()
        self.assertEqual(overlay_frames, [False, True, True, True, False])

    def test_region_only_preview_leaves_road_visible(self):
        detector = Mock()
        detector.detect.return_value = []
        incident = {"timestamp_sec": 0.2, "candidate_region": [60, 50, 130, 85]}
        results = scan_video(self.video, [incident], detector,
                             ScanSettings(window_sec=0.6, tracking_width=160, region_padding=0))
        preview = self.video.with_name("region_only.mp4")
        render_hazard_preview(self.video, results, preview, show_summary=False)
        capture = cv2.VideoCapture(str(preview))
        try:
            self.assertTrue(capture.read()[0])
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertTrue(np.any(frame[48:53, 70:120] > 120))
            self.assertFalse(np.any(frame[:40] > 120))
        finally:
            capture.release()

    def test_accident_preview_marks_only_the_localized_confirmed_box(self):
        detector = Mock()
        detector.detect.return_value = []
        results = scan_video(self.video, [
            {"timestamp_sec": 0.0, "candidate_region": [5, 10, 30, 90], "region_space": "source"},
            {"timestamp_sec": 0.6, "candidate_region": [65, 45, 125, 85], "region_space": "source",
             "confirmation_policy": "vit_impact_review"},
        ], detector, ScanSettings())
        path = self.video.with_name("accident_preview.mp4")
        render_hazard_preview(self.video, results, path, show_summary=False, accident_only=True)
        capture = cv2.VideoCapture(str(path))
        visible = []
        try:
            for _ in range(5):
                ok, image = capture.read()
                self.assertTrue(ok)
                self.assertFalse(np.any(image[:, :32] > 100))
                pixels = image.astype(np.int16)
                visible.append(bool(np.any((pixels[:, :, 2] - pixels[:, :, 1]) > 80)))
        finally:
            capture.release()
        self.assertEqual(visible, [False, False, False, True, True])
