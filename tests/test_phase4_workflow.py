import unittest

from src.confirm_accidents import localized_review_candidate
from src.run_phase4 import incident_evidence


class TestDetectorHandoff(unittest.TestCase):
    def test_localized_review_requires_contact_and_a_motion_response(self):
        anchor = {"timestamp_sec": 1.3, "candidate_region": [3, 5, 9, 10]}
        candidate = {"timestamp_sec": 3.0, "clip_start_sec": 1.0, "clip_end_sec": 4.9,
                     "metrics": {"contact_anchor": anchor},
                     "evidence": {"bounding_box_overlap": True, "post_interaction_stopping": True}}
        self.assertIsNone(localized_review_candidate(candidate))
        candidate["evidence"]["sudden_speed_drop"] = True
        reviewed = localized_review_candidate(candidate)
        self.assertEqual(reviewed["timestamp_sec"], 1.3)
        self.assertEqual(reviewed["candidate_region"], anchor["candidate_region"])
        self.assertEqual(reviewed["clip_end_sec"], 4.9)
        candidate["clip_start_sec"] = 1.5
        self.assertIsNone(localized_review_candidate(candidate))

    def test_localized_vit_incidents_are_used_even_with_a_negative_clip_verdict(self):
        localized = [{"incident_id": "001", "timestamp_sec": 1.0, "candidate_region": [3, 5, 9, 10]}]
        clip = {"predicted_accident": False, "p_accident": 0.2, "peak_time_sec": 2.0}
        self.assertEqual(incident_evidence(clip, localized), localized)
        self.assertEqual(incident_evidence(clip, []), [])

    def test_vit_clip_only_signal_uses_scene_wide_scan_with_approximate_time(self):
        clip = {"predicted_accident": True, "p_accident": 0.8, "peak_time_sec": 1.4}
        [row] = incident_evidence(clip, [])
        self.assertTrue(row["scan_full_frame"])
        self.assertTrue(row["timestamp_is_approximate"])
        self.assertEqual(row["timestamp_sec"], 1.4)
        self.assertIsNone(row["candidate_region"])

    def test_fusion_uses_source_pixel_box_only_when_detector_scores_support_it(self):
        clip = {"predicted_accident": True, "fusion_score": 0.7, "peak_detector_time_sec": 1.8,
                "peak_detector_box_xyxy": [10, 20, 100, 150], "peak_detector_box_score": 0.81}
        [row] = incident_evidence(clip, [])
        self.assertEqual(row["region_space"], "source")
        self.assertEqual(row["candidate_region"], clip["peak_detector_box_xyxy"])
        self.assertEqual(row["timestamp_sec"], 1.8)
        self.assertFalse(row["scan_full_frame"])

        clip["peak_detector_box_score"] = 0.05
        [fallback] = incident_evidence(clip, [])
        self.assertEqual(fallback["timestamp_sec"], 0)
        self.assertEqual(fallback["timestamp_source"], "clip_start")
        self.assertTrue(fallback["scan_full_frame"])
