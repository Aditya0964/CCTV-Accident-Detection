"""Checks for video-level calibration without downloading model weights."""

import unittest

from src.vit_workflow import DecisionRule, aggregate, binary_metrics, select_rule


class TestViTDecisionRule(unittest.TestCase):
    def test_metrics_and_aggregation(self) -> None:
        self.assertAlmostEqual(aggregate([0.1, 0.8, 0.9], "top2"), 0.85)
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_rule_is_selected_only_from_validation_scores(self) -> None:
        records = [
            {"label": 1, "frame_probabilities": [0.9, 0.8, 0.7]},
            {"label": 1, "frame_probabilities": [0.8, 0.7, 0.6]},
            {"label": 0, "frame_probabilities": [0.2, 0.1, 0.1]},
            {"label": 0, "frame_probabilities": [0.3, 0.2, 0.1]},
        ]
        rule, metrics = select_rule(records)
        self.assertIsInstance(rule, DecisionRule)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_validation_must_have_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both accident and normal"):
            select_rule([{"label": 1, "frame_probabilities": [0.9]}])


if __name__ == "__main__":
    unittest.main()
