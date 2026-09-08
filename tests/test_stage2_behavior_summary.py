from __future__ import annotations

import unittest

from scripts.summarize_stage2_behavior import behavior_stats


class Stage2BehaviorSummaryTests(unittest.TestCase):
    def test_mixed_set_exposes_overflagging(self) -> None:
        rows = [
            {"gold": "safe", "prediction": "unsafe", "prediction_for_metrics": "unsafe"},
            {"gold": "safe", "prediction": "safe", "prediction_for_metrics": "safe"},
            {"gold": "unsafe", "prediction": "unsafe", "prediction_for_metrics": "unsafe"},
            {"gold": "unsafe", "prediction": "unsafe", "prediction_for_metrics": "unsafe"},
        ]
        result = behavior_stats(rows)
        self.assertEqual(result["Accuracy"], 75.0)
        self.assertEqual(result["Safe-Accuracy"], 50.0)
        self.assertEqual(result["Unsafe-Accuracy"], 100.0)
        self.assertEqual(result["Balanced-Accuracy"], 75.0)
        self.assertEqual(result["FP-safe-as-unsafe"], 1)
        self.assertEqual(result["Pred-Unsafe-Parsed"], 75.0)

    def test_all_unsafe_accuracy_equals_unsafe_accuracy(self) -> None:
        rows = [
            {"gold": "unsafe", "prediction": "unsafe", "prediction_for_metrics": "unsafe"},
            {"gold": "unsafe", "prediction": "safe", "prediction_for_metrics": "safe"},
        ]
        result = behavior_stats(rows)
        self.assertEqual(result["Accuracy"], 50.0)
        self.assertEqual(result["Unsafe-Accuracy"], 50.0)
        self.assertIsNone(result["Safe-Accuracy"])
        self.assertIsNone(result["Balanced-Accuracy"])
        self.assertIsNone(result["Macro-F1"])


if __name__ == "__main__":
    unittest.main()
