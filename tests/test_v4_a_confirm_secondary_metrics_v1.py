"""Synthetic Secondary Metrics regression tests."""

import sys
import unittest

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

from v4_a_confirm_secondary_metrics_v1 import (
    summarize_secondary_metrics,
)


CLASSES = [f"ZONE_{i}" for i in range(15)]


class SecondaryMetricsTests(unittest.TestCase):

    def test_perfect_predictions_and_absent_classes(self):
        y = np.zeros(20, dtype=np.int64)
        p = np.zeros((20, 15), dtype=np.float64)
        p[:, 0] = 1.0

        result = summarize_secondary_metrics(
            y,
            p,
            np.arange(1, 21),
            CLASSES,
        )

        self.assertEqual(result["rows"], 20)
        self.assertAlmostEqual(result["log_loss"], 0.0)
        self.assertAlmostEqual(result["multiclass_brier"], 0.0)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["top2_accuracy"], 1.0)

        # All 15 classes participate in Macro-F1.
        self.assertAlmostEqual(result["macro_f1"], 1 / 15)

        self.assertEqual(result["per_zone"][0]["support"], 20)
        self.assertEqual(result["per_zone"][0]["recall"], 1.0)
        self.assertEqual(result["per_zone"][1]["support"], 0)
        self.assertIsNone(result["per_zone"][1]["recall"])

        self.assertAlmostEqual(
            result["calibration"]["top1_ece"],
            0.0,
        )

    def test_brier_accuracy_recall_and_ece(self):
        y = np.array([0] * 10 + [1] * 10)

        p = np.zeros((20, 15), dtype=np.float64)
        p[:, 0] = 0.8
        p[:, 1] = 0.2

        result = summarize_secondary_metrics(
            y,
            p,
            np.arange(1, 21),
            CLASSES,
        )

        self.assertAlmostEqual(result["multiclass_brier"], 0.68)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["top2_accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 15)

        self.assertEqual(result["per_zone"][0]["support"], 10)
        self.assertEqual(result["per_zone"][1]["support"], 10)
        self.assertAlmostEqual(result["per_zone"][0]["recall"], 1.0)
        self.assertAlmostEqual(result["per_zone"][1]["recall"], 0.0)

        self.assertAlmostEqual(
            result["calibration"]["top1_ece"],
            0.3,
        )

        self.assertEqual(
            sum(
                item["rows"]
                for item in result["calibration"]["bins"]
            ),
            20,
        )

    def test_top1_and_top2_ties_use_class_order(self):
        y = np.ones(20, dtype=np.int64)

        p = np.zeros((20, 15), dtype=np.float64)
        p[:, 0] = 0.4
        p[:, 1] = 0.4
        p[:, 2] = 0.2

        result = summarize_secondary_metrics(
            y,
            p,
            np.arange(1, 21),
            CLASSES,
        )

        self.assertAlmostEqual(result["accuracy"], 0.0)
        self.assertAlmostEqual(result["top2_accuracy"], 1.0)

    def test_equal_match_log_loss_uses_equal_weights(self):
        ranks = np.concatenate([
            np.ones(10, dtype=np.int64),
            np.arange(2, 21, dtype=np.int64),
        ])

        y = np.zeros(len(ranks), dtype=np.int64)

        p = np.zeros((len(ranks), 15), dtype=np.float64)
        p[:, 0] = 1.0

        p[:10, 0] = 0.25
        p[:10, 1] = 0.75

        result = summarize_secondary_metrics(
            y,
            p,
            ranks,
            CLASSES,
        )

        loss = -np.log(0.25)

        self.assertAlmostEqual(
            result["log_loss"],
            10 * loss / 29,
        )

        self.assertAlmostEqual(
            result["equal_match_log_loss"],
            loss / 20,
        )

    def test_invalid_inputs_are_rejected(self):
        y = np.zeros(20, dtype=np.int64)
        p = np.zeros((20, 15))
        p[:, 0] = 1.0
        ranks = np.arange(1, 21)

        with self.assertRaises(ValueError):
            summarize_secondary_metrics(
                y,
                p,
                ranks[:19],
                CLASSES,
            )

        with self.assertRaises(ValueError):
            summarize_secondary_metrics(
                y,
                p,
                ranks,
                CLASSES[:14],
            )

        with self.assertRaises(ValueError):
            summarize_secondary_metrics(
                y,
                p,
                np.ones(20, dtype=np.int64),
                CLASSES,
            )


if __name__ == "__main__":
    unittest.main()
