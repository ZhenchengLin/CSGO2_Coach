"""
Synthetic scoring-kernel tests.

No Confirmation files, Manifest, model artifacts,
network requests or real Demo data are accessed.
"""

import sys
import unittest

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from v4_a_confirm_scoring_kernel_v1 import (
    row_log_loss,
    summarize_paired_differences,
)


class ScoringKernelTests(unittest.TestCase):

    def test_true_class_log_loss_and_nontrue_zero(self):

        p = np.zeros(
            (2, 15),
            dtype=np.float32,
        )

        p[0, 0] = 1.0
        p[1, 0] = 0.25
        p[1, 1] = 0.75

        result = row_log_loss(
            np.array([0, 0]),
            p,
        )

        np.testing.assert_allclose(
            result,
            [0.0, -np.log(0.25)],
        )

    def test_zero_true_class_rejected_without_clipping(self):

        p = np.zeros((1, 15))
        p[0, 1] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            "Zero true-class probability",
        ):
            row_log_loss(
                np.array([0]),
                p,
            )

    def test_invalid_probability_and_labels_rejected(self):

        p = np.zeros((1, 15))
        p[0, 0] = 0.9

        cases = [
            (
                np.array([0]),
                p,
            ),
            (
                np.array([15]),
                np.eye(15)[:1],
            ),
            (
                np.array([0.0]),
                np.eye(15)[:1],
            ),
            (
                np.array([0]),
                np.full((1, 15), np.nan),
            ),
            (
                np.array([0]),
                np.zeros((1, 14)),
            ),
        ]

        for labels, probabilities in cases:

            with self.subTest(
                labels=labels,
                shape=probabilities.shape,
            ):

                with self.assertRaises(ValueError):

                    row_log_loss(
                        labels,
                        probabilities,
                    )

    def test_unequal_match_sizes_have_distinct_estimands(self):

        ranks = np.concatenate([
            np.ones(10, dtype=np.int64),
            np.arange(2, 21, dtype=np.int64),
        ])

        control = np.ones(len(ranks))

        candidate = control.copy()

        candidate[:10] += 2.0

        result = summarize_paired_differences(
            ranks,
            control,
            candidate,
        )

        self.assertEqual(
            result["matches"],
            20,
        )

        self.assertEqual(
            result["rows"],
            29,
        )

        self.assertAlmostEqual(
            result["pooled_difference"],
            20 / 29,
        )

        self.assertAlmostEqual(
            result["equal_match_difference"],
            2 / 20,
        )

        sums = np.array(
            [20.0] + [0.0] * 19
        )

        counts = np.array(
            [10] + [1] * 19
        )

        means = sums / counts

        draw = np.random.default_rng(
            20260919
        ).integers(
            0,
            20,
            size=(20000, 20),
        )

        expected_pooled = np.quantile(
            sums[draw].sum(axis=1)
            / counts[draw].sum(axis=1),
            [0.025, 0.975],
            method="linear",
        )

        expected_equal = np.quantile(
            means[draw].mean(axis=1),
            [0.025, 0.975],
            method="linear",
        )

        np.testing.assert_array_equal(
            result["pooled_match_bootstrap_95ci"],
            expected_pooled,
        )

        np.testing.assert_array_equal(
            result["equal_match_bootstrap_95ci"],
            expected_equal,
        )

        self.assertNotEqual(
            result["pooled_match_bootstrap_95ci"],
            result["equal_match_bootstrap_95ci"],
        )

    def test_bootstrap_ignores_row_and_match_order(self):

        ranks = np.arange(1, 21)

        control = np.ones(20)

        candidate = control + ranks / 100

        first = summarize_paired_differences(
            ranks,
            control,
            candidate,
        )

        reversed_result = summarize_paired_differences(
            ranks[::-1],
            control[::-1],
            candidate[::-1],
        )

        self.assertEqual(
            first,
            reversed_result,
        )

    def test_refuses_incomplete_match_set(self):

        ranks = np.arange(1, 20)

        with self.assertRaisesRegex(
            ValueError,
            "20 distinct",
        ):

            summarize_paired_differences(
                ranks,
                np.ones(19),
                np.ones(19),
            )

    def test_refuses_unaligned_or_nonfinite_input(self):

        ranks = np.arange(1, 21)

        control = np.ones(20)

        cases = [
            (
                control,
                control[:19],
            ),
            (
                control,
                np.full(20, np.nan),
            ),
            (
                control,
                np.full(20, -1),
            ),
        ]

        for control_loss, candidate_loss in cases:

            with self.assertRaises(ValueError):

                summarize_paired_differences(
                    ranks,
                    control_loss,
                    candidate_loss,
                )


if __name__ == "__main__":
    unittest.main()
