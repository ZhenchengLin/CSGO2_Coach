"""Synthetic tests only; no Confirmation artifacts are accessed."""

import sys
import unittest

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v4_a_confirm_row_alignment_v1 import validate_rows


class RowAlignmentTests(unittest.TestCase):

    def setUp(self):
        self.keys = [
            ("synthetic_a.dem", 1, 320, 5),
            ("synthetic_a.dem", 1, 640, 5),
            ("synthetic_b.dem", 2, 320, 5),
        ]

        self.labels = np.array([0, 1, 14], dtype=np.int64)

        self.control = np.zeros((3, 24), dtype=np.float32)
        self.candidate = np.zeros((3, 56), dtype=np.float32)

        self.control[:, 0] = [1, 2, 3]
        self.candidate[:, :24] = self.control

    def check(self, **changes):
        values = {
            "control_keys": self.keys,
            "candidate_keys": self.keys,
            "control_labels": self.labels,
            "candidate_labels": self.labels,
            "control_features": self.control,
            "candidate_features": self.candidate,
        }

        values.update(changes)
        return validate_rows(**values)

    def test_identical_rows_and_feature_prefix_pass(self):
        result = self.check()

        self.assertEqual(
            result["status"],
            "IN_MEMORY_ALIGNMENT_PASS",
        )

        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["horizon_sec"], 5)

        self.assertFalse(
            result["confirmation_scoring_authorized"]
        )

    def test_target_row_reordering_rejected(self):
        with self.assertRaisesRegex(ValueError, "order differs"):
            self.check(candidate_keys=self.keys[::-1])

    def test_duplicate_target_key_rejected(self):
        keys = [self.keys[0], self.keys[0], self.keys[2]]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check(
                control_keys=keys,
                candidate_keys=keys,
            )

    def test_target_label_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "labels differ"):
            self.check(
                candidate_labels=np.array([0, 2, 14])
            )

    def test_feature_prefix_mismatch_rejected(self):
        changed = self.candidate.copy()
        changed[1, 0] = 99

        with self.assertRaisesRegex(ValueError, "first 24"):
            self.check(candidate_features=changed)

    def test_invalid_feature_shape_and_nan_rejected(self):
        cases = [
            self.candidate[:, :55],
            np.full((3, 56), np.nan),
        ]

        for candidate in cases:
            with self.subTest(shape=candidate.shape):
                with self.assertRaises(ValueError):
                    self.check(candidate_features=candidate)

    def test_mixed_horizons_rejected(self):
        mixed = [
            self.keys[0],
            self.keys[1],
            ("synthetic_b.dem", 2, 320, 10),
        ]

        with self.assertRaisesRegex(ValueError, "one horizon"):
            self.check(
                control_keys=mixed,
                candidate_keys=mixed,
            )


if __name__ == "__main__":
    unittest.main()
