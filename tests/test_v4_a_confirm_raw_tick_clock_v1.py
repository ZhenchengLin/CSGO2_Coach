import sys
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from audit_v4_a_confirm_raw_tick_clock_v1 import (
    estimate_raw_clock,
    EXPECTED_RAW_TICKS_PER_SECOND,
    TICKRATE_TOLERANCE,
)


class RawTickClockTests(unittest.TestCase):

    def test_exact_64_tick_clock(self):

        observations = [
            (0, 0.00),
            (16, 0.25),
            (32, 0.50),
            (64, 1.00),
        ]

        measured, intervals = estimate_raw_clock(
            observations
        )

        self.assertAlmostEqual(
            measured,
            64.0,
        )

        self.assertEqual(
            intervals,
            3,
        )

    def test_exact_128_tick_clock(self):

        observations = [
            (0, 0.00),
            (32, 0.25),
            (64, 0.50),
            (128, 1.00),
        ]

        measured, _ = estimate_raw_clock(
            observations
        )

        self.assertAlmostEqual(
            measured,
            128.0,
        )

        self.assertGreater(
            abs(
                measured
                - EXPECTED_RAW_TICKS_PER_SECOND
            ),
            TICKRATE_TOLERANCE,
        )

    def test_median_resists_single_outlier(self):

        observations = [
            (0, 0.0),
            (64, 1.0),
            (128, 2.0),
            (192, 2.1),
            (256, 3.1),
            (320, 4.1),
        ]

        measured, intervals = estimate_raw_clock(
            observations
        )

        self.assertEqual(
            intervals,
            5,
        )

        self.assertAlmostEqual(
            measured,
            64.0,
        )

    def test_nonpositive_deltas_are_excluded(self):

        observations = [
            (0, 0.0),
            (64, 1.0),
            (64, 1.0),
            (128, 2.0),
            (192, 3.0),
        ]

        measured, intervals = estimate_raw_clock(
            observations
        )

        self.assertAlmostEqual(
            measured,
            64.0,
        )

        self.assertEqual(
            intervals,
            3,
        )

    def test_large_game_time_gap_is_excluded(self):

        observations = [
            (0, 0.0),
            (64, 1.0),
            (256, 4.0),
            (320, 5.0),
        ]

        measured, intervals = estimate_raw_clock(
            observations
        )

        self.assertAlmostEqual(
            measured,
            64.0,
        )

        self.assertEqual(
            intervals,
            2,
        )

    def test_no_usable_intervals_stops(self):

        observations = [
            (0, 0.0),
            (0, 0.0),
            (64, 3.0),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "RAW_CLOCK_REVIEW_REQUIRED",
        ):

            estimate_raw_clock(
                observations
            )


if __name__ == "__main__":
    unittest.main()
