"""Offline tests for independent Rank 1 occupancy auditing."""

import sys
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v4_a_confirm_rank1_technical_intake_v1 import (
    independent_occupancy,
    snapshot_keys_from_team_keys,
)


def player(steamid, side, place="TSpawn"):
    return {
        "steamid": steamid,
        "side": side,
        "health": 100,
        "place": place,
    }


class Rank1IndependentOccupancyTests(unittest.TestCase):

    def test_valid_snapshot_conserves_players(self):
        players = [
            player(111, "t"),
            player(222, "ct"),
        ]

        result = independent_occupancy(players)

        self.assertEqual(result["living_t"], 1)
        self.assertEqual(result["living_ct"], 1)

        self.assertEqual(
            sum(result["vector"][:15])
            + result["vector"][30],
            1,
        )

        self.assertEqual(
            sum(result["vector"][15:30])
            + result["vector"][31],
            1,
        )

    def test_duplicate_living_identity_rejected(self):
        players = [
            player(111, "t"),
            player(111, "ct"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate living Steam ID",
        ):
            independent_occupancy(players)

    def test_missing_living_identity_rejected(self):
        players = [
            player(None, "t"),
            player(222, "ct"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "missing Steam ID",
        ):
            independent_occupancy(players)

    def test_invalid_team_size_rejected(self):
        players = [
            player(111, "t"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid living team sizes",
        ):
            independent_occupancy(players)

    def test_unknown_place_is_counted_not_excluded(self):
        players = [
            player(111, "t", "UNMAPPED_TEST_PLACE"),
            player(222, "ct", "UNMAPPED_TEST_PLACE"),
        ]

        result = independent_occupancy(players)

        self.assertEqual(result["unknown_t"], 1)
        self.assertEqual(result["unknown_ct"], 1)
        self.assertEqual(result["vector"][30], 1)
        self.assertEqual(result["vector"][31], 1)


class Rank1SnapshotKeyAdapterTests(unittest.TestCase):

    def test_three_field_team_keys_convert_to_snapshot_keys(self):
        team_keys = {
            ("mirage.dem", 1, 100),
            ("mirage.dem", 1, 200),
            ("mirage.dem", 2, 300),
        }

        actual = snapshot_keys_from_team_keys(
            team_keys,
            "mirage.dem",
        )

        self.assertEqual(
            actual,
            {
                (1, 100),
                (1, 200),
                (2, 300),
            },
        )

    def test_wrong_demo_identity_is_rejected(self):
        team_keys = {
            ("other.dem", 1, 100),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "belongs to another Demo",
        ):
            snapshot_keys_from_team_keys(
                team_keys,
                "mirage.dem",
            )

    def test_wrong_team_key_shape_is_rejected(self):
        team_keys = {
            (1, 100),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "Unexpected Team Context key shape",
        ):
            snapshot_keys_from_team_keys(
                team_keys,
                "mirage.dem",
            )


if __name__ == "__main__":
    unittest.main()
