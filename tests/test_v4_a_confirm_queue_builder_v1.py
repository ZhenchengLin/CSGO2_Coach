import copy
import hashlib
import sys
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from build_v4_a_confirm_queue_v1 import (
    build_candidate_queue,
    load_frozen_rules,
    read_historical_match_ids,
    render_queue_csv,
)


CAPTURE = "2026-09-20T18:00:00Z"
EARLIEST = "2026-09-21T00:00:00Z"

HISTORICAL = {
    str(value)
    for value in range(9000001, 9000026)
}


def match(mid, start):
    return {
        "source_match_id": str(mid),
        "scheduled_start_utc": start,
        "event": "Synthetic Test Event",
        "team1": "TBD",
        "team2": "TBD",
        "source_url": (
            f"https://www.hltv.org/matches/"
            f"{mid}/synthetic-test-match"
        ),
    }


def fixture():
    rows = []

    for i in range(30):
        mid = 3000000 + i

        # Most matches occur on the same UTC date.
        hour = i % 12

        rows.append(
            match(
                mid,
                f"2026-09-21T{hour:02d}:00:00Z",
            )
        )

    return rows


class QueueBuilderTests(unittest.TestCase):

    def build(self, rows):
        return build_candidate_queue(
            snapshot_rows=rows,
            captured_at_utc=CAPTURE,
            historical_match_ids=HISTORICAL,
            earliest_start_utc=EARLIEST,
        )

    def test_frozen_rule_identities(self):
        protocol, amendment = load_frozen_rules()

        self.assertEqual(
            protocol["sample_plan"]["initial_candidate_queue"],
            25,
        )

        self.assertEqual(
            amendment["candidate_selection"]["initial_queue_size"],
            25,
        )

        self.assertEqual(
            amendment["new_prospective_sampling_window"][
                "earliest_scheduled_start_utc"
            ],
            EARLIEST,
        )

    def test_historical_corpus_from_repository(self):
        _, amendment = load_frozen_rules()

        paths = [
            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch1"],
            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch2"],
        ]

        historical = read_historical_match_ids(paths)

        self.assertEqual(len(historical), 25)

    def test_selects_25_deterministically(self):
        rows = fixture()

        forward = self.build(rows)
        backward = self.build(list(reversed(rows)))

        self.assertEqual(forward, backward)
        self.assertEqual(len(forward), 25)

        self.assertEqual(
            [row["candidate_rank"] for row in forward],
            list(range(1, 26)),
        )

        self.assertEqual(
            len({
                row["source_match_id"]
                for row in forward
            }),
            25,
        )

    def test_selection_order_differs_from_queue_order(self):
        rows = [
            match(3000100, "2026-09-21T01:00:00Z"),
            match(3000001, "2026-09-21T03:00:00Z"),
        ]

        rows.extend(
            match(
                3100000 + i,
                "2026-09-22T00:00:00Z",
            )
            for i in range(23)
        )

        rows.append(
            match(
                2000000,
                "2026-09-23T00:00:00Z",
            )
        )

        queue = self.build(rows)

        self.assertEqual(
            [
                row["source_match_id"]
                for row in queue[:2]
            ],
            ["3000001", "3000100"],
        )

        self.assertNotIn(
            "2000000",
            {
                row["source_match_id"]
                for row in queue
            },
        )

    def test_excludes_historical_and_past_matches(self):
        rows = fixture()

        rows.extend(
            [
                match(
                    9000001,
                    "2026-09-21T00:00:00Z",
                ),
                match(
                    4000000,
                    "2026-09-20T17:00:00Z",
                ),
                match(
                    4000001,
                    "2026-09-20T19:00:00Z",
                ),
            ]
        )

        queue = self.build(rows)

        ids = {
            row["source_match_id"]
            for row in queue
        }

        self.assertNotIn("9000001", ids)
        self.assertNotIn("4000000", ids)
        self.assertNotIn("4000001", ids)

    def test_timezone_conversion(self):
        rows = fixture()

        rows[0]["scheduled_start_utc"] = (
            "2026-09-21T03:00:00+03:00"
        )

        queue = self.build(rows)

        self.assertEqual(
            len(queue),
            25,
        )

    def test_rejects_naive_timestamp(self):
        rows = fixture()

        rows[0]["scheduled_start_utc"] = (
            "2026-09-21T00:00:00"
        )

        with self.assertRaisesRegex(
            ValueError,
            "timezone is required",
        ):
            self.build(rows)

    def test_rejects_duplicate_match_id(self):
        rows = fixture()
        rows.append(copy.deepcopy(rows[0]))

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate snapshot Match ID",
        ):
            self.build(rows)

    def test_rejects_url_id_mismatch(self):
        rows = fixture()

        rows[0]["source_url"] = (
            "https://www.hltv.org/matches/"
            "9999999/incorrect-match"
        )

        with self.assertRaisesRegex(
            ValueError,
            "URL/Match ID mismatch",
        ):
            self.build(rows)

    def test_rejects_insufficient_pool(self):
        rows = fixture()[:24]

        with self.assertRaisesRegex(
            ValueError,
            "Insufficient prospective candidate pool",
        ):
            self.build(rows)

    def test_missing_metadata_outside_selected_25(self):
        rows = fixture()

        # This later match belongs to the complete source
        # population but cannot enter the first 25.
        later = match(
            3999999,
            "2026-10-01T00:00:00Z",
        )

        later["event"] = ""
        later["team1"] = ""
        later["team2"] = ""

        rows.append(later)

        queue = self.build(rows)

        self.assertEqual(len(queue), 25)

        self.assertNotIn(
            "3999999",
            {
                row["source_match_id"]
                for row in queue
            },
        )

    def test_missing_metadata_inside_selected_25(self):
        rows = fixture()

        # The earliest match must be selected.
        rows[0]["team1"] = ""

        with self.assertRaisesRegex(
            ValueError,
            r"Selected Match 3000000: missing team1",
        ):
            self.build(rows)

    def test_missing_event_inside_selected_25(self):
        rows = fixture()

        rows[0]["event"] = ""

        with self.assertRaisesRegex(
            ValueError,
            r"Selected Match 3000000: missing event",
        ):
            self.build(rows)

    def test_deterministic_csv_bytes(self):
        queue = self.build(fixture())

        first = render_queue_csv(queue)
        second = render_queue_csv(queue)

        self.assertEqual(first, second)

        self.assertEqual(
            hashlib.sha256(first).hexdigest(),
            hashlib.sha256(second).hexdigest(),
        )

        self.assertEqual(
            len(first.decode("utf-8").splitlines()),
            26,
        )


if __name__ == "__main__":
    unittest.main()
