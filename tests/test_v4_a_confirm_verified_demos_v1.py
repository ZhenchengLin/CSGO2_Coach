"""Offline tests for V4-A verified Demo routing."""

import sys
import unittest

from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from load_v4_a_confirm_verified_demos_v1 import (
    load_verified_demos,
)


ROW = {
    "candidate_rank": 1,
    "source_match_id": "2397691",
}

EVENT = {
    "status": "ARCHIVE_ACQUIRED",
    "candidate_rank": 1,
    "source_match_id": "2397691",
    "archive_path": (
        "data/raw/v4_a_confirm_download_staging/"
        "rank_01_2397691/archive.rar"
    ),
    "archive_sha256": "a" * 64,
}


class VerifiedDemoLoaderTests(unittest.TestCase):

    def test_rar_uses_verified_record(self):
        result = {
            "candidate_rank": 1,
            "source_match_id": "2397691",
            "source_archive_sha256": "a" * 64,
            "demo_path": "synthetic/mirage.dem",
            "demo_size_bytes": 123,
            "demo_sha256": "b" * 64,
            "technical_eligibility_evaluated": False,
            "model_scoring_performed": False,
        }

        with patch(
            "read_v4_a_confirm_rar_extraction_record_v1."
            "read_verified_rank1_record",
            return_value=result,
        ), patch(
            "load_v4_a_confirm_verified_demos_v1."
            "Path.is_file",
            return_value=True,
        ), patch(
            "load_v4_a_confirm_verified_demos_v1."
            "Path.is_symlink",
            return_value=False,
        ):
            demos = load_verified_demos(ROW, EVENT)

        self.assertEqual(len(demos), 1)
        self.assertEqual(demos[0]["sha256"], "b" * 64)
        self.assertTrue(demos[0]["is_demo_filename"])

    def test_event_identity_mismatch_is_rejected(self):
        event = dict(EVENT)
        event["source_match_id"] = "other"

        with self.assertRaisesRegex(
            RuntimeError,
            "candidate/event identity mismatch",
        ):
            load_verified_demos(ROW, event)

    def test_other_rar_rank_is_not_implicitly_authorized(self):
        row = {
            "candidate_rank": 2,
            "source_match_id": "2397692",
        }
        event = dict(EVENT)
        event["candidate_rank"] = 2
        event["source_match_id"] = "2397692"

        with self.assertRaisesRegex(
            RuntimeError,
            "Rank 1 only",
        ):
            load_verified_demos(row, event)

    def test_unrecognized_archive_format_is_rejected(self):
        event = dict(EVENT)
        event["archive_path"] = "synthetic/archive.7z"

        with self.assertRaisesRegex(
            RuntimeError,
            "unsupported Archive format",
        ):
            load_verified_demos(ROW, event)


if __name__ == "__main__":
    unittest.main()
