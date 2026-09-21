"""Synthetic record-contract tests; no real Demo access."""

import copy
import sys
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from read_v4_a_confirm_rar_extraction_record_v1 import (
    ARCHIVE_SHA256,
    DEMO_BYTES,
    DEMO_SHA256,
    MATCH_ID,
    MEMBER,
    MEMBERS,
    RECORD_VERSION,
    validate_record,
)


ARCHIVE_PATH = (
    "data/raw/v4_a_confirm_download_staging/"
    "rank_01_2397691/archive.rar"
)

DEMO_PATH = (
    "data/raw/v4_a_confirm_download_staging/"
    "rank_01_2397691/"
    "brute-vs-sparta-m2-mirage.dem"
)


def fixture():
    event = {
        "archive_path": ARCHIVE_PATH,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_size_bytes": 466363231,
    }

    record = {
        "version": RECORD_VERSION,
        "candidate_rank": 1,
        "source_match_id": MATCH_ID,
        "source_archive_path": ARCHIVE_PATH,
        "source_archive_size_bytes": 466363231,
        "source_archive_sha256": ARCHIVE_SHA256,
        "archive_members": copy.deepcopy(MEMBERS),
        "selected_member": MEMBER,
        "selected_member_declared_bytes": DEMO_BYTES,
        "extracted_demo_path": DEMO_PATH,
        "extracted_demo_size_bytes": DEMO_BYTES,
        "extracted_demo_sha256": DEMO_SHA256,
        "selected_member_replay_sha256": DEMO_SHA256,
        "selected_member_replay_exit_zero": True,
        "selected_member_stream_matches_extracted_demo": True,
        "whole_archive_crc_verified": False,
        "unselected_members_extracted": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }

    return record, event


class RarRecordTests(unittest.TestCase):

    def check(self, record, event):
        return validate_record(
            record,
            event,
            expected_archive_path=ARCHIVE_PATH,
            expected_demo_path=DEMO_PATH,
        )

    def test_valid_record_contract(self):
        record, event = fixture()
        self.assertTrue(self.check(record, event))

    def test_wrong_archive_identity_rejected(self):
        record, event = fixture()
        record["source_archive_sha256"] = "0" * 64

        with self.assertRaisesRegex(RuntimeError, "Archive identity"):
            self.check(record, event)

    def test_wrong_demo_digest_rejected(self):
        record, event = fixture()
        record["extracted_demo_sha256"] = "0" * 64

        with self.assertRaisesRegex(RuntimeError, "Demo identity"):
            self.check(record, event)

    def test_wrong_replay_digest_rejected(self):
        record, event = fixture()
        record["selected_member_replay_sha256"] = "0" * 64

        with self.assertRaisesRegex(RuntimeError, "replay evidence"):
            self.check(record, event)

    def test_unexpected_archive_member_rejected(self):
        record, event = fixture()
        record["archive_members"].append({
            "path": "unexpected.dem",
            "declared_uncompressed_bytes": 123,
        })

        with self.assertRaisesRegex(RuntimeError, "member inventory"):
            self.check(record, event)

    def test_path_substitution_rejected(self):
        record, event = fixture()
        record["extracted_demo_path"] = "../../other.dem"

        with self.assertRaisesRegex(RuntimeError, "Demo identity"):
            self.check(record, event)

    def test_unsupported_whole_archive_claim_rejected(self):
        record, event = fixture()
        record["whole_archive_crc_verified"] = True

        with self.assertRaisesRegex(RuntimeError, "unsupported Archive"):
            self.check(record, event)

    def test_scoring_boundary_violation_rejected(self):
        record, event = fixture()
        record["model_scoring_performed"] = True

        with self.assertRaisesRegex(RuntimeError, "scoring boundary"):
            self.check(record, event)

    def test_unexpected_schema_rejected(self):
        record, event = fixture()
        record["unexpected"] = True

        with self.assertRaisesRegex(RuntimeError, "record schema"):
            self.check(record, event)


if __name__ == "__main__":
    unittest.main()
