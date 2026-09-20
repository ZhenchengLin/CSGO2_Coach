import hashlib
import json
import sys
import tempfile
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import manage_v4_a_confirm_intake_state_v1 as manager


class IntakeStateTests(unittest.TestCase):

    def test_frozen_initial_ledger(self):

        candidates = manager.load_initial_candidates()

        self.assertEqual(
            len(candidates),
            25,
        )

        self.assertEqual(
            [row["candidate_rank"] for row in candidates],
            list(range(1, 26)),
        )

        self.assertTrue(
            all(
                row["initial_status"] == "PENDING"
                for row in candidates
            )
        )

    def test_missing_archive_is_rejected(self):

        row = manager.load_initial_candidates()[0]

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            with self.assertRaises(RuntimeError):

                manager.verify_archive_record(
                    row,
                    repo_root=root,
                    staging_root=root / "staging",
                )

    def test_archive_record_and_sha256(self):

        row = manager.load_initial_candidates()[0]

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            staging = root / "staging"

            rank_dir = (
                staging
                / manager.rank_dir_name(row)
            )

            rank_dir.mkdir(parents=True)

            content = (
                b"PK\x03\x04"
                + b"synthetic archive test data"
            )

            archive_path = (
                rank_dir / "archive.zip"
            )

            archive_path.write_bytes(content)

            digest = hashlib.sha256(
                content
            ).hexdigest()

            record = {
                "version": (
                    "V4_A_CONFIRM_ARCHIVE_ACQUISITION_V1"
                ),
                "candidate_rank": row["candidate_rank"],
                "source_match_id": row["source_match_id"],
                "frozen_match_date": (
                    row["frozen_match_date"]
                ),
                "frozen_match_url": (
                    row["frozen_match_url"]
                ),
                "source_download_url": (
                    "https://www.hltv.org/download/demo/mock"
                ),
                "archive_signature_extension": ".zip",
                "archive_path": str(
                    archive_path.relative_to(root)
                ),
                "archive_size_bytes": len(content),
                "archive_sha256": digest,
                "technical_eligibility": "NOT_EVALUATED",
                "demo_extracted": False,
                "model_scoring_performed": False,
            }

            record_path = (
                rank_dir / "acquisition_record.json"
            )

            record_path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            verified = manager.verify_archive_record(
                row,
                repo_root=root,
                staging_root=staging,
            )

            self.assertEqual(
                verified["archive_sha256"],
                digest,
            )

            self.assertEqual(
                verified["archive_size_bytes"],
                len(content),
            )

            self.assertEqual(
                verified["record_sha256"],
                manager.sha256_file(record_path),
            )

            archive_path.write_bytes(
                content + b"modified"
            )

            with self.assertRaises(RuntimeError):

                manager.verify_archive_record(
                    row,
                    repo_root=root,
                    staging_root=staging,
                )

    def test_event_rejects_wrong_rank(self):

        row = manager.load_initial_candidates()[0]

        archive = {
            "record_sha256": "record",
            "archive_sha256": "archive",
            "archive_size_bytes": 123,
            "archive_relative_path": "example.zip",
        }

        event = {
            "version": manager.EVENT_VERSION,
            "status": "ARCHIVE_ACQUIRED",
            "candidate_rank": 2,
            "source_match_id": row["source_match_id"],
            "acquisition_record_sha256": "record",
            "archive_sha256": "archive",
            "archive_size_bytes": 123,
            "archive_path": "example.zip",
            "technical_eligibility_evaluated": False,
            "model_scoring_performed": False,
        }

        with self.assertRaises(RuntimeError):

            manager.validate_event(
                row,
                event,
                archive,
            )


if __name__ == "__main__":
    unittest.main()
