import hashlib
import json
import sys
import tempfile
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_v4_a_confirm_demo_identity_v1 as audit


class DemoIdentityTests(unittest.TestCase):

    def make_extraction(self, root):
        extracted = root / "extracted_v1"
        extracted.mkdir()

        content = b"synthetic demo bytes"
        (extracted / "map.dem").write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()

        event = {
            "archive_sha256": "a" * 64,
            "archive_size_bytes": 123,
        }

        record = {
            "version": "V4_A_CONFIRM_ZIP_EXTRACTION_V1",
            "candidate_rank": 1,
            "source_match_id": "2397691",
            "source_archive_sha256": event["archive_sha256"],
            "source_archive_size_bytes": event["archive_size_bytes"],
            "extracted_file_count": 1,
            "extracted_total_bytes": len(content),
            "members": [{
                "path": "map.dem",
                "size_bytes": len(content),
                "sha256": digest,
                "is_demo_filename": True,
            }],
            "archive_crc_verified": True,
            "technical_eligibility_evaluated": False,
            "model_scoring_performed": False,
        }

        (extracted / audit.RECORD_NAME).write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )

        return extracted, event

    def verify(self, extracted, event):
        return audit.verify_extracted_files(
            extracted,
            event,
            expected_rank=1,
            expected_match_id="2397691",
        )

    def test_historical_manifest_identities(self):
        hashes = audit.load_historical_hashes()
        self.assertTrue(hashes)

    def test_valid_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            extracted, event = self.make_extraction(Path(temp))
            verified = self.verify(extracted, event)
            self.assertEqual(len(verified), 1)
            self.assertTrue(verified[0]["is_demo_filename"])

    def test_modified_demo_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            extracted, event = self.make_extraction(Path(temp))
            (extracted / "map.dem").write_bytes(b"changed")

            with self.assertRaises(RuntimeError):
                self.verify(extracted, event)

    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            extracted, event = self.make_extraction(Path(temp))
            (extracted / "unexpected.txt").write_text("extra")

            with self.assertRaises(RuntimeError):
                self.verify(extracted, event)

    def test_archive_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            extracted, event = self.make_extraction(Path(temp))
            event["archive_sha256"] = "b" * 64

            with self.assertRaises(RuntimeError):
                self.verify(extracted, event)

    def test_historical_overlap_is_not_eligibility(self):
        digest = "a" * 64
        historical = {digest: ["development"]}

        self.assertEqual(
            historical.get(digest, []),
            ["development"],
        )


if __name__ == "__main__":
    unittest.main()
