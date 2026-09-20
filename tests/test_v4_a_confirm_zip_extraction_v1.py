import json
import sys
import tempfile
import unittest
import zipfile

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from extract_v4_a_confirm_zip_v1 import (
    EXTRACT_DIR_NAME,
    RECORD_NAME,
    extract_zip,
)


GIB = 1024 ** 3


def safe_free_space():
    return 14 * GIB


class SafeZipExtractionTests(unittest.TestCase):

    def make_zip(self, root, entries):

        archive_path = (
            root / "archive.zip"
        )

        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:

            for name, content in entries:

                archive.writestr(
                    name,
                    content,
                )

        return archive_path

    def test_valid_zip_extracts_and_records_sha256(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("map1.dem", b"synthetic demo 1"),
                    ("map2.dem", b"synthetic demo 2"),
                ],
            )

            output = (
                root / EXTRACT_DIR_NAME
            )

            record = extract_zip(
                archive,
                output,
                free_space_fn=safe_free_space,
                candidate_rank=1,
                source_match_id="2397691",
            )

            self.assertEqual(
                (output / "map1.dem").read_bytes(),
                b"synthetic demo 1",
            )

            self.assertEqual(
                (output / "map2.dem").read_bytes(),
                b"synthetic demo 2",
            )

            saved = json.loads(
                (output / RECORD_NAME).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                saved,
                record,
            )

            self.assertEqual(
                saved["extracted_file_count"],
                2,
            )

            self.assertTrue(
                saved["archive_crc_verified"]
            )

            self.assertFalse(
                saved["technical_eligibility_evaluated"]
            )

    def test_refuses_to_overwrite_extraction(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("map.dem", b"synthetic demo"),
                ],
            )

            output = root / EXTRACT_DIR_NAME

            extract_zip(
                archive,
                output,
                free_space_fn=safe_free_space,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "EXTRACTION_ALREADY_EXISTS",
            ):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=safe_free_space,
                )

    def test_path_traversal_rejected(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("../outside.dem", b"malicious"),
                ],
            )

            output = root / EXTRACT_DIR_NAME

            with self.assertRaises(RuntimeError):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=safe_free_space,
                )

            self.assertFalse(
                (root / "outside.dem").exists()
            )

            self.assertFalse(
                output.exists()
            )

    def test_storage_block_does_not_create_output(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("map.dem", b"x" * 1024),
                ],
            )

            output = root / EXTRACT_DIR_NAME

            with self.assertRaisesRegex(
                RuntimeError,
                "STORAGE_BLOCK",
            ):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=lambda: 11 * GIB,
                )

            self.assertFalse(
                output.exists()
            )

    def test_corrupt_zip_does_not_publish_output(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = root / "archive.zip"

            archive.write_bytes(
                b"PK\x03\x04not-a-valid-zip"
            )

            output = root / EXTRACT_DIR_NAME

            with self.assertRaises(RuntimeError):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=safe_free_space,
                )

            self.assertFalse(
                output.exists()
            )

    def test_archive_identity_mismatch_rejected(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("map.dem", b"synthetic"),
                ],
            )

            output = root / EXTRACT_DIR_NAME

            with self.assertRaisesRegex(
                RuntimeError,
                "SHA256",
            ):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=safe_free_space,
                    expected_archive_sha256="0" * 64,
                )

            self.assertFalse(
                output.exists()
            )

    def test_file_directory_conflict_rejected(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("folder", b"regular file"),
                    ("folder/map.dem", b"synthetic demo"),
                ],
            )

            output = root / EXTRACT_DIR_NAME

            with self.assertRaisesRegex(
                RuntimeError,
                "path conflict",
            ):

                extract_zip(
                    archive,
                    output,
                    free_space_fn=safe_free_space,
                )

            self.assertFalse(
                output.exists()
            )


if __name__ == "__main__":
    unittest.main()
