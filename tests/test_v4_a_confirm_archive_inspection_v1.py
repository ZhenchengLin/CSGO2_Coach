import stat
import sys
import tempfile
import unittest
import zipfile

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

from inspect_v4_a_confirm_archive_v1 import (
    inspect_zip,
)


class ArchiveInspectionTests(unittest.TestCase):

    def make_zip(self, directory, entries):
        path = directory / "synthetic.zip"

        with zipfile.ZipFile(
            path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:

            for name, content in entries:
                archive.writestr(name, content)

        return path

    def test_valid_archive_metadata(self):

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            archive = self.make_zip(
                root,
                [
                    ("match/map1.dem", b"synthetic demo"),
                    ("readme.txt", b"metadata"),
                ],
            )

            result = inspect_zip(
                archive,
                budget_bytes=1024 * 1024,
            )

            self.assertEqual(
                result["member_count"],
                2,
            )

            self.assertEqual(
                result["demo_filename_count"],
                1,
            )

            self.assertFalse(
                result["extraction_performed"]
            )

            self.assertFalse(
                result["technical_eligibility_evaluated"]
            )

    def test_path_traversal_rejected(self):

        with tempfile.TemporaryDirectory() as temp:
            archive = self.make_zip(
                Path(temp),
                [
                    ("../outside.dem", b"demo"),
                ],
            )

            with self.assertRaises(RuntimeError):
                inspect_zip(
                    archive,
                    budget_bytes=1024 * 1024,
                )

    def test_symlink_rejected(self):

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.zip"

            info = zipfile.ZipInfo("linked.dem")
            info.create_system = 3

            info.external_attr = (
                (stat.S_IFLNK | 0o777) << 16
            )

            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    info,
                    "outside.dem",
                )

            with self.assertRaises(RuntimeError):
                inspect_zip(
                    path,
                    budget_bytes=1024 * 1024,
                )

    def test_duplicate_name_rejected(self):

        with tempfile.TemporaryDirectory() as temp:
            archive = self.make_zip(
                Path(temp),
                [
                    ("A.dem", b"first"),
                    ("a.dem", b"second"),
                ],
            )

            with self.assertRaises(RuntimeError):
                inspect_zip(
                    archive,
                    budget_bytes=1024 * 1024,
                )

    def test_uncompressed_budget_enforced(self):

        with tempfile.TemporaryDirectory() as temp:
            archive = self.make_zip(
                Path(temp),
                [
                    ("demo.dem", b"x" * 128),
                ],
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "STORAGE_BLOCK",
            ):
                inspect_zip(
                    archive,
                    budget_bytes=32,
                )

    def test_corrupt_zip_rejected(self):

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.zip"
            path.write_bytes(b"PK\x03\x04not-a-zip")

            with self.assertRaises(RuntimeError):
                inspect_zip(
                    path,
                    budget_bytes=1024 * 1024,
                )


if __name__ == "__main__":
    unittest.main()
