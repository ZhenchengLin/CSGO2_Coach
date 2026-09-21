"""Offline regression tests for the Rank 1 RAR verifier."""

import sys
import unittest

from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_v4_a_confirm_rar_selected_member_v1 import (
    run_listing,
)


NAMES = (
    "brute-vs-sparta-m2-mirage.dem\n"
    "brute-vs-sparta-m1-dust2.dem\n"
)

DETAILS = (
    "-rw-r--r--  0 0      0   328763831 "
    "Sep 21 03:25 brute-vs-sparta-m2-mirage.dem\n"
    "-rw-r--r--  0 0      0   367052154 "
    "Sep 21 03:25 brute-vs-sparta-m1-dust2.dem\n"
)


def fake_result(stdout):
    return CompletedProcess(
        args=["bsdtar"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


class RarListingTests(unittest.TestCase):

    def test_real_listing_layout_parses_size_column(self):
        with patch(
            "verify_v4_a_confirm_rar_selected_member_v1."
            "subprocess.run",
            side_effect=[
                fake_result(NAMES),
                fake_result(DETAILS),
            ],
        ) as run:
            members = run_listing(Path("synthetic.rar"))

        self.assertEqual(
            members,
            [
                "brute-vs-sparta-m2-mirage.dem",
                "brute-vs-sparta-m1-dust2.dem",
            ],
        )
        self.assertEqual(run.call_count, 2)

    def test_wrong_declared_size_is_rejected(self):
        wrong = DETAILS.replace(
            "328763831",
            "328763830",
            1,
        )

        with patch(
            "verify_v4_a_confirm_rar_selected_member_v1."
            "subprocess.run",
            side_effect=[
                fake_result(NAMES),
                fake_result(wrong),
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "member sizes differ",
            ):
                run_listing(Path("synthetic.rar"))

    def test_unexpected_member_is_rejected(self):
        wrong_names = NAMES + "unexpected.dem\n"

        with patch(
            "verify_v4_a_confirm_rar_selected_member_v1."
            "subprocess.run",
            return_value=fake_result(wrong_names),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "unexpected member inventory",
            ):
                run_listing(Path("synthetic.rar"))

    def test_nonregular_member_is_rejected(self):
        wrong = DETAILS.replace(
            "-rw-r--r--",
            "drwxr-xr-x",
            1,
        )

        with patch(
            "verify_v4_a_confirm_rar_selected_member_v1."
            "subprocess.run",
            side_effect=[
                fake_result(NAMES),
                fake_result(wrong),
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "unexpected member type or name",
            ):
                run_listing(Path("synthetic.rar"))

    def test_unexpected_listing_layout_is_rejected(self):
        wrong = DETAILS.replace(
            "Sep 21 03:25",
            "Sep 21",
            1,
        )

        with patch(
            "verify_v4_a_confirm_rar_selected_member_v1."
            "subprocess.run",
            side_effect=[
                fake_result(NAMES),
                fake_result(wrong),
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "cannot parse member metadata",
            ):
                run_listing(Path("synthetic.rar"))


if __name__ == "__main__":
    unittest.main()
