"""Offline regression tests for V4-A V2 Rank 2 acquisition."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

import acquire_v4_a_confirm_next_rank_v2 as acquisition


ROW = {
    "candidate_rank": "2",
    "source_match_id": "2397692",
}

START = datetime(
    2026, 9, 21, 11, 0,
    tzinfo=timezone.utc,
)

SOURCE = {
    "page_sha256": "a" * 64,
    "demo_links": [
        "https://www.hltv.org/download/demo/111696",
    ],
}


class Rank2AcquisitionTests(unittest.TestCase):

    def setUp(self):
        self.guard = patch.object(
            acquisition,
            "guard_main",
        ).start()

        self.candidate = patch.object(
            acquisition,
            "verify_selected_candidate",
            return_value=(ROW, START),
        ).start()

        self.budget = patch.object(
            acquisition,
            "safe_download_budget",
            return_value=512 * acquisition.MIB,
        ).start()

        self.disk = patch.object(
            acquisition,
            "free_bytes",
            return_value=15 * acquisition.GIB,
        ).start()

        self.source = patch.object(
            acquisition,
            "inspect_match_page",
        ).start()

        self.download = patch.object(
            acquisition,
            "download_one_archive",
        ).start()

        self.addCleanup(patch.stopall)

    def execute(self, args):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            acquisition.main(args)

        return output.getvalue()

    def test_offline_plan_never_contacts_source_or_downloads(self):
        output = self.execute(["--rank", "2"])

        self.assertIn(
            "RANK2_ACQUISITION_OFFLINE_PLAN_READY",
            output,
        )

        self.guard.assert_called_once()
        self.source.assert_not_called()
        self.download.assert_not_called()

    def test_explicit_download_uses_one_verified_source(self):
        self.source.return_value = SOURCE

        output = self.execute([
            "--rank", "2", "--download",
        ])

        self.assertIn(
            "RANK2_ARCHIVE_DOWNLOADED",
            output,
        )

        self.download.assert_called_once_with(
            row=ROW,
            source_record=SOURCE,
        )

    def test_missing_link_does_not_download_or_exclude(self):
        self.source.return_value = {
            "page_sha256": "b" * 64,
            "demo_links": [],
        }

        output = self.execute([
            "--rank", "2", "--download",
        ])

        self.assertIn(
            "AWAITING_DEMO_OR_MATCH_RESULT",
            output,
        )

        self.download.assert_not_called()

    def test_multiple_links_require_review(self):
        self.source.return_value = {
            "page_sha256": "c" * 64,
            "demo_links": [
                "https://www.hltv.org/download/demo/111696",
                "https://www.hltv.org/download/demo/111697",
            ],
        }

        output = self.execute([
            "--rank", "2", "--download",
        ])

        self.assertIn("SOURCE_REVIEW_REQUIRED", output)
        self.download.assert_not_called()

    def test_unapproved_rank_rejected_before_other_actions(self):
        with self.assertRaises(RuntimeError):
            self.execute([
                "--rank", "3", "--download",
            ])

        self.guard.assert_not_called()
        self.source.assert_not_called()
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
