#!/usr/bin/env python3

"""
V4-A Confirmation V2: next-rank acquisition entry point.

The frozen queue and committed decision records determine rank
order. The V2 Guard currently validates Rank 1 and prepares
Rank 2 only; no later rank is authorized by this version.

Default mode is offline. --download explicitly authorizes
one Rank 2 archive acquisition through the existing guarded
single-archive Downloader.

This entry point does not extract a Demo, determine technical
eligibility, create the final Manifest, or score models.
"""

from __future__ import annotations

import argparse
import sys

from datetime import datetime, timezone

from guard_v4_a_confirm_next_rank_v2 import (
    main as guard_main,
)

from download_v4_a_confirm_archive_v1 import (
    GIB,
    MIB,
    download_one_archive,
    free_bytes,
    inspect_match_page,
    safe_download_budget,
    verify_selected_candidate,
)

from manage_v4_a_confirm_intake_state_v1 import require


AUTHORIZED_RANK = 2
EXPECTED_MATCH_ID = "2397692"


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=AUTHORIZED_RANK,
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly authorize one Rank 2 archive download.",
    )

    args = parser.parse_args(argv)

    require(
        args.rank == AUTHORIZED_RANK,
        "RANK_ORDER_BLOCK: this V2 entry point currently "
        "authorizes Rank 2 only.",
    )

    print("=== V4-A V2 ACQUISITION PREFLIGHT ===")

    # Requires synchronized, clean main; verifies the pinned
    # Rank 1 decision and evidence; rejects existing Rank 2
    # staging and Archive Event; enforces the 12 GiB floor.
    guard_main()

    row, original_start = verify_selected_candidate(
        AUTHORIZED_RANK
    )

    require(
        int(row["candidate_rank"]) == AUTHORIZED_RANK
        and row["source_match_id"] == EXPECTED_MATCH_ID,
        "Frozen Rank 2 candidate identity mismatch.",
    )

    budget = safe_download_budget()

    print()
    print("=== SELECTED FROZEN CANDIDATE ===")
    print("Candidate Rank:", row["candidate_rank"])
    print("Match ID:", row["source_match_id"])
    print("Original scheduled UTC:", original_start.isoformat())
    print("Available disk: %.2f GiB" % (free_bytes() / GIB))
    print("Maximum archive budget: %.1f MiB" % (budget / MIB))
    print("Minimum remaining disk: 12 GiB")

    if not args.download:
        print()
        print("STATUS: RANK2_ACQUISITION_OFFLINE_PLAN_READY")
        print("HLTV requests: NONE")
        print("Archive downloads: NONE")
        print("Technical eligibility: NOT EVALUATED")
        print("Model scoring: NONE")
        return

    require(
        datetime.now(timezone.utc) >= original_start,
        "SOURCE_WAIT: original scheduled start has not passed.",
    )

    print()
    print("=== INSPECT RANK 2 SOURCE ===")

    # The existing source inspector follows only approved
    # same-Match-ID HLTV match-page redirects.
    source_record = inspect_match_page(row)

    links = source_record["demo_links"]

    print("Match-page SHA256:", source_record["page_sha256"])
    print("Visible Demo links:", len(links))

    if not links:
        print("STATUS: AWAITING_DEMO_OR_MATCH_RESULT")
        print("Technical exclusion assigned: NO")
        return

    if len(links) != 1:
        print("STATUS: SOURCE_REVIEW_REQUIRED")
        print("Multiple Demo links; no automatic selection.")

        for link in links:
            print(" ", link)

        return

    print()
    print("=== EXPLICIT SINGLE-ARCHIVE DOWNLOAD ===")

    # Existing Downloader rechecks the storage budget,
    # validates redirects, enforces download-size limits,
    # and creates the staging directory exclusively.
    download_one_archive(
        row=row,
        source_record=source_record,
    )

    print()
    print("STATUS: RANK2_ARCHIVE_DOWNLOADED")
    print("Archive verification and extraction: PENDING")
    print("Technical eligibility: NOT EVALUATED")
    print("Final Confirmation Manifest: NOT CREATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        print("V2 ACQUISITION STOP:", exc, file=sys.stderr)
        raise SystemExit(1)
