#!/usr/bin/env python3

"""
V4-A Confirmation V2: read-only Rank-order Guard.

Establish the next candidate from the frozen queue and
already committed formal technical-decision evidence.

V2 currently recognizes the completed Rank 1 decision and
authorizes Rank 2 PREPARATION only. A future version must
extend decision validation before advancing beyond Rank 2.

No Demo download, extraction, eligibility decision,
Manifest creation, model loading, or scoring occurs here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

from pathlib import Path

from probe_v4_a_confirm_demo_source_v1 import (
    ROOT,
    read_frozen_queue,
)

from manage_v4_a_confirm_intake_state_v1 import (
    STAGING_ROOT,
    load_initial_candidates,
    rank_dir_name,
    require,
)

from write_v4_a_confirm_rank1_technical_evidence_v1 import (
    validate_payload,
)


RANK1_DECISION = ROOT / (
    "docs/v4_a_confirm_intake_audits/"
    "rank_01_technical_eligibility_v1.json"
)

RANK1_DATA = ROOT / (
    "docs/v4_a_confirm_intake_audits/"
    "rank_01_data_integrity_v1.json"
)

RANK1_DATE = ROOT / (
    "docs/v4_a_confirm_intake_audits/"
    "rank_01_match_date_source_v1.json"
)

RANK1_HTML = ROOT / (
    "docs/v4_a_confirm_intake_audits/"
    "rank_01_match_page_source_v1.html"
)

EXPECTED_DECISION_SHA256 = (
    "cbaa368ddd251667d74070a64a4219b1515ee0b81fd3722bbc9ce9c58d035a9e"
)

EXPECTED_RANK1_DEMO_SHA256 = (
    "848a9e9d8b0593e3684a19f681d4f9c3902a9490ae530fe20667cdb7cc5843b1"
)

MINIMUM_FREE_BYTES = 12 * 1024**3


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def verify_rank1():
    """Verify committed Rank 1 decision and its source links."""

    require(
        RANK1_DECISION.is_file()
        and not RANK1_DECISION.is_symlink()
        and sha256_file(RANK1_DECISION)
        == EXPECTED_DECISION_SHA256,
        "Rank 1 formal decision is missing or has changed.",
    )

    decision = json.loads(
        RANK1_DECISION.read_text(encoding="utf-8")
    )

    require(
        decision["version"]
        == "V4_A_CONFIRM_RANK1_TECHNICAL_ELIGIBILITY_V1"
        and decision["record_type"]
        == "FORMAL_TECHNICAL_ELIGIBILITY_DECISION"
        and decision["candidate_rank"] == 1
        and decision["source_match_id"] == "2397691"
        and decision["decision"] == "ELIGIBLE",
        "Rank 1 decision identity/status mismatch.",
    )

    expected_files = (
        (
            RANK1_DATA,
            "data_integrity_path",
            "data_integrity_sha256",
        ),
        (
            RANK1_DATE,
            "match_date_path",
            "match_date_sha256",
        ),
        (
            RANK1_HTML,
            "match_page_path",
            "match_page_sha256",
        ),
    )

    for path, path_key, hash_key in expected_files:
        require(
            path.is_file()
            and not path.is_symlink()
            and decision["evidence"][path_key]
            == path.relative_to(ROOT).as_posix()
            and decision["evidence"][hash_key]
            == sha256_file(path),
            f"Rank 1 evidence identity mismatch: {path.name}",
        )

    data = json.loads(
        RANK1_DATA.read_text(encoding="utf-8")
    )

    # Checks the original 24 pinned source-file identities.
    # Does not reparse the Rank 1 Demo.
    validate_payload(data)

    require(
        data["observation"]["demo_sha256"]
        == decision["verified_observations"]["demo_sha256"]
        == EXPECTED_RANK1_DEMO_SHA256,
        "Rank 1 Demo identity differs across evidence records.",
    )

    require(
        decision["scope_limitations"][
            "legacy_development_match_id_coverage"
        ] == "INCOMPLETE"
        and decision["scope_limitations"][
            "missing_historical_ids_assumed_nonoverlapping"
        ] is False
        and decision["scope_limitations"][
            "future_v4_demo_uniqueness_checks_required"
        ] is True
        and decision["final_confirmation_manifest_created"] is False
        and decision["model_scoring_performed"] is False,
        "Rank 1 scientific limitations were not preserved.",
    )

    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            decision["review_source_commit"],
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )

    return decision


def main():
    require(
        git("branch", "--show-current") == "main"
        and git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "Repository must be synchronized on main.",
    )

    subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=True,
    )

    require(
        not (ROOT / "docs/v4_a_confirm_manifest_v1.csv").exists(),
        "Final Confirmation Manifest already exists.",
    )

    queue, _ = read_frozen_queue()
    initial = load_initial_candidates()

    require(
        len(queue) == len(initial) == 25,
        "Unexpected frozen candidate count.",
    )

    for index, (queued, ledger) in enumerate(
        zip(queue, initial),
        start=1,
    ):
        require(
            int(queued["candidate_rank"])
            == ledger["candidate_rank"]
            == index
            and queued["source_match_id"]
            == ledger["source_match_id"],
            "Frozen queue/initial ledger mismatch.",
        )

    decision = verify_rank1()

    require(
        decision["source_match_id"]
        == initial[0]["source_match_id"],
        "Rank 1 decision differs from frozen candidate.",
    )

    next_row = initial[1]

    require(
        next_row["candidate_rank"] == 2
        and next_row["source_match_id"] == "2397692"
        and next_row["initial_status"] == "PENDING",
        "Next frozen candidate is not Rank 2.",
    )

    rank2_staging = STAGING_ROOT / rank_dir_name(next_row)

    require(
        not rank2_staging.exists()
        and not rank2_staging.is_symlink(),
        "Rank 2 staging already exists; inspect it before proceeding.",
    )

    rank2_archive_event = ROOT / (
        "docs/v4_a_confirm_intake_events_v1/"
        "rank_02_2397692_archive_acquired.json"
    )

    require(
        not rank2_archive_event.exists()
        and not rank2_archive_event.is_symlink(),
        "Rank 2 Archive Event already exists.",
    )

    free_bytes = shutil.disk_usage(ROOT).free

    require(
        free_bytes >= MINIMUM_FREE_BYTES,
        "Available disk is below the frozen 12 GiB threshold.",
    )

    print("=== V4-A RANK-ORDER GUARD V2 ===")
    print("Frozen Queue / Initial Ledger: VERIFIED")
    print("Rank 1 Decision: ELIGIBLE — VERIFIED")
    print("Rank 1 pinned Evidence Sources: VERIFIED")
    print("Eligible matches recorded: 1 / 20")
    print()
    print("Next Candidate Rank:", next_row["candidate_rank"])
    print("Next Match ID:", next_row["source_match_id"])
    print("Rank 2 staging: ABSENT")
    print("Rank 2 Archive Event: ABSENT")
    print("Available disk: %.2f GiB" % (free_bytes / 1024**3))
    print()
    print("STATUS: RANK2_PIPELINE_PREPARATION_READY")
    print("Download authorization granted by this guard: NO")
    print("Rank 2 technical eligibility: NOT EVALUATED")
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
        subprocess.CalledProcessError,
    ) as exc:
        print(
            "RANK-ORDER GUARD STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
