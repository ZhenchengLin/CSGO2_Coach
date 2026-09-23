#!/usr/bin/env python3
"""Read-only, resumable V4-A Confirmation Intake planner.

Reconstructs the next permitted stage from the frozen queue
and existing evidence.

No network access, file mutation, Demo extraction,
eligibility decision or model scoring is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AUDITS = ROOT / "docs/v4_a_confirm_intake_audits"

EVENTS = ROOT / "docs/v4_a_confirm_intake_events_v1"

STAGING = ROOT / "data/raw/v4_a_confirm_download_staging"

QUEUE = ROOT / "docs/v4_a_confirm_acquisition_queue_v1.csv"

PROTOCOL = ROOT / "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json"

MANIFEST = ROOT / "docs/v4_a_confirm_manifest_v1.csv"

QUEUE_SHA = (
    "3eddd2dbcf038911b5e9bb41b2a5e218e710fec5a9f3f4ea3b759028f1a829bc"
)

PROTOCOL_SHA = (
    "5bd7e756ac78589e8b60d23521e1d133a722e08d2cd975e7129ffe03d130c3e4"
)


class IntakeStop(RuntimeError):
    """Stop rather than guess or repair inconsistent evidence."""


def require(condition, message):
    if not condition:
        raise IntakeStop(message)


def safe_file(path):
    return path.is_file() and not path.is_symlink()


def sha256_file(path):
    require(safe_file(path), f"Missing or unsafe file: {path}")

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


def classify_stage(
    *,
    source,
    staging,
    event,
    extraction,
    audit,
):
    """Pure state classifier: no filesystem or network operations."""

    require(
        not staging or source,
        "Archive staging exists without Source Evidence.",
    )

    require(
        not event or staging,
        "Acquisition Event exists without Archive staging.",
    )

    require(
        not extraction or event,
        "Extraction Evidence exists without Acquisition Event.",
    )

    require(
        not audit or extraction,
        "Technical Audit exists without Extraction Evidence.",
    )

    if audit:
        return "FORMAL_DECISION_PENDING"

    if extraction:
        return "TECHNICAL_AUDIT_PENDING"

    if event:
        return "DEMO_EXTRACTION_PENDING"

    if staging:
        return "ARCHIVE_EVENT_PENDING"

    if source:
        return "SOURCE_REVIEW_PENDING"

    return "SOURCE_CAPTURE_PENDING"


def verify_decision(row, rank, allowed_reasons):
    path = AUDITS / (
        f"rank_{rank:02d}_technical_eligibility_v1.json"
    )

    if not path.exists() and not path.is_symlink():
        return None

    require(
        safe_file(path),
        f"Rank {rank}: unsafe Formal Decision path.",
    )

    decision = json.loads(path.read_text(encoding="utf-8"))

    require(
        decision.get("record_type")
        == "FORMAL_TECHNICAL_ELIGIBILITY_DECISION"
        and decision.get("candidate_rank") == rank
        and decision.get("source_match_id")
        == row["source_match_id"],
        f"Rank {rank}: Formal Decision identity mismatch.",
    )

    status = decision.get("decision")

    require(
        status in {"ELIGIBLE", "EXCLUDED"},
        f"Rank {rank}: unknown Decision status.",
    )

    if status == "EXCLUDED":
        require(
            decision.get("exclusion_reason") in allowed_reasons,
            f"Rank {rank}: unauthorized exclusion reason.",
        )

    require(
        decision.get("model_scoring_performed") is False
        and decision.get("final_confirmation_manifest_created") is False,
        f"Rank {rank}: scoring boundary violation.",
    )

    evidence = decision.get("evidence")

    require(
        isinstance(evidence, dict)
        and evidence.get("acquisition_protocol_sha256")
        == PROTOCOL_SHA,
        f"Rank {rank}: frozen protocol provenance mismatch.",
    )

    checked = 0

    # Check previously committed metadata without reparsing
    # already audited large Demo files.
    for key, value in evidence.items():

        if not key.endswith("_path"):
            continue

        prefix = key[:-5]

        if prefix not in {
            "match_page",
            "match_date",
            "map_source",
            "archive_event",
            "selected_demo_extraction",
            "data_integrity",
        }:
            continue

        expected_sha = evidence.get(prefix + "_sha256")

        require(
            isinstance(value, str)
            and isinstance(expected_sha, str)
            and len(expected_sha) == 64,
            f"Rank {rank}: incomplete {prefix} provenance.",
        )

        relative = Path(value)

        require(
            not relative.is_absolute()
            and len(relative.parts) >= 2
            and relative.parts[0] == "docs"
            and ".." not in relative.parts,
            f"Rank {rank}: unsafe evidence reference.",
        )

        actual_sha = sha256_file(ROOT / relative)

        require(
            actual_sha == expected_sha,
            f"Rank {rank}: {prefix} SHA256 mismatch.",
        )

        checked += 1

    require(
        checked > 0,
        f"Rank {rank}: no verifiable metadata evidence.",
    )

    return status


def verify_committed_source(path):
    require(
        safe_file(path),
        "Active Rank Source Evidence missing or unsafe.",
    )

    relative = path.relative_to(ROOT).as_posix()

    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
        )

    except subprocess.CalledProcessError as exc:
        raise IntakeStop(
            "Active Rank Source Evidence is not committed."
        ) from exc

    require(
        hashlib.sha256(committed).hexdigest()
        == sha256_file(path),
        "Active Rank Source differs from committed evidence.",
    )


def resolve_plan():
    """Find the earliest undecided Rank and validate its state."""

    sys.path.insert(0, str(ROOT / "scripts"))

    from probe_v4_a_confirm_demo_source_v1 import (
        read_frozen_queue,
    )

    from manage_v4_a_confirm_intake_state_v1 import (
        load_initial_candidates,
        validate_event,
        verify_archive_record,
    )

    require(
        not MANIFEST.exists() and not MANIFEST.is_symlink(),
        "Final Confirmation Manifest already exists.",
    )

    require(
        sha256_file(QUEUE) == QUEUE_SHA,
        "Frozen Queue SHA256 mismatch.",
    )

    require(
        sha256_file(PROTOCOL) == PROTOCOL_SHA,
        "Frozen Protocol SHA256 mismatch.",
    )

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    require(
        protocol["domain"]["map"] == "de_mirage",
        "Unexpected frozen map requirement.",
    )

    rows, _ = read_frozen_queue()

    ledger = load_initial_candidates()

    require(
        len(rows) == len(ledger) == 25,
        "Frozen candidate count mismatch.",
    )

    statuses = []
    active = None

    for rank, (row, initial) in enumerate(
        zip(rows, ledger),
        start=1,
    ):
        require(
            int(row["candidate_rank"]) == rank
            and initial["candidate_rank"] == rank
            and row["source_match_id"]
            == initial["source_match_id"],
            f"Rank {rank}: Queue / Ledger mismatch.",
        )

        status = verify_decision(
            row,
            rank,
            protocol["eligibility"]["allowed_exclusion_reasons"],
        )

        if status is not None:
            require(
                active is None,
                "Later Rank has a Formal Decision while "
                "an earlier Rank is unfinished.",
            )

            statuses.append(status)
            continue

        if active is None:
            active = (rank, row, initial)

    eligible = statuses.count("ELIGIBLE")
    excluded = statuses.count("EXCLUDED")

    require(
        eligible <= 20,
        "Eligible count exceeds frozen sample target.",
    )

    if active is None:
        return {
            "rank": None,
            "match_id": None,
            "stage": "QUEUE_EXHAUSTED",
            "eligible": eligible,
            "excluded": excluded,
        }

    rank, row, initial = active

    match_id = row["source_match_id"]

    source = AUDITS / (
        f"rank_{rank:02d}_match_page_source_v1.html"
    )

    directory = STAGING / f"rank_{rank:02d}_{match_id}"

    event_path = EVENTS / (
        f"rank_{rank:02d}_{match_id}_archive_acquired.json"
    )

    extraction = AUDITS / (
        f"rank_{rank:02d}_selected_demo_extraction_v1.json"
    )

    audit = AUDITS / (
        f"rank_{rank:02d}_data_integrity_v1.json"
    )

    def present(path):
        return path.exists() or path.is_symlink()

    has_source = present(source)
    has_staging = present(directory)
    has_event = present(event_path)
    has_extraction = present(extraction)
    has_audit = present(audit)

    stage = classify_stage(
        source=has_source,
        staging=has_staging,
        event=has_event,
        extraction=has_extraction,
        audit=has_audit,
    )

    if has_source:
        verify_committed_source(source)

    if has_staging:
        require(
            directory.is_dir() and not directory.is_symlink(),
            "Active Rank staging directory is unsafe.",
        )

        # Existing Archive is verified, never redownloaded.
        archive = verify_archive_record(initial)

        acquisition_record = json.loads(
            (directory / "acquisition_record.json").read_text(
                encoding="utf-8"
            )
        )

        require(
            acquisition_record.get("match_page_sha256")
            == sha256_file(source),
            "Archive provenance differs from committed Source.",
        )

        if has_event:
            require(
                safe_file(event_path),
                "Acquisition Event is missing or unsafe.",
            )

            event = json.loads(
                event_path.read_text(encoding="utf-8")
            )

            validate_event(initial, event, archive)

            # The Event must be committed before continuing.
            relative = event_path.relative_to(ROOT).as_posix()

            committed_event = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"],
                cwd=ROOT,
            )

            require(
                hashlib.sha256(committed_event).hexdigest()
                == sha256_file(event_path),
                "Acquisition Event differs from committed evidence.",
            )

    for label, path, exists in (
        ("Extraction", extraction, has_extraction),
        ("Technical Audit", audit, has_audit),
    ):
        if exists:
            require(
                safe_file(path),
                f"{label} Evidence path is unsafe.",
            )

            data = json.loads(path.read_text(encoding="utf-8"))

            require(
                data.get("candidate_rank") == rank
                and data.get("source_match_id") == match_id
                and data.get("model_scoring_performed") is False,
                f"{label} Evidence identity/scope mismatch.",
            )

    return {
        "rank": rank,
        "match_id": match_id,
        "stage": stage,
        "eligible": eligible,
        "excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--status",
        action="store_true",
        help="Read and verify the current Intake state.",
    )

    args = parser.parse_args()

    require(
        git("branch", "--show-current") == "main",
        "Expected main branch.",
    )

    require(
        git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "Local HEAD differs from origin/main.",
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

    result = resolve_plan()

    print("=== V4-A CONFIRMATION INTAKE PLANNER V1 ===")
    print("Frozen Queue / Protocol: VERIFIED")
    print("Completed eligible:", result["eligible"], "/ 20")
    print("Completed excluded:", result["excluded"])
    print("Current Rank:", result["rank"])
    print("Match ID:", result["match_id"])
    print("NEXT STAGE:", result["stage"])
    print("Archive downloads: NONE")
    print("Existing files modified: NONE")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()

    except (
        IntakeStop,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:

        print("INTAKE PLANNER STOP:", exc, file=sys.stderr)

        raise SystemExit(1)
