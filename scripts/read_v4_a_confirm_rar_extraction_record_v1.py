#!/usr/bin/env python3

"""
Independent reader for the V4-A Rank 1 RAR Extraction Record.

This module verifies an already created record. It never
creates, replaces, repairs, or approves an extraction record.

The selected-member record establishes provenance for one
extracted Demo. It does not claim whole-archive CRC verification,
technical eligibility, or permission to run model scoring.

Rank 1 constants are explicit. Future ranks require a
separately reviewed, generalized extraction contract.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    STAGING_ROOT,
    load_initial_candidates,
    rank_dir_name,
    read_event,
    require,
    sha256_file,
)


RANK = 1
MATCH_ID = "2397691"

MEMBER = "brute-vs-sparta-m2-mirage.dem"
DEMO_BYTES = 328_763_831

DEMO_SHA256 = (
    "848a9e9d8b0593e3684a19f681d4f9c3902a9490ae530fe20667cdb7cc5843b1"
)

ARCHIVE_SHA256 = (
    "e2b47081d53f1aad01976998bfafe3fd796814eb96353f8d0cd3a9848a037d76"
)

MEMBERS = [
    {
        "path": "brute-vs-sparta-m2-mirage.dem",
        "declared_uncompressed_bytes": 328_763_831,
    },
    {
        "path": "brute-vs-sparta-m1-dust2.dem",
        "declared_uncompressed_bytes": 367_052_154,
    },
]

RECORD_NAME = "rar_selected_extraction_record_v1.json"

RECORD_VERSION = (
    "V4_A_CONFIRM_RAR_SELECTED_EXTRACTION_V1"
)

EXPECTED_FIELDS = {
    "version",
    "candidate_rank",
    "source_match_id",
    "source_archive_path",
    "source_archive_size_bytes",
    "source_archive_sha256",
    "archive_members",
    "selected_member",
    "selected_member_declared_bytes",
    "extracted_demo_path",
    "extracted_demo_size_bytes",
    "extracted_demo_sha256",
    "selected_member_replay_sha256",
    "selected_member_replay_exit_zero",
    "selected_member_stream_matches_extracted_demo",
    "whole_archive_crc_verified",
    "unselected_members_extracted",
    "technical_eligibility_evaluated",
    "model_scoring_performed",
}


def validate_record(
    record,
    event,
    *,
    expected_archive_path,
    expected_demo_path,
):
    """
    Validate the record against independently established
    Rank 1 identities and the verified Acquisition Event.

    This function does not trust paths supplied by the record
    to choose which files are read from disk.
    """

    require(
        isinstance(record, dict)
        and set(record) == EXPECTED_FIELDS,
        "RAR_RECORD_REVIEW_REQUIRED: unexpected record schema.",
    )

    require(
        record["version"] == RECORD_VERSION,
        "RAR_RECORD_REVIEW_REQUIRED: version mismatch.",
    )

    require(
        record["candidate_rank"] == RANK
        and record["source_match_id"] == MATCH_ID,
        "RAR_RECORD_REVIEW_REQUIRED: candidate identity mismatch.",
    )

    require(
        record["source_archive_path"] == expected_archive_path
        and record["source_archive_path"] == event["archive_path"],
        "RAR_RECORD_REVIEW_REQUIRED: source Archive path mismatch.",
    )

    require(
        record["source_archive_sha256"] == ARCHIVE_SHA256
        and record["source_archive_sha256"] == event["archive_sha256"]
        and record["source_archive_size_bytes"]
        == event["archive_size_bytes"],
        "RAR_RECORD_REVIEW_REQUIRED: source Archive identity mismatch.",
    )

    require(
        record["archive_members"] == MEMBERS,
        "RAR_RECORD_REVIEW_REQUIRED: member inventory mismatch.",
    )

    require(
        record["selected_member"] == MEMBER
        and record["selected_member_declared_bytes"] == DEMO_BYTES,
        "RAR_RECORD_REVIEW_REQUIRED: selected member mismatch.",
    )

    require(
        record["extracted_demo_path"] == expected_demo_path
        and record["extracted_demo_size_bytes"] == DEMO_BYTES
        and record["extracted_demo_sha256"] == DEMO_SHA256,
        "RAR_RECORD_REVIEW_REQUIRED: extracted Demo identity mismatch.",
    )

    require(
        record["selected_member_replay_sha256"] == DEMO_SHA256
        and record["selected_member_replay_exit_zero"] is True
        and record["selected_member_stream_matches_extracted_demo"] is True,
        "RAR_RECORD_REVIEW_REQUIRED: selected-member replay evidence mismatch.",
    )

    require(
        record["whole_archive_crc_verified"] is False
        and record["unselected_members_extracted"] is False,
        "RAR_RECORD_REVIEW_REQUIRED: unsupported Archive verification claim.",
    )

    require(
        record["technical_eligibility_evaluated"] is False
        and record["model_scoring_performed"] is False,
        "RAR_RECORD_REVIEW_REQUIRED: extraction record crossed "
        "the technical/scoring boundary.",
    )

    return True


def read_verified_rank1_record():
    """
    Recheck the Acquisition Event, original Archive identity,
    extraction record and existing Demo bytes.

    Returns verified Demo metadata, not an eligibility decision.
    """

    row = load_initial_candidates()[RANK - 1]

    require(
        row["candidate_rank"] == RANK
        and row["source_match_id"] == MATCH_ID,
        "Frozen Rank 1 identity mismatch.",
    )

    # The Intake Manager independently rehashes the original
    # Archive while verifying the committed Acquisition Event.
    event = read_event(row)

    require(
        event is not None
        and event["status"] == "ARCHIVE_ACQUIRED",
        "A verified ARCHIVE_ACQUIRED event is required.",
    )

    staging = STAGING_ROOT / rank_dir_name(row)

    require(
        staging.is_dir() and not staging.is_symlink(),
        "Invalid Rank 1 staging directory.",
    )

    archive = staging / "archive.rar"
    demo = staging / MEMBER
    record_path = staging / RECORD_NAME

    for path in (archive, demo, record_path):
        require(
            path.is_file() and not path.is_symlink(),
            f"RAR_RECORD_REVIEW_REQUIRED: missing or invalid {path.name}.",
        )

    # Construct trusted filesystem paths independently;
    # never open an arbitrary path from the JSON record.
    archive_relative = archive.relative_to(ROOT).as_posix()
    demo_relative = demo.relative_to(ROOT).as_posix()

    record = json.loads(
        record_path.read_text(encoding="utf-8")
    )

    validate_record(
        record,
        event,
        expected_archive_path=archive_relative,
        expected_demo_path=demo_relative,
    )

    require(
        archive.stat().st_size == event["archive_size_bytes"],
        "RAR_RECORD_REVIEW_REQUIRED: original Archive size changed.",
    )

    require(
        demo.stat().st_size == DEMO_BYTES,
        "RAR_RECORD_REVIEW_REQUIRED: extracted Demo size changed.",
    )

    actual_sha256 = sha256_file(demo)

    require(
        actual_sha256 == DEMO_SHA256,
        "RAR_RECORD_REVIEW_REQUIRED: extracted Demo bytes changed.",
    )

    return {
        "status": "RAR_SELECTED_EXTRACTION_RECORD_VERIFIED",
        "candidate_rank": RANK,
        "source_match_id": MATCH_ID,
        "demo_path": demo_relative,
        "demo_sha256": actual_sha256,
        "demo_size_bytes": DEMO_BYTES,
        "source_archive_sha256": event["archive_sha256"],
        "whole_archive_crc_verified": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    require(
        args.rank == RANK,
        "Only frozen Rank 1 is supported by this record reader.",
    )

    print("=== V4 RANK 1 RAR EXTRACTION RECORD AUDIT ===")

    row = load_initial_candidates()[RANK - 1]
    staging = STAGING_ROOT / rank_dir_name(row)
    record_path = staging / RECORD_NAME

    if not record_path.exists():
        print("STATUS: WAITING_FOR_RAR_EXTRACTION_RECORD")
        print("Archive Acquisition Event: UNCHANGED")
        print("Technical eligibility: NOT EVALUATED")
        print("Model scoring: NONE")
        return

    verified = read_verified_rank1_record()

    print("STATUS:", verified["status"])
    print("Candidate Rank:", verified["candidate_rank"])
    print("Match ID:", verified["source_match_id"])
    print("Demo:", verified["demo_path"])
    print("Demo SHA256:", verified["demo_sha256"])
    print("Whole-archive CRC: NOT ESTABLISHED")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print("RAR RECORD AUDIT STOP:", exc, file=sys.stderr)
        raise SystemExit(1)
