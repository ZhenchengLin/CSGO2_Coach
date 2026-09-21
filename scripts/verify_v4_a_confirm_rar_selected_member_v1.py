#!/usr/bin/env python3

"""
V4-A Confirmation: Rank 1 RAR selected-member verification.

Reads the verified original RAR and independently replays
the already extracted Mirage member into a SHA256 digest.

This verifier does not:
    extract a second Demo;
    claim whole-archive CRC verification;
    select a match for the final corpus;
    assign technical eligibility;
    load models or calculate confirmation metrics.

Default: read-only verification.
--record: exclusively create a RAR-specific extraction record
          after the same verification succeeds.

Rank 1's known RAR inventory and selected member are explicit.
Future candidates require their own verified inventory; do
not silently reuse these Rank 1 constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import subprocess
import time

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

SELECTED_MEMBER = "brute-vs-sparta-m2-mirage.dem"
SELECTED_SIZE = 328_763_831

EXPECTED_DEMO_SHA256 = (
    "848a9e9d8b0593e3684a19f681d4f9c3902a9490ae530fe20667cdb7cc5843b1"
)

EXPECTED_ARCHIVE_SHA256 = (
    "e2b47081d53f1aad01976998bfafe3fd796814eb96353f8d0cd3a9848a037d76"
)

EXPECTED_MEMBERS = {
    "brute-vs-sparta-m2-mirage.dem": 328_763_831,
    "brute-vs-sparta-m1-dust2.dem": 367_052_154,
}

RECORD_NAME = "rar_selected_extraction_record_v1.json"
RECORD_VERSION = "V4_A_CONFIRM_RAR_SELECTED_EXTRACTION_V1"

STREAM_CHUNK = 1024 * 1024
STREAM_TIMEOUT_SECONDS = 300


def run_listing(archive, *, verbose=False):
    """Read and validate the exact known Rank 1 RAR inventory."""

    names = subprocess.run(
        ["bsdtar", "-tf", str(archive)],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    ).stdout.splitlines()

    require(
        len(names) == len(EXPECTED_MEMBERS)
        and len(set(names)) == len(names)
        and set(names) == set(EXPECTED_MEMBERS),
        "RAR_REVIEW_REQUIRED: unexpected member inventory.",
    )

    detailed = subprocess.run(
        ["bsdtar", "-tvf", str(archive)],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    ).stdout.splitlines()

    sizes = {}

    for line in detailed:
        fields = line.split()

        require(
            len(fields) == 9,
            "RAR_REVIEW_REQUIRED: cannot parse member metadata.",
        )

        # Observed bsdtar verbose listing:
        # permissions owner group link_count size month day time name
        # Rank 1 filenames contain no whitespace.
        name = fields[-1]

        require(
            name in EXPECTED_MEMBERS
            and name not in sizes
            and fields[0].startswith("-"),
            "RAR_REVIEW_REQUIRED: unexpected member type or name.",
        )

        try:
            size = int(fields[4])
        except ValueError as exc:
            raise RuntimeError(
                "RAR_REVIEW_REQUIRED: invalid member size."
            ) from exc

        sizes[name] = size

    require(
        sizes == EXPECTED_MEMBERS,
        "RAR_REVIEW_REQUIRED: member sizes differ "
        "from the previously inspected Rank 1 inventory.",
    )

    if verbose:
        for name in names:
            print(f"RAR member: {name} | {sizes[name]:,} bytes")

    return names


def replay_selected_member_sha256(archive):
    """
    Independently read the selected member from the RAR.

    Pipe output is consumed incrementally. No second Demo
    file is written and output cannot exceed the known size.

    bsdtar's successful exit and matching full-stream SHA256
    establish selected-member consistency, not a claim that
    every other archive member passed a CRC audit.
    """

    process = subprocess.Popen(
        [
            "bsdtar",
            "-xOf",
            str(archive),
            SELECTED_MEMBER,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    digest = hashlib.sha256()
    total = 0
    deadline = time.monotonic() + STREAM_TIMEOUT_SECONDS

    try:
        require(
            process.stdout is not None,
            "RAR_REVIEW_REQUIRED: missing extraction stream.",
        )

        fd = process.stdout.fileno()

        while True:
            remaining = deadline - time.monotonic()

            require(
                remaining > 0,
                "RAR_REVIEW_REQUIRED: replay timed out.",
            )

            ready, _, _ = select.select(
                [fd],
                [],
                [],
                min(remaining, 10.0),
            )

            if not ready:
                continue

            chunk = os.read(fd, STREAM_CHUNK)

            if not chunk:
                break

            total += len(chunk)

            require(
                total <= SELECTED_SIZE,
                "RAR_REVIEW_REQUIRED: member expanded "
                "beyond its verified listed size.",
            )

            digest.update(chunk)

        remaining = deadline - time.monotonic()

        require(
            remaining > 0,
            "RAR_REVIEW_REQUIRED: replay timed out.",
        )

        returncode = process.wait(timeout=remaining)

        require(
            returncode == 0,
            f"RAR_REVIEW_REQUIRED: bsdtar returned {returncode}.",
        )

        require(
            total == SELECTED_SIZE,
            "RAR_REVIEW_REQUIRED: replay size mismatch.",
        )

        return total, digest.hexdigest()

    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

        if process.stdout is not None:
            process.stdout.close()


def verify():
    """Verify source provenance, local Demo, and RAR replay."""

    require(
        shutil.which("bsdtar") is not None,
        "RAR_REVIEW_REQUIRED: bsdtar is unavailable.",
    )

    candidates = load_initial_candidates()

    row = candidates[RANK - 1]

    require(
        row["candidate_rank"] == RANK
        and row["source_match_id"] == MATCH_ID,
        "Frozen Rank 1 identity mismatch.",
    )

    event = read_event(row)

    require(
        event is not None
        and event["status"] == "ARCHIVE_ACQUIRED",
        "Verified Archive Acquisition Event is required.",
    )

    staging = STAGING_ROOT / rank_dir_name(row)

    require(
        staging.is_dir() and not staging.is_symlink(),
        "Invalid Rank 1 staging directory.",
    )

    archive = staging / "archive.rar"
    demo = staging / SELECTED_MEMBER

    for path in (archive, demo):
        require(
            path.is_file() and not path.is_symlink(),
            f"Missing or invalid file: {path}",
        )

    require(
        event["archive_sha256"] == EXPECTED_ARCHIVE_SHA256
        and event["archive_size_bytes"] == archive.stat().st_size,
        "Archive differs from committed Acquisition Event.",
    )

    # read_event() already verifies the acquisition record
    # and rehashes the complete original Archive.
    print("Frozen Rank 1 and Archive Event: VERIFIED")

    names = run_listing(archive, verbose=True)

    require(
        demo.stat().st_size == SELECTED_SIZE,
        "Extracted Mirage Demo size mismatch.",
    )

    local_sha = sha256_file(demo)

    require(
        local_sha == EXPECTED_DEMO_SHA256,
        "Extracted Mirage Demo SHA256 mismatch.",
    )

    print("Existing Mirage Demo SHA256: VERIFIED")

    replay_size, replay_sha = replay_selected_member_sha256(
        archive
    )

    require(
        replay_sha == local_sha,
        "RAR_REVIEW_REQUIRED: archive member replay "
        "does not match the existing extracted Demo.",
    )

    print("RAR selected-member full replay: VERIFIED")
    print("Replayed bytes:", replay_size)
    print("Replayed SHA256:", replay_sha)

    # The source Archive remains unchanged throughout
    # the selected-member verification.
    require(
        sha256_file(archive) == EXPECTED_ARCHIVE_SHA256,
        "Original Archive changed during verification.",
    )

    print("Original Archive after replay: VERIFIED")

    record = {
        "version": RECORD_VERSION,
        "candidate_rank": RANK,
        "source_match_id": MATCH_ID,
        "source_archive_path": str(archive.relative_to(ROOT)),
        "source_archive_size_bytes": archive.stat().st_size,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_members": [
            {
                "path": name,
                "declared_uncompressed_bytes": EXPECTED_MEMBERS[name],
            }
            for name in names
        ],
        "selected_member": SELECTED_MEMBER,
        "selected_member_declared_bytes": SELECTED_SIZE,
        "extracted_demo_path": str(demo.relative_to(ROOT)),
        "extracted_demo_size_bytes": demo.stat().st_size,
        "extracted_demo_sha256": local_sha,
        "selected_member_replay_sha256": replay_sha,
        "selected_member_replay_exit_zero": True,
        "selected_member_stream_matches_extracted_demo": True,
        "whole_archive_crc_verified": False,
        "unselected_members_extracted": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }

    return staging, record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--record",
        action="store_true",
        help="Create the RAR-specific record after verification.",
    )

    args = parser.parse_args()

    print("=== V4 RANK 1 RAR SELECTED-MEMBER VERIFICATION ===")

    staging, record = verify()

    record_path = staging / RECORD_NAME

    if not args.record:
        print()
        print("STATUS: RAR_SELECTED_MEMBER_VERIFIED")
        print("RAR Extraction Record: NOT WRITTEN")
        print("Whole-archive CRC: NOT ESTABLISHED")
        print("Technical eligibility: NOT EVALUATED")
        print("Model scoring: NONE")
        return

    require(
        not record_path.exists() and not record_path.is_symlink(),
        "RAR extraction record already exists. "
        "Do not overwrite it.",
    )

    # Exclusive creation: an existing record is never
    # replaced, even if this command is run twice.
    payload = (
        json.dumps(record, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

    with record_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    print()
    print("STATUS: RAR_SELECTED_EXTRACTION_RECORDED")
    print("Record:", record_path.relative_to(ROOT))
    print("Record SHA256:", hashlib.sha256(payload).hexdigest())
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    main()
