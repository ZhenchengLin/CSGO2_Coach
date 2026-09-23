#!/usr/bin/env python3
"""Generic V4-A RAR selected-member Extraction Evidence Recorder.

Read-only by default. --record explicitly creates Evidence.

Replays the selected Archive member into a bounded memory stream,
compares it with the existing Demo, and verifies the actual map.

Does not download or extract another Demo, independently measure the
raw tick clock, decide eligibility, create a Manifest, or score models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHUNK_SIZE = 1024 * 1024
MAX_MEMBER_BYTES = 1024**3
REPLAY_TIMEOUT_SECONDS = 300

VERSION = "V4_A_CONFIRM_SELECTED_DEMO_EXTRACTION_V1"


class EvidenceStop(RuntimeError):
    """Stop when verification or evidence integrity is uncertain."""


def require(condition, message):
    if not condition:
        raise EvidenceStop(message)


def digest_file(path):
    path = Path(path)

    require(
        path.is_file() and not path.is_symlink(),
        f"Missing or unsafe file: {path}",
    )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(CHUNK_SIZE),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def replay_member(
    command,
    declared_bytes,
    timeout=REPLAY_TIMEOUT_SECONDS,
):
    """Hash one bounded subprocess stream without writing a Demo."""

    require(
        isinstance(declared_bytes, int)
        and not isinstance(declared_bytes, bool)
        and 0 < declared_bytes <= MAX_MEMBER_BYTES,
        "Invalid declared selected-member size.",
    )

    require(
        timeout > 0,
        "Invalid replay timeout.",
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    digest = hashlib.sha256()
    total = 0
    deadline = time.monotonic() + timeout

    try:
        require(
            process.stdout is not None,
            "Missing selected-member replay stream.",
        )

        fd = process.stdout.fileno()

        while True:
            remaining = deadline - time.monotonic()

            require(
                remaining > 0,
                "Selected-member replay timed out.",
            )

            readable, _, _ = select.select(
                [fd],
                [],
                [],
                min(remaining, 10.0),
            )

            if not readable:
                continue

            block = os.read(fd, CHUNK_SIZE)

            if not block:
                break

            require(
                total + len(block) <= declared_bytes,
                "Replayed member exceeded declared size.",
            )

            total += len(block)
            digest.update(block)

        remaining = deadline - time.monotonic()

        require(
            remaining > 0,
            "Selected-member replay timed out.",
        )

        returncode = process.wait(timeout=remaining)

        require(
            returncode == 0,
            f"Selected-member replay returned {returncode}.",
        )

        require(
            total == declared_bytes,
            "Replayed member size mismatch.",
        )

        return {
            "size_bytes": total,
            "sha256": digest.hexdigest(),
        }

    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

        if process.stdout is not None:
            process.stdout.close()


def publish_record(path, record):
    """Publish a JSON record without replacing an existing file."""

    path = Path(path)

    require(
        path.parent.is_dir()
        and not path.parent.is_symlink(),
        "Unsafe Evidence directory.",
    )

    require(
        not path.exists() and not path.is_symlink(),
        "Extraction Evidence already exists.",
    )

    payload = (
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    require(
        not temporary.exists() and not temporary.is_symlink(),
        "Temporary Evidence path already exists.",
    )

    temporary_created = False

    try:
        with temporary.open("xb") as handle:
            temporary_created = True

            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # Same-filesystem, create-only publication.
        # An existing destination will not be overwritten.
        os.link(temporary, path)

    finally:
        if temporary_created and temporary.exists():
            temporary.unlink()

    return hashlib.sha256(payload).hexdigest()


def verify_rank(rank):
    """Verify an existing Demo against its frozen Archive provenance."""

    sys.path.insert(0, str(ROOT / "scripts"))

    from inspect_v4_a_confirm_rar_inventory_v1 import (
        inspect_rank,
    )

    from plan_v4_a_confirm_intake_v1 import (
        resolve_plan,
    )

    from cs2_tactical_intelligence.v0.timing import (
        open_v0_demo,
    )

    plan = resolve_plan()

    require(
        plan["rank"] == rank
        and plan["stage"] == "DEMO_EXTRACTION_PENDING",
        "Rank is not at the authorized Extraction Evidence stage.",
    )

    inventory = inspect_rank(rank)

    match_id = inventory["source_match_id"]
    member = inventory["selected_member"]
    declared = inventory["selected_member_declared_bytes"]

    staging = ROOT / (
        "data/raw/v4_a_confirm_download_staging/"
        f"rank_{rank:02d}_{match_id}"
    )

    archive = staging / "archive.rar"
    demo = staging / member

    require(
        staging.is_dir() and not staging.is_symlink(),
        "Missing or unsafe staging directory.",
    )

    require(
        archive.is_file() and not archive.is_symlink(),
        "Missing or unsafe source Archive.",
    )

    require(
        demo.is_file() and not demo.is_symlink(),
        "Missing or unsafe selected Demo.",
    )

    require(
        demo.stat().st_size == declared,
        "Selected Demo size differs from Archive inventory.",
    )

    require(
        not list(staging.glob("*.partial.*")),
        "Unresolved partial extraction requires review.",
    )

    evidence_relative = (
        "docs/v4_a_confirm_intake_audits/"
        f"rank_{rank:02d}_selected_demo_extraction_v1.json"
    )

    evidence_path = ROOT / evidence_relative

    require(
        not evidence_path.exists()
        and not evidence_path.is_symlink(),
        "Extraction Evidence already exists.",
    )

    demo_sha = digest_file(demo)

    # Independently read the selected member from the original Archive.
    # stdout is hashed in memory; no second Demo file is created.
    replay = replay_member(
        [
            "bsdtar",
            "-xOf",
            str(archive),
            member,
        ],
        declared,
    )

    require(
        replay["sha256"] == demo_sha,
        "Archive replay SHA256 differs from extracted Demo.",
    )

    # Verify both files again after the replay.
    require(
        demo.stat().st_size == declared
        and digest_file(demo) == demo_sha,
        "Extracted Demo changed during verification.",
    )

    require(
        archive.stat().st_size
        == inventory["source_archive_size_bytes"]
        and digest_file(archive)
        == inventory["source_archive_sha256"],
        "Original Archive changed during verification.",
    )

    # Header verification does not independently measure the raw clock.
    parsed_demo = open_v0_demo(
        demo,
        verbose=False,
    )

    actual_map = parsed_demo.header.get("map_name")

    require(
        actual_map == "de_mirage",
        f"Unexpected actual Demo Header map: {actual_map!r}",
    )

    event_relative = (
        "docs/v4_a_confirm_intake_events_v1/"
        f"rank_{rank:02d}_{match_id}_archive_acquired.json"
    )

    source_relative = (
        "docs/v4_a_confirm_intake_audits/"
        f"rank_{rank:02d}_match_page_source_v1.html"
    )

    record = {
        "version": VERSION,
        "record_type": "SELECTED_MEMBER_EXTRACTION_EVIDENCE",
        "candidate_rank": rank,
        "source_match_id": match_id,
        "archive_event_path": event_relative,
        "archive_event_sha256": digest_file(
            ROOT / event_relative
        ),
        "source_snapshot_path": source_relative,
        "source_snapshot_sha256": digest_file(
            ROOT / source_relative
        ),
        "source_archive_path": inventory["source_archive_path"],
        "source_archive_size_bytes": (
            inventory["source_archive_size_bytes"]
        ),
        "source_archive_sha256": (
            inventory["source_archive_sha256"]
        ),
        "archive_members": inventory["archive_members"],
        "selected_member": member,
        "selected_member_declared_bytes": declared,
        "extracted_demo_path": demo.relative_to(ROOT).as_posix(),
        "extracted_demo_size_bytes": declared,
        "extracted_demo_sha256": demo_sha,
        "selected_member_replay_sha256": replay["sha256"],
        "selected_member_replay_exit_zero": True,
        "selected_member_stream_matches_extracted_demo": True,
        "actual_demo_header_map": actual_map,
        "demo_header_map_verified": True,
        "whole_archive_crc_verified": False,
        "unselected_members_extracted": False,
        "raw_tick_clock_independently_measured": False,
        "frozen_feature_and_target_pipeline_evaluated": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }

    return evidence_path, record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Explicitly create verified Extraction Evidence.",
    )

    args = parser.parse_args()

    require(
        1 <= args.rank <= 25,
        "Rank must belong to the frozen candidate queue.",
    )

    evidence_path, record = verify_rank(args.rank)

    print("=== V4-A GENERIC RAR EXTRACTION EVIDENCE ===")
    print("Rank:", record["candidate_rank"])
    print("Match ID:", record["source_match_id"])
    print("Selected member:", record["selected_member"])
    print(
        "Verified Demo bytes:",
        record["extracted_demo_size_bytes"],
    )
    print(
        "Verified Demo SHA256:",
        record["extracted_demo_sha256"],
    )
    print("Independent Archive replay: MATCHED")
    print(
        "Actual Demo Header map:",
        record["actual_demo_header_map"],
    )

    if not args.record:
        print()
        print("MODE: READ-ONLY VERIFICATION")
        print("Extraction Evidence created: NO")

    else:
        record_sha = publish_record(
            evidence_path,
            record,
        )

        print()
        print("Extraction Evidence: CREATED")
        print(
            "Evidence path:",
            evidence_path.relative_to(ROOT),
        )
        print("Evidence SHA256:", record_sha)

    print("Independent raw tick clock: NOT MEASURED")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()

    except (
        EvidenceStop,
        RuntimeError,
        KeyError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:

        print(
            "EXTRACTION EVIDENCE STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
