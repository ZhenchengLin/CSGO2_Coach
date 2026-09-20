#!/usr/bin/env python3

"""
V4-A Confirmation Intake State Manager.

The initial Ledger remains immutable.

An ARCHIVE_ACQUIRED event can be created only after
verifying the downloader's acquisition record and
the complete archive bytes.

No technical eligibility decision or model scoring
is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from init_v4_a_confirm_intake_ledger_v1 import (
    LEDGER,
    validate_initial_state,
)

from probe_v4_a_confirm_demo_source_v1 import (
    ROOT,
    MANIFEST,
)

from download_v4_a_confirm_archive_v1 import (
    STAGING_ROOT,
    archive_extension,
)


EVENT_ROOT = ROOT / (
    "docs/v4_a_confirm_intake_events_v1"
)

EVENT_VERSION = (
    "V4_A_CONFIRM_ARCHIVE_ACQUIRED_EVENT_V1"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_initial_candidates():

    require(
        LEDGER.is_file() and not LEDGER.is_symlink(),
        "Frozen Initial Ledger is missing or invalid.",
    )

    state = json.loads(
        LEDGER.read_text(encoding="utf-8")
    )

    validate_initial_state(state)

    require(
        state["candidate_count"] == 25
        and state["required_eligible_matches"] == 20,
        "Unexpected frozen sample plan.",
    )

    return state["candidates"]


def rank_dir_name(row):

    return (
        f'rank_{row["candidate_rank"]:02d}_'
        f'{row["source_match_id"]}'
    )


def acquisition_directory(row, staging_root):

    return staging_root / rank_dir_name(row)


def event_path(row, event_root):

    return event_root / (
        rank_dir_name(row)
        + "_archive_acquired.json"
    )


def validate_download_url(url):

    require(
        isinstance(url, str),
        "Missing Demo download URL.",
    )

    parsed = urlparse(url)

    require(
        parsed.scheme == "https"
        and parsed.hostname in {
            "hltv.org",
            "www.hltv.org",
        }
        and parsed.username is None
        and parsed.password is None
        and re.fullmatch(
            r"/download/demo/[^/?#]+/?",
            parsed.path,
        ) is not None,
        "Acquisition record contains an invalid Demo URL.",
    )


def verify_archive_record(
    row,
    *,
    repo_root=ROOT,
    staging_root=STAGING_ROOT,
):
    """
    Verify both the acquisition record and archived bytes.

    A record alone is insufficient to establish that
    the archived data still exist and remain unchanged.
    """

    rank_dir = acquisition_directory(
        row,
        staging_root,
    )

    require(
        rank_dir.is_dir() and not rank_dir.is_symlink(),
        "Candidate staging directory is missing or invalid.",
    )

    record_path = (
        rank_dir / "acquisition_record.json"
    )

    require(
        record_path.is_file()
        and not record_path.is_symlink(),
        "Acquisition record is missing or invalid.",
    )

    record_bytes = record_path.read_bytes()

    record = json.loads(
        record_bytes.decode("utf-8")
    )

    rank = row["candidate_rank"]

    match_id = row["source_match_id"]

    require(
        record.get("version")
        == "V4_A_CONFIRM_ARCHIVE_ACQUISITION_V1",
        "Unexpected Acquisition Record version.",
    )

    require(
        record.get("candidate_rank") == rank
        and record.get("source_match_id") == match_id,
        "Acquisition Record does not match the frozen Rank.",
    )

    require(
        record.get("frozen_match_date")
        == row["frozen_match_date"]
        and record.get("frozen_match_url")
        == row["frozen_match_url"],
        "Acquisition Record differs from frozen metadata.",
    )

    require(
        record.get("technical_eligibility")
        == "NOT_EVALUATED"
        and record.get("demo_extracted") is False
        and record.get("model_scoring_performed") is False,
        "Unexpected downstream activity in Acquisition Record.",
    )

    validate_download_url(
        record.get("source_download_url")
    )

    extension = record.get(
        "archive_signature_extension"
    )

    require(
        extension in {".zip", ".rar", ".7z"},
        "Unrecognized archive extension.",
    )

    archive_path = (
        rank_dir / f"archive{extension}"
    )

    require(
        archive_path.is_file()
        and not archive_path.is_symlink(),
        "Archived file is missing or invalid.",
    )

    expected_relative_path = str(
        archive_path.relative_to(repo_root)
    )

    require(
        record.get("archive_path")
        == expected_relative_path,
        "Archive path differs from Acquisition Record.",
    )

    actual_size = archive_path.stat().st_size

    require(
        actual_size > 0
        and actual_size == record.get("archive_size_bytes"),
        "Archive size differs from Acquisition Record.",
    )

    with archive_path.open("rb") as handle:
        signature = handle.read(8)

    require(
        archive_extension(signature) == extension,
        "Archive signature differs from Acquisition Record.",
    )

    actual_sha = sha256_file(
        archive_path
    )

    require(
        actual_sha == record.get("archive_sha256"),
        "Archive SHA256 mismatch.",
    )

    return {
        "record_sha256": sha256_bytes(record_bytes),
        "archive_sha256": actual_sha,
        "archive_size_bytes": actual_size,
        "archive_relative_path": expected_relative_path,
    }


def validate_event(row, event, verified_archive):

    require(
        event.get("version") == EVENT_VERSION,
        "Unexpected Intake Event version.",
    )

    require(
        event.get("status") == "ARCHIVE_ACQUIRED",
        "Unexpected Intake Event status.",
    )

    require(
        event.get("candidate_rank")
        == row["candidate_rank"]
        and event.get("source_match_id")
        == row["source_match_id"],
        "Intake Event does not match frozen Rank.",
    )

    require(
        event.get("acquisition_record_sha256")
        == verified_archive["record_sha256"],
        "Intake Event Acquisition Record identity mismatch.",
    )

    require(
        event.get("archive_sha256")
        == verified_archive["archive_sha256"]
        and event.get("archive_size_bytes")
        == verified_archive["archive_size_bytes"]
        and event.get("archive_path")
        == verified_archive["archive_relative_path"],
        "Intake Event Archive identity mismatch.",
    )

    require(
        event.get("technical_eligibility_evaluated") is False
        and event.get("model_scoring_performed") is False,
        "An Archive Event cannot contain a technical decision.",
    )


def read_event(row):

    path = event_path(
        row,
        EVENT_ROOT,
    )

    if not path.exists():
        return None

    require(
        path.is_file() and not path.is_symlink(),
        "Intake Event path is invalid.",
    )

    event = json.loads(
        path.read_text(encoding="utf-8")
    )

    archive = verify_archive_record(row)

    validate_event(
        row,
        event,
        archive,
    )

    return event


def status():

    candidates = load_initial_candidates()

    expected_event_files = {
        event_path(
            row,
            EVENT_ROOT,
        ).name
        for row in candidates
    }

    if EVENT_ROOT.exists():

        require(
            EVENT_ROOT.is_dir()
            and not EVENT_ROOT.is_symlink(),
            "Intake Event directory is invalid.",
        )

        actual_event_files = {
            path.name
            for path in EVENT_ROOT.iterdir()
        }

        require(
            actual_event_files <= expected_event_files,
            "Unknown file or event found in Intake Event directory.",
        )

    print()
    print("=== V4 CONFIRMATION INTAKE STATE ===")
    print("Frozen Initial Ledger: VERIFIED")

    acquired_count = 0
    review_count = 0

    for row in candidates:

        event = read_event(row)

        rank_dir = acquisition_directory(
            row,
            STAGING_ROOT,
        )

        if event is not None:

            current_status = "ARCHIVE_ACQUIRED"
            acquired_count += 1

        elif rank_dir.exists():

            current_status = "STAGING_REVIEW_REQUIRED"
            review_count += 1

        else:

            current_status = "PENDING"

        print(
            f'{row["candidate_rank"]:02d}'
            f' | {row["source_match_id"]}'
            f' | {current_status}'
        )

    print()
    print("Verified Archive Events:", acquired_count)
    print("Staging items requiring review:", review_count)
    print("Technical eligibility decisions: NONE")
    print("Model scoring: NONE")

    require(
        review_count == 0,
        "Some staging directories have no verified "
        "ARCHIVE_ACQUIRED event. Inspect them before proceeding.",
    )


def sync_archive(rank):

    require(
        not MANIFEST.exists(),
        "Final Confirmation Manifest already exists.",
    )

    require(
        rank == 1,
        "Only Rank 1 is authorized until the "
        "Technical Intake workflow is extended.",
    )

    candidates = load_initial_candidates()

    row = candidates[rank - 1]

    existing = event_path(
        row,
        EVENT_ROOT,
    )

    require(
        not existing.exists(),
        "ARCHIVE_ACQUIRED event already exists. "
        "Use --status to verify the existing event.",
    )

    archive = verify_archive_record(
        row
    )

    event = {
        "version": EVENT_VERSION,
        "status": "ARCHIVE_ACQUIRED",
        "candidate_rank": row["candidate_rank"],
        "source_match_id": row["source_match_id"],
        "acquisition_record_sha256": (
            archive["record_sha256"]
        ),
        "archive_sha256": (
            archive["archive_sha256"]
        ),
        "archive_size_bytes": (
            archive["archive_size_bytes"]
        ),
        "archive_path": (
            archive["archive_relative_path"]
        ),
        "recorded_utc": utc_now(),
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }

    validate_event(
        row,
        event,
        archive,
    )

    EVENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = event_path(
        row,
        EVENT_ROOT,
    )

    payload = (
        json.dumps(
            event,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")

    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    try:

        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

        os.link(
            temporary,
            path,
        )

    finally:

        if temporary.exists():
            temporary.unlink()

    print("ARCHIVE_ACQUIRED event created:", path)
    print("Archive SHA256:", archive["archive_sha256"])
    print("Technical eligibility: NOT EVALUATED")


def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--status",
        action="store_true",
        help="Reconstruct state from the frozen Ledger and events.",
    )

    mode.add_argument(
        "--sync-archive",
        action="store_true",
        help="Verify acquired archive and create an immutable event.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.status:
        status()
    else:
        sync_archive(args.rank)


if __name__ == "__main__":

    try:
        main()

    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:

        print(
            "\nINTAKE STATE STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
