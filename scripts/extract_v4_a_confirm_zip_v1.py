#!/usr/bin/env python3

"""
V4-A Confirmation: Safe ZIP Extraction.

Only Rank 1 is currently authorized.

Production extraction requires a previously verified
ARCHIVE_ACQUIRED event.

The immutable ZIP archive is retained. Extracted files
are written into a new private temporary directory.

The final directory is published only after all files
have been extracted and verified.

This module does not select a map, assign technical
eligibility, modify the frozen queue, or score models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

from datetime import datetime, timezone
from pathlib import Path

from inspect_v4_a_confirm_archive_v1 import (
    inspect_zip,
)

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    STAGING_ROOT,
    MANIFEST,
    load_initial_candidates,
    rank_dir_name,
    read_event,
    require,
    sha256_file,
)

from download_v4_a_confirm_archive_v1 import (
    GIB,
    MINIMUM_FREE_BYTES,
    SAFETY_RESERVE_BYTES,
)


CHUNK_SIZE = 1024 * 1024

MAX_EXTRACTION_BYTES = 2 * GIB

EXTRACT_DIR_NAME = "extracted_v1"

TEMP_DIR_NAME = "extracted_v1.tmp"

RECORD_NAME = "__v4_extraction_record__.json"

EXTRA_RECORD_RESERVE = 1024 * 1024


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def available_bytes():
    return shutil.disk_usage(ROOT).free


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require_disk_space(free_space_fn, upcoming_bytes):

    require(
        free_space_fn()
        >= (
            MINIMUM_FREE_BYTES
            + SAFETY_RESERVE_BYTES
            + upcoming_bytes
        ),
        "STORAGE_BLOCK: insufficient remaining disk space.",
    )


def extraction_budget(free_space_fn):

    free = free_space_fn()

    require(
        free >= MINIMUM_FREE_BYTES,
        "STORAGE_BLOCK: free disk is below 12 GiB.",
    )

    budget = min(
        MAX_EXTRACTION_BYTES,
        free
        - MINIMUM_FREE_BYTES
        - SAFETY_RESERVE_BYTES
        - EXTRA_RECORD_RESERVE,
    )

    require(
        budget > 0,
        "STORAGE_BLOCK: no safe extraction budget remains.",
    )

    return budget


def validate_member_layout(metadata):

    """
    Reject file/directory conflicts before writing.

    Example:
        ZIP member 'a'
        ZIP member 'a/b.dem'

    Both cannot occupy the same filesystem path.
    """

    files = set()
    directories = set()

    for member in metadata["members"]:

        name = member["path"]

        require(
            name.casefold() != RECORD_NAME.casefold(),
            "ARCHIVE_REVIEW_REQUIRED: reserved "
            "extraction record filename.",
        )

        parts = name.split("/")

        files.add(
            name.casefold()
        )

        for index in range(1, len(parts)):

            directories.add(
                "/".join(parts[:index]).casefold()
            )

    require(
        not (files & directories),
        "ARCHIVE_REVIEW_REQUIRED: file/directory "
        "path conflict.",
    )


def extract_zip(
    archive_path,
    output_dir,
    *,
    free_space_fn=available_bytes,
    candidate_rank=None,
    source_match_id=None,
    expected_archive_sha256=None,
):
    """
    Extract a ZIP into a new output directory.

    This function is used directly by offline tests.
    Production CLI verifies the frozen acquisition
    event before invoking it.
    """

    archive_path = Path(archive_path)

    output_dir = Path(output_dir)

    require(
        archive_path.is_file()
        and not archive_path.is_symlink(),
        "ARCHIVE_REVIEW_REQUIRED: invalid ZIP Archive.",
    )

    require(
        archive_path.suffix.lower() == ".zip",
        "ARCHIVE_FORMAT_REVIEW_REQUIRED: ZIP only.",
    )

    require(
        not output_dir.exists()
        and not output_dir.is_symlink(),
        "EXTRACTION_ALREADY_EXISTS: refusing overwrite.",
    )

    temp_dir = output_dir.with_name(
        TEMP_DIR_NAME
    )

    require(
        not temp_dir.exists()
        and not temp_dir.is_symlink(),
        "EXTRACTION_REVIEW_REQUIRED: "
        "temporary extraction directory already exists.",
    )

    require(
        output_dir.parent.is_dir()
        and not output_dir.parent.is_symlink(),
        "EXTRACTION_REVIEW_REQUIRED: invalid output parent.",
    )

    budget = extraction_budget(
        free_space_fn
    )

    archive_sha = sha256_file(
        archive_path
    )

    if expected_archive_sha256 is not None:

        require(
            archive_sha == expected_archive_sha256,
            "ARCHIVE_REVIEW_REQUIRED: Archive SHA256 "
            "differs from verified Acquisition Event.",
        )

    metadata = inspect_zip(
        archive_path,
        budget_bytes=budget,
    )

    validate_member_layout(
        metadata
    )

    members = []
    total_written = 0

    created_temp_dir = False

    try:

        temp_dir.mkdir(
            exist_ok=False
        )

        created_temp_dir = True

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            for info in archive.infolist():

                if info.is_dir():
                    continue

                # All paths were validated by inspect_zip.
                name = info.filename

                destination = (
                    temp_dir / name
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                require(
                    destination.parent.resolve()
                    .is_relative_to(temp_dir.resolve()),
                    "ARCHIVE_REVIEW_REQUIRED: unsafe "
                    "extraction destination.",
                )

                require(
                    not destination.exists()
                    and not destination.is_symlink(),
                    "ARCHIVE_REVIEW_REQUIRED: "
                    "extraction destination already exists.",
                )

                written = 0

                digest = hashlib.sha256()

                require_disk_space(
                    free_space_fn,
                    0,
                )

                with (
                    archive.open(info, "r") as source,
                    destination.open("xb") as output,
                ):

                    while True:

                        chunk = source.read(
                            CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        next_total = (
                            total_written + len(chunk)
                        )

                        require(
                            next_total <= budget,
                            "STORAGE_BLOCK: actual extraction "
                            "exceeded authorized budget.",
                        )

                        require(
                            written + len(chunk)
                            <= info.file_size,
                            "ARCHIVE_REVIEW_REQUIRED: member "
                            "expanded beyond declared size.",
                        )

                        require_disk_space(
                            free_space_fn,
                            len(chunk),
                        )

                        output.write(
                            chunk
                        )

                        digest.update(
                            chunk
                        )

                        written += len(chunk)

                        total_written = next_total

                    output.flush()

                    os.fsync(
                        output.fileno()
                    )

                # ZipFile.open() verifies CRC when the
                # complete member has been consumed.
                require(
                    written == info.file_size,
                    "ARCHIVE_REVIEW_REQUIRED: extracted "
                    "member size differs from ZIP metadata.",
                )

                members.append(
                    {
                        "path": name,
                        "size_bytes": written,
                        "sha256": digest.hexdigest(),
                        "is_demo_filename": (
                            name.lower().endswith(".dem")
                        ),
                    }
                )

        require(
            total_written
            == metadata["declared_uncompressed_bytes"],
            "ARCHIVE_REVIEW_REQUIRED: total extracted "
            "size differs from ZIP metadata.",
        )

        require(
            len(members)
            == metadata["member_count"],
            "ARCHIVE_REVIEW_REQUIRED: extracted "
            "member count differs from ZIP metadata.",
        )

        require_disk_space(
            free_space_fn,
            EXTRA_RECORD_RESERVE,
        )

        # Recheck the original Archive after extraction.
        # An Archive changed during extraction cannot
        # produce a completed extraction record.
        require(
            sha256_file(archive_path)
            == archive_sha,
            "ARCHIVE_REVIEW_REQUIRED: original Archive "
            "changed during extraction.",
        )

        record = {
            "version": (
                "V4_A_CONFIRM_ZIP_EXTRACTION_V1"
            ),
            "candidate_rank": candidate_rank,
            "source_match_id": source_match_id,
            "source_archive_sha256": archive_sha,
            "source_archive_size_bytes": (
                archive_path.stat().st_size
            ),
            "extracted_utc": utc_now(),
            "extracted_file_count": len(members),
            "extracted_total_bytes": total_written,
            "members": members,
            "archive_crc_verified": True,
            "technical_eligibility_evaluated": False,
            "model_scoring_performed": False,
        }

        record_path = (
            temp_dir / RECORD_NAME
        )

        with record_path.open(
            "x",
            encoding="utf-8",
        ) as handle:

            json.dump(
                record,
                handle,
                indent=2,
                ensure_ascii=False,
            )

            handle.write("\n")

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        require(
            not output_dir.exists(),
            "EXTRACTION_ALREADY_EXISTS: "
            "output appeared during extraction.",
        )

        temp_dir.rename(
            output_dir
        )

        created_temp_dir = False

        print()
        print("=== ZIP EXTRACTION COMPLETE ===")

        print(
            "Output:",
            output_dir,
        )

        print(
            "Files:",
            len(members),
        )

        print(
            "Extracted bytes:",
            total_written,
        )

        print(
            "Source Archive SHA256:",
            archive_sha,
        )

        print(
            "Extraction Record:",
            output_dir / RECORD_NAME,
        )

        print(
            "Technical eligibility: NOT EVALUATED"
        )

        return record

    except Exception:

        if (
            created_temp_dir
            and temp_dir.is_dir()
            and not temp_dir.is_symlink()
        ):

            # Remove ONLY the temporary directory
            # created by this extraction attempt.
            shutil.rmtree(
                temp_dir
            )

        raise


def inspect_current_rank(rank):

    require(
        rank == 1,
        "RANK_ORDER_BLOCK: only Rank 1 is authorized.",
    )

    candidates = load_initial_candidates()

    row = candidates[rank - 1]

    event = read_event(
        row
    )

    print()
    print("=== FROZEN CANDIDATE ===")

    print(
        "Rank:",
        row["candidate_rank"],
    )

    print(
        "Match ID:",
        row["source_match_id"],
    )

    if event is None:

        print()
        print("STATUS: WAITING_FOR_VERIFIED_ARCHIVE")
        print("Extraction performed: NO")
        print("Technical eligibility: NOT EVALUATED")

        return None

    rank_dir = (
        STAGING_ROOT
        / rank_dir_name(row)
    )

    require(
        rank_dir.is_dir()
        and not rank_dir.is_symlink(),
        "EXTRACTION_REVIEW_REQUIRED: invalid "
        "candidate staging directory.",
    )

    archive = (
        rank_dir
        / Path(event["archive_path"]).name
    )

    require(
        archive.is_file()
        and not archive.is_symlink(),
        "EXTRACTION_REVIEW_REQUIRED: "
        "verified Archive is missing.",
    )

    if archive.suffix.lower() != ".zip":

        print()
        print("STATUS: ARCHIVE_FORMAT_REVIEW_REQUIRED")
        print("RAR/7z extraction: NOT IMPLEMENTED")
        print("Technical exclusion: NONE")

        return None

    output = (
        rank_dir / EXTRACT_DIR_NAME
    )

    if output.exists() or output.is_symlink():

        print()
        print("STATUS: EXTRACTION_ALREADY_EXISTS")
        print("Inspect the existing extraction before retrying.")

        return None

    if (
        rank_dir / TEMP_DIR_NAME
    ).exists():

        print()
        print("STATUS: EXTRACTION_REVIEW_REQUIRED")
        print("Previous temporary extraction exists.")

        return None

    return row, event, archive, output


def main():

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--status",
        action="store_true",
        help="Inspect the current frozen Rank.",
    )

    mode.add_argument(
        "--extract",
        action="store_true",
        help="Explicitly authorize verified ZIP extraction.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    require(
        not MANIFEST.exists(),
        "Final Confirmation Manifest already exists.",
    )

    selected = inspect_current_rank(
        args.rank
    )

    if selected is None:
        return

    row, event, archive, output = selected

    if not args.extract:

        print()
        print("STATUS: VERIFIED_ARCHIVE_READY_FOR_EXTRACTION")
        print("Extraction performed: NO")

        return

    extract_zip(
        archive,
        output,
        candidate_rank=row["candidate_rank"],
        source_match_id=row["source_match_id"],
        expected_archive_sha256=event["archive_sha256"],
    )


if __name__ == "__main__":

    try:
        main()

    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:

        print(
            "\nZIP EXTRACTION STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
