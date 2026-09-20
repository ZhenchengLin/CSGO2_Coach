#!/usr/bin/env python3

"""
V4-A Confirmation Archive Safety Inspection.

Read-only checks:
    Frozen Rank and Acquisition Event identity.
    Downloaded Archive SHA256 via the State Manager.
    ZIP member paths, types and declared sizes.
    Available disk budget for future extraction.

This module does NOT extract files, identify maps,
assign technical eligibility, or score models.

RAR and 7z remain REVIEW_REQUIRED until dedicated
inspectors are implemented.
"""

from __future__ import annotations

import argparse
import shutil
import stat
import sys
import zipfile

from pathlib import Path, PurePosixPath

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    STAGING_ROOT,
    load_initial_candidates,
    rank_dir_name,
    read_event,
    require,
)

from download_v4_a_confirm_archive_v1 import (
    GIB,
    MIB,
    MINIMUM_FREE_BYTES,
    SAFETY_RESERVE_BYTES,
)


MAX_MEMBERS = 4096


def safe_member_name(name):
    require(
        isinstance(name, str)
        and bool(name)
        and len(name) <= 4096,
        "ARCHIVE_REVIEW_REQUIRED: invalid member name.",
    )

    require(
        not name.startswith("/")
        and "\\" not in name
        and "\x00" not in name
        and ":" not in name,
        "ARCHIVE_REVIEW_REQUIRED: unsafe member path.",
    )

    parts = name.rstrip("/").split("/")

    require(
        all(part not in {"", ".", ".."} for part in parts),
        "ARCHIVE_REVIEW_REQUIRED: unsafe member path segment.",
    )

    normalized = PurePosixPath(name.rstrip("/"))

    require(
        not normalized.is_absolute(),
        "ARCHIVE_REVIEW_REQUIRED: absolute member path.",
    )

    return normalized.as_posix()


def inspect_zip(archive_path, budget_bytes):
    """
    Inspect ZIP metadata without extracting its contents.

    Declared sizes are only an initial safety check.
    Actual extraction must enforce independent runtime
    limits and verify file content in a later stage.
    """

    require(
        archive_path.is_file()
        and not archive_path.is_symlink(),
        "ARCHIVE_REVIEW_REQUIRED: missing or invalid ZIP.",
    )

    require(
        archive_path.suffix.lower() == ".zip",
        "ARCHIVE_REVIEW_REQUIRED: this inspector supports ZIP only.",
    )

    require(
        zipfile.is_zipfile(archive_path),
        "ARCHIVE_REVIEW_REQUIRED: invalid ZIP structure.",
    )

    require(
        budget_bytes > 0,
        "STORAGE_BLOCK: no space is available for extraction.",
    )

    members = []
    seen_names = set()
    total_uncompressed = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:

            infos = archive.infolist()

            require(
                0 < len(infos) <= MAX_MEMBERS,
                "ARCHIVE_REVIEW_REQUIRED: unexpected ZIP member count.",
            )

            for info in infos:

                normalized = safe_member_name(info.filename)

                key = normalized.casefold()

                require(
                    key not in seen_names,
                    "ARCHIVE_REVIEW_REQUIRED: duplicate member path.",
                )

                seen_names.add(key)

                file_type = stat.S_IFMT(
                    info.external_attr >> 16
                )

                if info.is_dir():
                    require(
                        file_type in {0, stat.S_IFDIR},
                        "ARCHIVE_REVIEW_REQUIRED: invalid directory type.",
                    )
                    continue

                require(
                    file_type in {0, stat.S_IFREG},
                    "ARCHIVE_REVIEW_REQUIRED: links or special files "
                    "are not permitted.",
                )

                require(
                    not (info.flag_bits & 0x1),
                    "ARCHIVE_REVIEW_REQUIRED: encrypted ZIP member.",
                )

                require(
                    info.compress_type in {
                        zipfile.ZIP_STORED,
                        zipfile.ZIP_DEFLATED,
                    },
                    "ARCHIVE_REVIEW_REQUIRED: unsupported compression.",
                )

                require(
                    info.file_size >= 0,
                    "ARCHIVE_REVIEW_REQUIRED: invalid declared size.",
                )

                total_uncompressed += info.file_size

                require(
                    total_uncompressed <= budget_bytes,
                    "STORAGE_BLOCK: declared uncompressed size "
                    "exceeds the available extraction budget.",
                )

                members.append({
                    "path": normalized,
                    "declared_uncompressed_bytes": info.file_size,
                    "declared_compressed_bytes": info.compress_size,
                    "is_demo_filename": (
                        normalized.lower().endswith(".dem")
                    ),
                })

    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "ARCHIVE_REVIEW_REQUIRED: ZIP metadata is corrupt."
        ) from exc

    return {
        "archive_format": "ZIP",
        "member_count": len(members),
        "declared_uncompressed_bytes": total_uncompressed,
        "demo_filename_count": sum(
            member["is_demo_filename"]
            for member in members
        ),
        "members": members,
        "content_verified": False,
        "extraction_performed": False,
        "technical_eligibility_evaluated": False,
    }


def inspect_rank(rank):

    require(
        rank == 1,
        "RANK_ORDER_BLOCK: only Rank 1 is authorized "
        "for the current intake workflow.",
    )

    rows = load_initial_candidates()

    row = rows[rank - 1]

    event = read_event(row)

    print()
    print("=== V4 ARCHIVE INSPECTION ===")
    print("Candidate Rank:", rank)
    print("Match ID:", row["source_match_id"])

    if event is None:

        print("STATUS: WAITING_FOR_VERIFIED_ARCHIVE")
        print("No Archive was inspected.")
        print("Technical eligibility: NOT EVALUATED")
        return

    archive_path = (
        STAGING_ROOT
        / rank_dir_name(row)
        / Path(event["archive_path"]).name
    )

    require(
        archive_path.is_file()
        and not archive_path.is_symlink(),
        "ARCHIVE_REVIEW_REQUIRED: verified Archive is missing.",
    )

    print("Verified Archive SHA256:", event["archive_sha256"])
    print("Archive:", archive_path)

    if archive_path.suffix.lower() != ".zip":

        print("STATUS: ARCHIVE_FORMAT_REVIEW_REQUIRED")
        print(
            "RAR/7z support must be implemented before "
            "attempting extraction."
        )
        print("Technical exclusion: NONE")
        return

    free = shutil.disk_usage(ROOT).free

    budget = (
        free
        - MINIMUM_FREE_BYTES
        - SAFETY_RESERVE_BYTES
    )

    print("Free disk:", f"{free / GIB:.2f} GiB")
    print(
        "Current extraction budget:",
        f"{max(budget, 0) / MIB:.1f} MiB",
    )

    result = inspect_zip(
        archive_path,
        budget_bytes=budget,
    )

    print()
    print("STATUS: ZIP_METADATA_INSPECTED")
    print("ZIP file members:", result["member_count"])

    print(
        "Declared uncompressed size:",
        result["declared_uncompressed_bytes"],
        "bytes",
    )

    print(
        "Demo filenames found:",
        result["demo_filename_count"],
    )

    for member in result["members"]:
        print(
            " ",
            member["path"],
            "|",
            member["declared_uncompressed_bytes"],
            "bytes",
        )

    print()
    print("Archive content/CRC verified: NO")
    print("Extraction performed: NO")
    print("Map identified: NO")
    print("Technical eligibility: NOT EVALUATED")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    inspect_rank(args.rank)


if __name__ == "__main__":

    try:
        main()

    except (
        RuntimeError,
        ValueError,
        OSError,
        zipfile.BadZipFile,
    ) as exc:

        print(
            "\nARCHIVE INSPECTION STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
