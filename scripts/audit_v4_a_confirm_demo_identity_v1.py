#!/usr/bin/env python3

"""
V4-A Confirmation: read-only Demo identity audit.

Verifies historical SHA256 references, extraction records,
extracted file identities and actual Demo header map names.

Does not assign eligibility, select a Demo, measure the raw
tick clock, modify the frozen queue or perform model scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from pathlib import Path

from build_v4_a_confirm_queue_v1 import load_frozen_rules

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    STAGING_ROOT,
    load_initial_candidates,
    rank_dir_name,
    read_event,
    require,
    sha256_file,
)

from extract_v4_a_confirm_zip_v1 import (
    EXTRACT_DIR_NAME,
    RECORD_NAME,
)

from inspect_v4_a_confirm_archive_v1 import (
    safe_member_name,
)


HISTORICAL_GROUPS = (
    "development",
    "v2_confirmation",
    "v3_confirmation",
)


def load_historical_hashes():
    protocol, _ = load_frozen_rules()

    specifications = protocol["provenance"][
        "historical_manifests"
    ]

    historical = {}
    row_count = 0

    for group in HISTORICAL_GROUPS:
        specification = specifications[group]

        manifest = ROOT / specification["path"]

        require(
            manifest.is_file() and not manifest.is_symlink(),
            f"Missing historical manifest: {group}",
        )

        require(
            sha256_file(manifest) == specification["sha256"],
            f"Historical manifest SHA256 mismatch: {group}",
        )

        with manifest.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))

        require(
            len(rows) == specification["rows"],
            f"Historical manifest row count mismatch: {group}",
        )

        for row in rows:
            digest = row.get("sha256", "").lower()

            require(
                len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                f"Invalid historical Demo SHA256: {group}",
            )

            historical.setdefault(digest, set()).add(group)

        row_count += len(rows)

    require(
        row_count == protocol["provenance"][
            "historical_demo_sha256_count"
        ],
        "Historical corpus size differs from frozen protocol.",
    )

    return {
        digest: sorted(groups)
        for digest, groups in historical.items()
    }


def verify_extracted_files(
    extraction_dir,
    event,
    *,
    expected_rank,
    expected_match_id,
):
    extraction_dir = Path(extraction_dir)

    require(
        extraction_dir.is_dir()
        and not extraction_dir.is_symlink(),
        "Invalid extraction directory.",
    )

    record_path = extraction_dir / RECORD_NAME

    require(
        record_path.is_file()
        and not record_path.is_symlink(),
        "Extraction Record is missing or invalid.",
    )

    record = json.loads(
        record_path.read_text(encoding="utf-8")
    )

    require(
        record.get("version")
        == "V4_A_CONFIRM_ZIP_EXTRACTION_V1",
        "Unexpected Extraction Record version.",
    )

    require(
        record.get("candidate_rank") == expected_rank
        and record.get("source_match_id") == expected_match_id,
        "Extraction Record candidate identity mismatch.",
    )

    require(
        record.get("source_archive_sha256")
        == event["archive_sha256"]
        and record.get("source_archive_size_bytes")
        == event["archive_size_bytes"],
        "Extraction Record archive identity mismatch.",
    )

    require(
        record.get("archive_crc_verified") is True
        and record.get("technical_eligibility_evaluated") is False
        and record.get("model_scoring_performed") is False,
        "Unexpected Extraction Record state.",
    )

    members = record.get("members")

    require(
        isinstance(members, list)
        and len(members) == record.get("extracted_file_count"),
        "Extraction Record member count mismatch.",
    )

    expected_files = {RECORD_NAME.casefold()}
    verified = []
    total_bytes = 0

    for member in members:
        name = safe_member_name(member["path"])
        key = name.casefold()

        require(
            key not in expected_files,
            "Duplicate or reserved extraction path.",
        )

        expected_files.add(key)

        path = extraction_dir / name

        require(
            path.is_file() and not path.is_symlink(),
            f"Missing or invalid extracted file: {name}",
        )

        parent = path.parent

        while parent != extraction_dir:
            require(
                parent.is_dir() and not parent.is_symlink(),
                "Unsafe extracted parent directory.",
            )
            parent = parent.parent

        size = path.stat().st_size
        digest = sha256_file(path)

        require(
            size == member["size_bytes"]
            and digest == member["sha256"],
            f"Extracted file identity mismatch: {name}",
        )

        total_bytes += size

        verified.append({
            "path": name,
            "absolute_path": path,
            "size_bytes": size,
            "sha256": digest,
            "is_demo_filename": name.lower().endswith(".dem"),
        })

    require(
        total_bytes == record.get("extracted_total_bytes"),
        "Extracted total bytes mismatch.",
    )

    actual_files = set()

    for path in extraction_dir.rglob("*"):
        require(
            not path.is_symlink(),
            "Unexpected symlink in extracted directory.",
        )

        if path.is_dir():
            continue

        require(
            path.is_file(),
            "Unexpected special file in extraction.",
        )

        actual_files.add(
            path.relative_to(extraction_dir).as_posix().casefold()
        )

    require(
        actual_files == expected_files,
        "Missing or unexpected extracted files.",
    )

    return verified


def read_demo_header(path):
    from cs2_tactical_intelligence.v0.timing import (
        open_v0_demo,
    )

    demo = open_v0_demo(path, verbose=False)

    return {
        "map_name": demo.header.get("map_name"),
        "parser_configured_tickrate": float(demo.tickrate),
    }


def audit_rank(rank):
    require(
        rank == 1,
        "Only Rank 1 is authorized at this intake stage.",
    )

    row = load_initial_candidates()[rank - 1]

    print("=== V4 DEMO IDENTITY AUDIT ===")
    print("Candidate Rank:", rank)
    print("Match ID:", row["source_match_id"])

    event = read_event(row)

    if event is None:
        print("STATUS: WAITING_FOR_VERIFIED_ARCHIVE")
        print("Technical eligibility: NOT EVALUATED")
        return

    extraction_dir = (
        STAGING_ROOT
        / rank_dir_name(row)
        / EXTRACT_DIR_NAME
    )

    if not extraction_dir.exists():
        print("STATUS: WAITING_FOR_VERIFIED_EXTRACTION")
        print("Technical eligibility: NOT EVALUATED")
        return

    verified = verify_extracted_files(
        extraction_dir,
        event,
        expected_rank=rank,
        expected_match_id=row["source_match_id"],
    )

    historical = load_historical_hashes()

    demos = [
        item for item in verified
        if item["is_demo_filename"]
    ]

    print("Verified extracted files:", len(verified))
    print("Demo filenames:", len(demos))

    for item in demos:
        print()
        print("Demo path:", item["path"])
        print("Demo SHA256:", item["sha256"])
        print(
            "Historical SHA256 overlap:",
            historical.get(item["sha256"], []),
        )

        try:
            header = read_demo_header(item["absolute_path"])
        except Exception as exc:
            print("Header status: REVIEW_REQUIRED")
            print("Parser error:", repr(exc))
            continue

        print("Actual map:", header["map_name"])
        print(
            "Parser configured tickrate:",
            header["parser_configured_tickrate"],
        )
        print("Independent raw tickrate measured: NO")

    print()
    print("STATUS: DEMO_IDENTITY_INSPECTED")
    print("Mirage selection performed: NO")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    audit_rank(args.rank)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(
            "\nDEMO IDENTITY AUDIT STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
