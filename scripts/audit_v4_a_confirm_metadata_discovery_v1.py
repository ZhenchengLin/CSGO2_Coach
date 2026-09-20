#!/usr/bin/env python3

"""
CS2 Tactical Intelligence

V4 — Implementation 8B-2F

Confirmation Metadata Discovery Audit
=====================================

Purpose
-------
Audit the existing 25 discovered match metadata records.

Preserve the distinction between:

1. Structural metadata validation.
2. Independent source-page verification.
3. Frozen candidate-queue readiness.

This script does not automatically freeze a candidate queue.

Safety
------
- Do not modify existing discovery batches.
- Do not download raw demos.
- Do not access confirmation targets or features.
- Do not load or score models.
- Do not modify frozen contracts.
- Do not create the final confirmation manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys

from datetime import date
from pathlib import Path
from urllib.parse import urlparse


# ============================================================
# 1. Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DOCS = ROOT / "docs"

PROTOCOL = (
    DOCS
    / "v4_a_confirm_acquisition_protocol_v1_frozen.json"
)

BATCH1 = (
    DOCS
    / "v4_a_confirm_metadata_staging_snapshot_v1.csv"
)

BATCH2 = (
    DOCS
    / "v4_a_confirm_metadata_discovery_batch2_v1.csv"
)

SOURCE_REVIEW = (
    DOCS
    / "v4_a_confirm_metadata_source_review_v1.json"
)

LEDGER = (
    DOCS
    / "v4_a_confirm_metadata_verification_ledger_v1.csv"
)

REPORT = (
    DOCS
    / "v4_a_confirm_metadata_freeze_readiness_v1.json"
)

FINAL_QUEUE = (
    DOCS
    / "v4_a_confirm_acquisition_queue_v1.csv"
)


# ============================================================
# 2. Frozen source identities
# ============================================================

EXPECTED_SHA256 = {
    "protocol": (
        "5bd7e756ac78589e8b60d23521e1d133"
        "a722e08d2cd975e7129ffe03d130c3e4"
    ),
    "batch1": (
        "efb93bcc0dbdf5359225e81cb55af77"
        "db15139e2804d11ed071168b5b21ee32d"
    ),
    "batch2": (
        "83efe9c2581badfca3f61e649ab60fe"
        "2ceac3036046255ff012720249cdc4b49"
    ),
}


# ============================================================
# 3. Frozen metadata schema
# ============================================================

COLUMNS = [
    "match_date",
    "source_match_id",
    "event",
    "team1",
    "team2",
    "source",
    "source_url",
]

LEDGER_COLUMNS = [
    "provisional_sort_index",
    *COLUMNS,
    "structural_validation",
    "source_page_verification",
    "date_verification",
    "review_flags",
]


# ============================================================
# 4. General utilities
# ============================================================

def require(condition, message):
    """Stop execution when a required invariant fails."""

    if not condition:
        raise AssertionError(message)


def sha256(path):
    """Calculate the SHA256 identity of a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:

        while True:

            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def verify_identity(path, expected):
    """Verify an existing file against its recorded identity."""

    require(
        path.is_file() and not path.is_symlink(),
        f"Required source file missing or invalid: {path}",
    )

    actual = sha256(path)

    require(
        actual == expected,
        (
            f"Source SHA256 mismatch: {path}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        ),
    )

    print("SHA256 PASS:", path.name)


# ============================================================
# 5. Load frozen acquisition protocol
# ============================================================

def load_protocol():

    verify_identity(
        PROTOCOL,
        EXPECTED_SHA256["protocol"],
    )

    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    require(
        protocol["status"]
        == "FROZEN_BEFORE_V4_CONFIRMATION_ACQUISITION",
        "Unexpected acquisition protocol status.",
    )

    require(
        protocol["research_role"]["dataset"]
        == "D_V4_A_CONFIRM",
        "Unexpected dataset identity.",
    )

    require(
        protocol["sample_plan"]["initial_candidate_queue"]
        == 25,
        "Frozen candidate count mismatch.",
    )

    require(
        protocol["sample_plan"]["required_eligible_matches"]
        == 20,
        "Frozen eligible-match count mismatch.",
    )

    require(
        protocol["candidate_queue"]["required_fields"]
        == ["candidate_rank", *COLUMNS],
        "Frozen candidate metadata schema mismatch.",
    )

    return protocol


# ============================================================
# 6. Read existing discovery batches
# ============================================================

def read_batch(path, expected_sha256):

    verify_identity(
        path,
        expected_sha256,
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        require(
            reader.fieldnames == COLUMNS,
            f"Metadata schema mismatch: {path}",
        )

        rows = list(reader)

    for index, row in enumerate(rows, start=1):

        require(
            None not in row,
            (
                f"Unexpected columns: "
                f"{path.name}, row {index}"
            ),
        )

        for column in COLUMNS:

            value = row.get(column)

            require(
                value is not None and value.strip(),
                (
                    f"Missing {column}: "
                    f"{path.name}, row {index}"
                ),
            )

            row[column] = value.strip()

    return rows


# ============================================================
# 7. Validate one metadata record
# ============================================================

def validate_metadata(row, earliest_date):

    match_id = row["source_match_id"]

    require(
        match_id.isascii()
        and match_id.isdigit()
        and int(match_id) > 0,
        f"Invalid Match ID: {match_id}",
    )

    recorded_date = date.fromisoformat(
        row["match_date"]
    )

    require(
        recorded_date >= earliest_date,
        (
            f"Staged date violates temporal boundary: "
            f"{match_id}"
        ),
    )

    require(
        row["source"] == "HLTV",
        f"Unexpected source: {match_id}",
    )

    parsed = urlparse(
        row["source_url"]
    )

    require(
        parsed.scheme == "https",
        f"Invalid URL scheme: {match_id}",
    )

    require(
        parsed.hostname in {
            "hltv.org",
            "www.hltv.org",
        },
        f"Unexpected source domain: {match_id}",
    )

    require(
        not parsed.username
        and not parsed.password,
        f"Unexpected URL authentication: {match_id}",
    )

    match = re.fullmatch(
        r"/matches/([0-9]+)/[^/?#]+/?",
        parsed.path,
    )

    require(
        match is not None,
        f"Invalid source URL structure: {match_id}",
    )

    require(
        int(match.group(1)) == int(match_id),
        f"Source Match ID / URL mismatch: {match_id}",
    )

    return True


# ============================================================
# 8. Identify records requiring additional review
# ============================================================

def review_flags(match_id):

    flags = []

    if match_id == "2398267":

        flags.append(
            "SOURCE_PAGE_DATE_CONFLICT"
        )

    if match_id == "2398107":

        flags.append(
            "VERIFY_MATCH_PARTICIPANTS"
        )

    if match_id == "2398269":

        flags.append(
            "VERIFY_MATCH_PARTICIPANTS"
        )

    if match_id == "2398312":

        flags.append(
            "FORFEIT_TECHNICAL_ELIGIBILITY_PENDING"
        )

    return flags


# ============================================================
# 9. Render CSV without modifying source data
# ============================================================

def render_ledger(rows):

    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=LEDGER_COLUMNS,
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(rows)

    return buffer.getvalue().encode("utf-8")


# ============================================================
# 10. Safely preserve generated outputs
# ============================================================

def preserve_output(path, content):

    if path.exists() or path.is_symlink():

        require(
            path.is_file() and not path.is_symlink(),
            f"Invalid existing output: {path}",
        )

        require(
            path.read_bytes() == content,
            (
                f"Existing output differs: {path}\n"
                "Refusing to overwrite."
            ),
        )

        print("EXISTING OUTPUT VERIFIED:", path.name)

        return

    with path.open("xb") as handle:

        handle.write(content)

    require(
        path.read_bytes() == content,
        f"Output verification failed: {path}",
    )

    print("CREATED:", path.name)


# ============================================================
# 11. Main
# ============================================================

def main():

    print()
    print("=" * 64)
    print(" V4 IMPLEMENTATION 8B-2F")
    print(" CONFIRMATION METADATA DISCOVERY AUDIT")
    print("=" * 64)

    require(
        not FINAL_QUEUE.exists(),
        "Final candidate queue already exists.",
    )

    print()
    print("=== FROZEN PROTOCOL ===")

    protocol = load_protocol()

    earliest_date = date.fromisoformat(
        protocol["temporal_boundary"][
            "earliest_allowed_match_date"
        ]
    )

    print("Acquisition protocol: VERIFIED")
    print("Earliest match date:", earliest_date)

    print()
    print("=== DISCOVERY BATCHES ===")

    first = read_batch(
        BATCH1,
        EXPECTED_SHA256["batch1"],
    )

    second = read_batch(
        BATCH2,
        EXPECTED_SHA256["batch2"],
    )

    require(
        len(first) == 11,
        "First discovery batch count mismatch.",
    )

    require(
        len(second) == 14,
        "Second discovery batch count mismatch.",
    )

    rows = first + second

    require(
        len(rows) == 25,
        "Expected exactly 25 discovery records.",
    )

    print("First batch:", len(first))
    print("Second batch:", len(second))

    print()
    print("=== METADATA IDENTITY VALIDATION ===")

    seen_ids = set()

    for row in rows:

        validate_metadata(
            row,
            earliest_date,
        )

        match_id = row["source_match_id"]

        require(
            match_id not in seen_ids,
            f"Duplicate Match ID: {match_id}",
        )

        seen_ids.add(match_id)

    print("Structural validation: PASS")
    print("Match ID uniqueness: PASS")
    print("Staged temporal boundary: PASS")

    print()
    print("=== EXISTING SOURCE REVIEW ===")

    require(
        SOURCE_REVIEW.is_file(),
        "Existing source-review record is missing.",
    )

    previous_review = json.loads(
        SOURCE_REVIEW.read_text(encoding="utf-8")
    )

    require(
        previous_review["metadata_snapshot"]["sha256"]
        == EXPECTED_SHA256["batch1"],
        "Previous source review references another snapshot.",
    )

    print("Previous source-review identity: PASS")

    print()
    print("=== PROVISIONAL ORDERING ===")

    ordered = sorted(
        rows,
        key=lambda row: (
            row["match_date"],
            int(row["source_match_id"]),
        ),
    )

    ledger_rows = []

    flagged_records = []

    for index, row in enumerate(ordered, start=1):

        match_id = row["source_match_id"]

        flags = review_flags(match_id)

        if flags:

            flagged_records.append({
                "source_match_id": match_id,
                "flags": flags,
            })

        ledger_rows.append({
            "provisional_sort_index": index,
            **row,
            "structural_validation": "PASS",
            "source_page_verification": (
                "REQUIRES_INDEPENDENT_VERIFICATION"
            ),
            "date_verification": (
                "REQUIRES_CONSISTENT_DATE_CONVENTION"
            ),
            "review_flags": "|".join(flags),
        })

        print(
            f"[{index:02d}/25]",
            row["match_date"],
            "|",
            match_id,
            "|",
            row["team1"],
            "vs",
            row["team2"],
            "|",
            ",".join(flags) if flags else "STRUCTURAL_PASS",
        )

    print()
    print(
        "NOTE: Provisional sort index is not a "
        "frozen candidate rank."
    )

    print()
    print("=== CREATE VERIFICATION LEDGER ===")

    ledger_bytes = render_ledger(
        ledger_rows
    )

    preserve_output(
        LEDGER,
        ledger_bytes,
    )

    print("Ledger SHA256:", sha256(LEDGER))

    print()
    print("=== CREATE FREEZE READINESS REPORT ===")

    report = {
        "version": (
            "V4_A_CONFIRM_METADATA_FREEZE_READINESS_V1"
        ),

        "status": "NOT_READY_FOR_QUEUE_FREEZE",

        "dataset": "D_V4_A_CONFIRM",

        "repository_head_at_development_verification": (
            "1da00d2"
        ),

        "source_artifacts": {
            "batch1": {
                "path": str(BATCH1.relative_to(ROOT)),
                "sha256": sha256(BATCH1),
                "records": len(first),
            },
            "batch2": {
                "path": str(BATCH2.relative_to(ROOT)),
                "sha256": sha256(BATCH2),
                "records": len(second),
            },
            "verification_ledger": {
                "path": str(LEDGER.relative_to(ROOT)),
                "sha256": sha256(LEDGER),
                "records": len(ledger_rows),
            },
        },

        "validation": {
            "discovery_records": len(rows),
            "distinct_match_ids": len(seen_ids),
            "structural_validation": "PASS",
            "source_page_verification": "INCOMPLETE",
            "date_convention": "UNRESOLVED",
            "sampling_frame": "NOT_DOCUMENTED_AS_FIXED",
            "technical_eligibility": "NOT_PERFORMED",
        },

        "flagged_records": flagged_records,

        "blocking_requirements": [
            (
                "Establish and document a consistent "
                "source match-date convention."
            ),
            (
                "Independently verify source metadata "
                "for the proposed candidate corpus."
            ),
            (
                "Document the candidate sampling frame "
                "and the method used to select the "
                "25 discovered matches."
            ),
            (
                "Resolve any source metadata conflicts "
                "without making performance-based "
                "or map-based selection decisions."
            ),
            (
                "Complete an independent queue review "
                "before final queue creation and commit."
            ),
        ],

        "experimental_boundaries": {
            "frozen_queue_created": False,
            "confirmation_demo_downloaded": False,
            "confirmation_model_scored": False,
            "frozen_contract_modified": False,
        },
    }

    report_bytes = (
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    preserve_output(
        REPORT,
        report_bytes,
    )

    print("Report SHA256:", sha256(REPORT))

    print()
    print("=== FINAL SUMMARY ===")

    print("Discovery records: 25")
    print("Unique Match IDs: 25")

    print()
    print("STRUCTURAL VALIDATION: PASS")
    print("SOURCE VERIFICATION: INCOMPLETE")
    print("DATE CONVENTION: UNRESOLVED")
    print("SAMPLING FRAME: NOT DOCUMENTED AS FIXED")

    print()
    print("FINAL CANDIDATE QUEUE: NOT CREATED")
    print("RAW DEMO ACQUISITION: NOT PERFORMED")
    print("MODEL SCORING: NOT PERFORMED")

    print()
    print("V4_METADATA_FREEZE_READINESS_AUDIT_COMPLETE")


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"\nSTOP: {exc}",
            file=sys.stderr,
        )

        raise
