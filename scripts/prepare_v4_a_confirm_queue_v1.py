#!/usr/bin/env python3

"""
CS2 Tactical Intelligence

V4 — Implementation 8B-2A

Confirmation Candidate Metadata Collection
==========================================

Purpose
-------
Prepare a deterministic 25-candidate metadata queue
according to the frozen V4-A acquisition protocol.

This implementation supports HLTV match metadata.

The metadata is supplied through an independently
maintained staging CSV.

This script does not automatically invent, acquire,
or substitute candidate matches.

Safety
------
- No raw demo downloading.
- No confirmation feature extraction.
- No model loading.
- No model scoring.
- No frozen contract modification.
- No confirmation manifest creation.
- No Git commit or push.

A successfully generated preview is NOT a frozen queue.

Source URLs and match metadata require independent
verification before the queue can be frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re

from datetime import date
from pathlib import Path
from urllib.parse import urlparse


# ============================================================
# 1. Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL = (
    ROOT
    / "docs"
    / "v4_a_confirm_acquisition_protocol_v1_frozen.json"
)

STAGING = Path(
    "/tmp/csgo2_v4_candidate_metadata_staging.csv"
)

PREVIEW = Path(
    "/tmp/csgo2_v4_candidate_queue_preview.csv"
)


# ============================================================
# 2. Frozen protocol identity
# ============================================================

EXPECTED_PROTOCOL_SHA256 = (
    "5bd7e756ac78589e8b60d23521e1d133"
    "a722e08d2cd975e7129ffe03d130c3e4"
)


# ============================================================
# 3. Metadata schemas
# ============================================================

STAGING_COLUMNS = [
    "match_date",
    "source_match_id",
    "event",
    "team1",
    "team2",
    "source",
    "source_url",
]

QUEUE_COLUMNS = [
    "candidate_rank",
    *STAGING_COLUMNS,
]


# ============================================================
# 4. General utilities
# ============================================================

def require(condition, message):
    """Raise an explicit validation failure."""

    if not condition:
        raise AssertionError(message)


def sha256(path):
    """Calculate SHA256 without changing the file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:

        while True:

            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 5. Load frozen acquisition protocol
# ============================================================

def load_protocol():
    """
    Load the existing frozen acquisition protocol.

    Never modify or regenerate the frozen protocol.
    """

    require(
        PROTOCOL.is_file(),
        f"Frozen protocol missing: {PROTOCOL}",
    )

    require(
        sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256,
        "Frozen acquisition protocol SHA256 mismatch.",
    )

    contract = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    require(
        contract["status"]
        == "FROZEN_BEFORE_V4_CONFIRMATION_ACQUISITION",
        "Unexpected acquisition protocol status.",
    )

    require(
        contract["research_role"]["dataset"]
        == "D_V4_A_CONFIRM",
        "Unexpected confirmation dataset identity.",
    )

    require(
        contract["sample_plan"]["initial_candidate_queue"]
        == 25,
        "Frozen candidate queue size changed.",
    )

    require(
        contract["sample_plan"]["required_eligible_matches"]
        == 20,
        "Frozen eligible-match requirement changed.",
    )

    require(
        contract["sample_plan"]["selection_rule"]
        == (
            "Select the first 20 technically eligible "
            "candidates in frozen candidate_rank order."
        ),
        "Frozen candidate selection rule changed.",
    )

    require(
        contract["candidate_queue"]["required_fields"]
        == QUEUE_COLUMNS,
        "Frozen candidate metadata schema changed.",
    )

    return contract


# ============================================================
# 6. Create metadata staging file
# ============================================================

def create_staging_file(path):
    """
    Create a staging CSV with the required metadata header.

    Existing staging files are never overwritten.
    """

    if path.exists():

        print("Existing metadata staging file:", path)

        return

    require(
        not path.is_symlink(),
        "Refusing to write through a staging symlink.",
    )

    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(STAGING_COLUMNS)

    print("Created metadata staging file:", path)


# ============================================================
# 7. Read staged metadata
# ============================================================

def read_staging(path):
    """
    Read metadata records without changing their contents.

    Candidate rank is assigned only when creating
    the deterministic queue preview.
    """

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        require(
            reader.fieldnames == STAGING_COLUMNS,
            "Staging CSV schema mismatch.",
        )

        rows = list(reader)

    for index, row in enumerate(rows, start=1):

        require(
            None not in row,
            f"Unexpected extra columns at row {index}.",
        )

        for column in STAGING_COLUMNS:

            require(
                row[column] is not None,
                f"Missing {column} at row {index}.",
            )

            row[column] = row[column].strip()

            require(
                bool(row[column]),
                f"Empty {column} at row {index}.",
            )

    return rows


# ============================================================
# 8. Validate candidate metadata
# ============================================================

def validate_candidate(row, index, earliest_date):
    """
    Validate the structural integrity of one HLTV candidate.

    This does not independently verify that the supplied
    teams, event, match date, or match result are truthful.

    Source-page verification remains a separate requirement.
    """

    match_date = date.fromisoformat(
        row["match_date"]
    )

    require(
        match_date >= earliest_date,
        (
            f"Candidate {index}: match date violates "
            "the frozen temporal boundary."
        ),
    )

    source = row["source"]

    require(
        source.casefold() == "hltv",
        (
            f"Candidate {index}: this metadata collector "
            "currently supports HLTV records only."
        ),
    )

    match_id = row["source_match_id"]

    require(
        match_id.isascii() and match_id.isdigit(),
        (
            f"Candidate {index}: expected a numeric "
            "HLTV Match ID."
        ),
    )

    match_id_number = int(match_id)

    require(
        match_id_number > 0,
        f"Candidate {index}: invalid Match ID.",
    )

    parsed = urlparse(
        row["source_url"]
    )

    require(
        parsed.scheme == "https",
        (
            f"Candidate {index}: expected an HTTPS "
            "source URL."
        ),
    )

    require(
        parsed.hostname in {
            "hltv.org",
            "www.hltv.org",
        },
        (
            f"Candidate {index}: source URL is not "
            "an HLTV page."
        ),
    )

    match = re.fullmatch(
        r"/matches/([0-9]+)/[^/?#]+/?",
        parsed.path,
    )

    require(
        match is not None,
        (
            f"Candidate {index}: expected a direct "
            "HLTV match-page URL."
        ),
    )

    require(
        int(match.group(1)) == match_id_number,
        (
            f"Candidate {index}: Source Match ID "
            "does not match the URL."
        ),
    )

    require(
        not parsed.username
        and not parsed.password,
        (
            f"Candidate {index}: unexpected URL "
            "authentication information."
        ),
    )

    return {
        **row,
        "source_match_id": str(match_id_number),
    }


# ============================================================
# 9. Validate complete staging corpus
# ============================================================

def validate_staging(rows, contract):
    """
    Validate temporal boundaries and source identities.

    No performance-based selection occurs here.
    """

    earliest_date = date.fromisoformat(
        contract["temporal_boundary"][
            "earliest_allowed_match_date"
        ]
    )

    validated = []

    seen_ids = set()

    seen_urls = set()

    for index, row in enumerate(rows, start=1):

        item = validate_candidate(
            row,
            index,
            earliest_date,
        )

        match_id = item["source_match_id"]

        url = item["source_url"].rstrip("/")

        require(
            match_id not in seen_ids,
            (
                "Duplicate HLTV Match ID "
                f"in staging: {match_id}"
            ),
        )

        require(
            url not in seen_urls,
            (
                "Duplicate HLTV source URL "
                f"in staging: {url}"
            ),
        )

        seen_ids.add(match_id)

        seen_urls.add(url)

        validated.append(item)

    return validated


# ============================================================
# 10. Deterministic candidate ordering
# ============================================================

def build_queue(rows):
    """
    Apply the frozen ordering:

        match_date ascending
        source_match_id ascending

    For HLTV, source_match_id is a numeric identifier.
    """

    ordered = sorted(
        rows,
        key=lambda row: (
            date.fromisoformat(row["match_date"]),
            int(row["source_match_id"]),
        ),
    )

    queue = []

    for rank, row in enumerate(ordered, start=1):

        queue.append({
            "candidate_rank": rank,
            **row,
        })

    return queue


# ============================================================
# 11. Render deterministic CSV
# ============================================================

def render_queue(queue):
    """
    Construct canonical UTF-8 CSV bytes.

    Output is used for preview only.
    """

    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=QUEUE_COLUMNS,
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(queue)

    return buffer.getvalue().encode("utf-8")


# ============================================================
# 12. Create queue preview
# ============================================================

def write_preview(queue):
    """
    Write a preview outside the repository.

    Never overwrite a differing existing preview.
    """

    data = render_queue(queue)

    if PREVIEW.exists():

        require(
            PREVIEW.is_file()
            and not PREVIEW.is_symlink(),
            "Existing preview is not a regular file.",
        )

        require(
            PREVIEW.read_bytes() == data,
            (
                "Existing queue preview differs. "
                "Review the previous preview before "
                "creating a new version."
            ),
        )

        print("Existing preview verified:", PREVIEW)

    else:

        with PREVIEW.open("xb") as handle:

            handle.write(data)

        print("Created queue preview:", PREVIEW)

    print(
        "Preview SHA256:",
        sha256(PREVIEW),
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen V4-A confirmation "
            "candidate metadata queue."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=STAGING,
        help="Candidate metadata staging CSV.",
    )

    args = parser.parse_args()

    print()
    print("=" * 64)
    print(" V4 IMPLEMENTATION 8B-2A")
    print(" CONFIRMATION CANDIDATE METADATA COLLECTION")
    print("=" * 64)

    contract = load_protocol()

    print()
    print("Frozen acquisition protocol: VERIFIED")

    expected_count = int(
        contract["sample_plan"]["initial_candidate_queue"]
    )

    print(
        "Required candidate count:",
        expected_count,
    )

    print(
        "Earliest match date:",
        contract["temporal_boundary"][
            "earliest_allowed_match_date"
        ],
    )

    print()
    print("=== METADATA STAGING ===")

    create_staging_file(args.input)

    rows = read_staging(args.input)

    validated = validate_staging(
        rows,
        contract,
    )

    print()
    print("Staged candidate count:", len(validated))

    print()
    print("=== CANDIDATE METADATA ===")

    for row in sorted(
        validated,
        key=lambda item: (
            item["match_date"],
            int(item["source_match_id"]),
        ),
    ):

        print(
            row["match_date"],
            "|",
            row["source_match_id"],
            "|",
            row["team1"],
            "vs",
            row["team2"],
        )

    print()
    print("=== QUEUE COMPLETENESS ===")

    require(
        len(validated) <= expected_count,
        (
            "Staging exceeds the frozen 25-candidate "
            "queue size. Do not perform discretionary "
            "selection from an oversized staging corpus."
        ),
    )

    if len(validated) < expected_count:

        missing = expected_count - len(validated)

        print(
            "QUEUE STATUS: INCOMPLETE"
        )

        print(
            "Additional metadata records required:",
            missing,
        )

        print()
        print(
            "No queue preview was generated."
        )

        print(
            "No frozen confirmation queue was created."
        )

        return

    queue = build_queue(validated)

    require(
        len(queue) == expected_count,
        "Candidate queue size mismatch.",
    )

    require(
        [
            row["candidate_rank"]
            for row in queue
        ] == list(range(1, 26)),
        "Candidate rank invariant failed.",
    )

    print(
        "Candidate count: 25/25"
    )

    print(
        "Deterministic ordering: PASS"
    )

    print(
        "Source Match ID uniqueness: PASS"
    )

    print(
        "Temporal boundary: PASS"
    )

    print()
    print("=== GENERATE QUEUE PREVIEW ===")

    write_preview(queue)

    print()
    print("QUEUE PREVIEW: STRUCTURAL VALIDATION PASS")

    print()
    print(
        "IMPORTANT: Source-page verification "
        "is still required."
    )

    print(
        "The preview is not the frozen "
        "Confirmation Acquisition Queue."
    )


if __name__ == "__main__":

    main()
