#!/usr/bin/env python3

"""
V4-A Confirmation — Deterministic Queue Selection Engine.

This module selects candidate matches from normalized metadata.

It does NOT:
    - acquire or verify an HLTV source snapshot;
    - establish source-listing completeness;
    - download or inspect demos;
    - assess technical eligibility;
    - write the official confirmation queue;
    - load or score frozen models.

A queue returned by this module is not a frozen queue.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT
    / "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json"
)

AMENDMENT_PATH = (
    ROOT
    / "docs/v4_a_confirm_acquisition_amendment_v1.json"
)

EXPECTED_PROTOCOL_SHA256 = (
    "5bd7e756ac78589e8b60d23521e1d133"
    "a722e08d2cd975e7129ffe03d130c3e4"
)

EXPECTED_AMENDMENT_SHA256 = (
    "c98cb8c3b271275aa6b688b63a36f2f0d"
    "a9a64e2feec7a7b638061d0ec520d7d"
)

QUEUE_COLUMNS = [
    "candidate_rank",
    "match_date",
    "source_match_id",
    "event",
    "team1",
    "team2",
    "source",
    "source_url",
]

REQUIRED_INPUT_FIELDS = [
    "source_match_id",
    "scheduled_start_utc",
    "event",
    "team1",
    "team2",
    "source_url",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_frozen_rules():
    """
    Read the original protocol and Amendment V1.

    Their exact bytes must match the committed identities.
    """
    require(
        sha256_file(PROTOCOL_PATH)
        == EXPECTED_PROTOCOL_SHA256,
        "Frozen acquisition protocol SHA256 mismatch.",
    )

    require(
        sha256_file(AMENDMENT_PATH)
        == EXPECTED_AMENDMENT_SHA256,
        "Acquisition Amendment V1 SHA256 mismatch.",
    )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    amendment = json.loads(
        AMENDMENT_PATH.read_text(encoding="utf-8")
    )

    require(
        protocol["candidate_queue"]["required_fields"]
        == QUEUE_COLUMNS,
        "Frozen queue schema mismatch.",
    )

    require(
        protocol["sample_plan"]["initial_candidate_queue"]
        == amendment["candidate_selection"]["initial_queue_size"]
        == 25,
        "Frozen queue-size mismatch.",
    )

    require(
        amendment["candidate_selection"]["pool_order"]
        == [
            "scheduled_start_utc ascending",
            "numeric source_match_id ascending",
        ],
        "Unexpected candidate-pool ordering.",
    )

    require(
        amendment["candidate_selection"]["final_queue_order"]
        == [
            "match_date ascending",
            "numeric source_match_id ascending",
        ],
        "Unexpected final-queue ordering.",
    )

    return protocol, amendment


def parse_utc(value, field_name):
    """
    Convert an explicitly timezone-aware ISO timestamp to UTC.

    Reject naive timestamps such as:
        2026-09-21T09:00:00

    Accept:
        2026-09-21T09:00:00Z
        2026-09-21T12:00:00+03:00
    """
    require(
        isinstance(value, str) and value.strip(),
        f"{field_name}: missing timestamp.",
    )

    try:
        parsed = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"{field_name}: invalid ISO timestamp."
        ) from exc

    require(
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None,
        f"{field_name}: timezone is required.",
    )

    return parsed.astimezone(timezone.utc)


def read_historical_match_ids(paths):
    """
    Read the 25 provisional Discovery Match IDs.

    These IDs are excluded by Amendment V1.
    """
    result = []

    for path in paths:
        with Path(path).open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            reader = csv.DictReader(handle)

            require(
                "source_match_id"
                in (reader.fieldnames or []),
                f"Historical CSV missing source_match_id: {path}",
            )

            for row in reader:
                result.append(
                    normalize_match_id(
                        row["source_match_id"]
                    )
                )

    require(
        len(result) == 25 and len(set(result)) == 25,
        "Historical provisional corpus must contain 25 unique IDs.",
    )

    return set(result)


def normalize_match_id(value):
    text = str(value).strip()

    require(
        bool(re.fullmatch(r"[0-9]+", text)),
        f"Invalid source_match_id: {value!r}",
    )

    number = int(text)

    require(
        number > 0,
        "source_match_id must be positive.",
    )

    return number


def validate_source_url(url, match_id):
    require(
        isinstance(url, str),
        "Missing source_url.",
    )

    parsed = urlparse(url)

    require(
        parsed.scheme == "https"
        and parsed.hostname in {
            "hltv.org",
            "www.hltv.org",
        }
        and parsed.username is None
        and parsed.password is None,
        f"Invalid HLTV URL for match {match_id}.",
    )

    match = re.fullmatch(
        r"/matches/([0-9]+)/[^/?#]+/?",
        parsed.path,
    )

    require(
        match is not None
        and int(match.group(1)) == match_id,
        f"URL/Match ID mismatch: {match_id}.",
    )


def build_candidate_queue(
    snapshot_rows,
    captured_at_utc,
    historical_match_ids,
    earliest_start_utc,
    queue_size=25,
):
    """
    Pure selection function.

    Inputs:
        snapshot_rows:
            All normalized match records from a source snapshot.

        captured_at_utc:
            Actual source-capture timestamp.

        historical_match_ids:
            Predeclared provisional Discovery IDs.

        earliest_start_utc:
            Prospective sampling-window lower bound.

    The caller must separately establish that the original
    source snapshot is complete and independently auditable.

    This function never creates or freezes an official queue.
    """
    capture_time = parse_utc(
        captured_at_utc,
        "captured_at_utc",
    )

    earliest_time = parse_utc(
        earliest_start_utc,
        "earliest_start_utc",
    )

    require(
        isinstance(queue_size, int)
        and not isinstance(queue_size, bool)
        and queue_size == 25,
        "Amendment V1 requires exactly 25 candidates.",
    )

    require(
        isinstance(snapshot_rows, (list, tuple)),
        "snapshot_rows must be a sequence.",
    )

    historical = {
        normalize_match_id(value)
        for value in historical_match_ids
    }

    require(
        len(historical) == 25,
        "Expected exactly 25 historical provisional Match IDs.",
    )

    all_rows = []
    seen_ids = set()

    for index, row in enumerate(snapshot_rows, start=1):
        require(
            isinstance(row, dict),
            f"Snapshot row {index} is not a dictionary.",
        )

        for field in REQUIRED_INPUT_FIELDS:
            require(
                field in row
                and isinstance(row[field], str)
                and bool(row[field].strip()),
                f"Snapshot row {index}: missing {field}.",
            )

        match_id = normalize_match_id(
            row["source_match_id"]
        )

        require(
            match_id not in seen_ids,
            f"Duplicate snapshot Match ID: {match_id}.",
        )

        seen_ids.add(match_id)

        start_time = parse_utc(
            row["scheduled_start_utc"],
            f"Match {match_id} scheduled_start_utc",
        )

        validate_source_url(
            row["source_url"],
            match_id,
        )

        all_rows.append(
            {
                "match_id": match_id,
                "start_time": start_time,
                "event": row["event"].strip(),
                "team1": row["team1"].strip(),
                "team2": row["team2"].strip(),
                "source_url": row["source_url"].strip(),
            }
        )

    # Selection is independent of map, demo availability,
    # target distribution, match outcomes and model scores.
    candidate_pool = [
        row
        for row in all_rows
        if row["match_id"] not in historical
        and row["start_time"] >= earliest_time
        and row["start_time"] > capture_time
    ]

    require(
        len(candidate_pool) >= queue_size,
        "Insufficient prospective candidate pool: "
        f"{len(candidate_pool)}/{queue_size}.",
    )

    # Amendment V1: determine WHICH 25 matches are selected.
    selected = sorted(
        candidate_pool,
        key=lambda row: (
            row["start_time"],
            row["match_id"],
        ),
    )[:queue_size]

    # Parent protocol: determine the frozen candidate RANK.
    selected.sort(
        key=lambda row: (
            row["start_time"].date(),
            row["match_id"],
        ),
    )

    queue = []

    for rank, row in enumerate(selected, start=1):
        queue.append(
            {
                "candidate_rank": rank,
                "match_date": (
                    row["start_time"].date().isoformat()
                ),
                "source_match_id": str(row["match_id"]),
                "event": row["event"],
                "team1": row["team1"],
                "team2": row["team2"],
                "source": "HLTV",
                "source_url": row["source_url"],
            }
        )

    require(
        len(queue) == 25
        and len({
            row["source_match_id"]
            for row in queue
        }) == 25,
        "Final selection invariant failed.",
    )

    return queue


def render_queue_csv(queue):
    """Produce deterministic CSV bytes without writing a file."""
    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=QUEUE_COLUMNS,
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()
    writer.writerows(queue)

    return output.getvalue().encode("utf-8")


if __name__ == "__main__":
    protocol, amendment = load_frozen_rules()

    historical = read_historical_match_ids(
        [
            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch1"],
            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch2"],
        ]
    )

    print("Frozen acquisition protocol: VERIFIED")
    print("Acquisition Amendment V1: VERIFIED")
    print("Historical provisional Match IDs:", len(historical))
    print("Deterministic Queue Builder: READY")
    print()
    print("No source snapshot was acquired.")
    print("No official Candidate Queue was created.")
    print("No Demo was downloaded.")
