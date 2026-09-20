#!/usr/bin/env python3

"""
V4-A Confirmation Acquisition Readiness Preflight.

Reads the already frozen acquisition evidence.

Does not:
    change the frozen queue;
    contact HLTV;
    download a demo;
    assign technical eligibility;
    create a confirmation manifest;
    access the frozen models;
    perform model scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys

from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from build_v4_a_confirm_queue_v1 import (
    QUEUE_COLUMNS,
    load_frozen_rules,
    parse_utc,
    read_historical_match_ids,
)


ROOT = Path(__file__).resolve().parents[1]

QUEUE_PATH = ROOT / (
    "docs/v4_a_confirm_acquisition_queue_v1.csv"
)

SNAPSHOT_PATH = ROOT / (
    "docs/v4_a_confirm_hltv_source_snapshot_v1.html"
)

CAPTURE_PATH = ROOT / (
    "docs/v4_a_confirm_hltv_source_capture_v1.json"
)

EVIDENCE_PATH = ROOT / (
    "docs/v4_a_confirm_queue_freeze_evidence_v1.json"
)

MANIFEST_PATH = ROOT / (
    "docs/v4_a_confirm_manifest_v1.csv"
)

MINIMUM_FREE_GIB = 12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def utc_text(value):
    return value.astimezone(
        timezone.utc
    ).isoformat().replace(
        "+00:00",
        "Z",
    )


def scheduled_times_from_snapshot(
    snapshot_path,
    selected_ids,
):
    """
    Extract original scheduled timestamps for selected
    Match IDs. Pinned/regular duplicate appearances must
    agree. Never replace frozen timestamps with live data.
    """

    soup = BeautifulSoup(
        snapshot_path.read_bytes(),
        "html.parser",
    )

    result = {}

    for wrapper in soup.select(
        "div.match-wrapper[data-match-id]"
    ):
        raw_id = wrapper.get(
            "data-match-id",
            "",
        )

        if not re.fullmatch(
            r"[0-9]+",
            raw_id,
        ):
            continue

        match_id = raw_id

        if match_id not in selected_ids:
            continue

        times = wrapper.select(
            "div.match-time[data-unix]"
        )

        if not times:
            continue

        require(
            len(times) == 1,
            f"Match {match_id}: ambiguous timestamps "
            "inside a single source wrapper.",
        )

        raw_unix = str(
            times[0].get("data-unix", "")
        )

        require(
            bool(re.fullmatch(
                r"[0-9]{13}",
                raw_unix,
            )),
            f"Match {match_id}: invalid frozen timestamp.",
        )

        start = datetime.fromtimestamp(
            int(raw_unix) / 1000,
            tz=timezone.utc,
        )

        result.setdefault(
            match_id,
            set(),
        ).add(start)

    require(
        set(result) == selected_ids,
        "Some selected Match IDs have no original "
        "scheduled timestamp in the source snapshot.",
    )

    normalized = {}

    for match_id, timestamps in result.items():
        require(
            len(timestamps) == 1,
            f"Match {match_id}: conflicting original "
            "scheduled timestamps.",
        )

        normalized[match_id] = next(
            iter(timestamps)
        )

    return normalized


def main():

    print()
    print("=== VERIFY FROZEN ACQUISITION RULES ===")

    protocol, amendment = load_frozen_rules()

    require(
        protocol["sample_plan"]["required_eligible_matches"]
        == 20,
        "Unexpected required eligible-match count.",
    )

    require(
        amendment["candidate_selection"]["initial_queue_size"]
        == 25,
        "Unexpected queue-size requirement.",
    )

    print("Protocol and Amendment: VERIFIED")


    print()
    print("=== VERIFY SOURCE AND QUEUE IDENTITIES ===")

    for path in (
        QUEUE_PATH,
        SNAPSHOT_PATH,
        CAPTURE_PATH,
        EVIDENCE_PATH,
    ):
        require(
            path.is_file() and not path.is_symlink(),
            f"Required frozen artifact missing: {path}",
        )

    capture = read_json(CAPTURE_PATH)
    evidence = read_json(EVIDENCE_PATH)

    require(
        sha256(SNAPSHOT_PATH)
        == capture["snapshot_sha256"]
        == evidence["source_snapshot_sha256"],
        "Frozen source snapshot identity mismatch.",
    )

    require(
        sha256(QUEUE_PATH)
        == evidence["queue_sha256"],
        "Frozen Queue SHA256 mismatch.",
    )

    require(
        evidence["status"]
        == "SOURCE_AND_SELECTION_AUDITED",
        "Unexpected Queue Freeze evidence status.",
    )

    require(
        evidence["selected_candidates"] == 25,
        "Freeze evidence does not record 25 candidates.",
    )

    require(
        evidence["raw_demo_downloaded"] is False
        and evidence["model_scoring_performed"] is False,
        "Unexpected state in original freeze evidence.",
    )

    print("Source Snapshot: VERIFIED")
    print("Frozen Queue: VERIFIED")
    print("Freeze Evidence: VERIFIED")


    print()
    print("=== VERIFY CANDIDATE IDENTITIES ===")

    with QUEUE_PATH.open(
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        require(
            reader.fieldnames == QUEUE_COLUMNS,
            "Frozen Queue schema mismatch.",
        )

        queue = list(reader)

    require(
        len(queue) == 25,
        "Expected exactly 25 queue records.",
    )

    ranks = [
        int(row["candidate_rank"])
        for row in queue
    ]

    require(
        ranks == list(range(1, 26)),
        "Candidate ranks must be exactly 1 through 25.",
    )

    selected_ids = {
        row["source_match_id"]
        for row in queue
    }

    require(
        len(selected_ids) == 25,
        "Frozen Queue contains duplicate Match IDs.",
    )

    historical_ids = read_historical_match_ids(
        [
            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch1"],

            ROOT / amendment[
                "historical_discovery_provenance"
            ]["batch2"],
        ]
    )

    require(
        not selected_ids.intersection({
            str(match_id)
            for match_id in historical_ids
        }),
        "Frozen Queue overlaps provisional Discovery IDs.",
    )

    scheduled_times = scheduled_times_from_snapshot(
        SNAPSHOT_PATH,
        selected_ids,
    )

    capture_time = parse_utc(
        capture["capture_completed_utc"],
        "capture_completed_utc",
    )

    earliest_allowed = parse_utc(
        amendment[
            "new_prospective_sampling_window"
        ]["earliest_scheduled_start_utc"],
        "earliest_scheduled_start_utc",
    )

    for row in queue:

        match_id = row["source_match_id"]

        start = scheduled_times[match_id]

        require(
            start >= earliest_allowed
            and start > capture_time,
            f"Match {match_id}: frozen temporal "
            "sampling rule mismatch.",
        )

        require(
            row["match_date"]
            == start.date().isoformat(),
            f"Match {match_id}: Queue date differs "
            "from the frozen UTC start date.",
        )

    expected_order = sorted(
        queue,
        key=lambda row: (
            row["match_date"],
            int(row["source_match_id"]),
        ),
    )

    require(
        [
            row["source_match_id"]
            for row in queue
        ]
        == [
            row["source_match_id"]
            for row in expected_order
        ],
        "Frozen Queue order does not match protocol.",
    )

    print("Candidate ranks: 1 through 25")
    print("Unique selected Match IDs: 25")
    print("Historical provisional overlap: NONE")
    print("Original UTC schedule identities: VERIFIED")


    print()
    print("=== VERIFY HISTORICAL CORPUS IDENTITIES ===")

    historical_manifests = protocol[
        "provenance"
    ]["historical_manifests"]

    for name, metadata in historical_manifests.items():

        path = ROOT / metadata["path"]

        require(
            path.is_file(),
            f"Historical manifest missing: {path}",
        )

        require(
            sha256(path) == metadata["sha256"],
            f"Historical manifest SHA256 mismatch: {name}",
        )

        with path.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            count = sum(
                1
                for _ in csv.DictReader(handle)
            )

        require(
            count == metadata["rows"],
            f"Historical manifest row-count mismatch: {name}",
        )

        print(
            f"{name}: {count} rows; SHA256 VERIFIED"
        )


    print()
    print("=== VERIFY EXECUTION BOUNDARIES ===")

    require(
        not MANIFEST_PATH.exists(),
        "V4 Confirmation Manifest already exists. "
        "Do not restart acquisition blindly.",
    )

    free_bytes = shutil.disk_usage(
        ROOT
    ).free

    free_gib = free_bytes / (1024 ** 3)

    print(
        f"Available disk: {free_gib:.2f} GiB"
    )

    require(
        free_gib >= MINIMUM_FREE_GIB,
        "Available disk is below the frozen "
        "12 GiB acquisition threshold.",
    )

    print("V4 Manifest: NOT CREATED")
    print("Storage policy: PASS")


    print()
    print("=== FROZEN ACQUISITION ORDER ===")

    now = datetime.now(timezone.utc)

    print("Current UTC:", utc_text(now))
    print()

    for row in queue:

        rank = int(
            row["candidate_rank"]
        )

        match_id = row[
            "source_match_id"
        ]

        start = scheduled_times[match_id]

        schedule_status = (
            "START_TIME_PASSED"
            if now >= start
            else "SCHEDULED"
        )

        print(
            f"{rank:02d}"
            f" | Match {match_id}"
            f" | {utc_text(start)}"
            f" | {schedule_status}"
        )

    print()
    print("First candidate_rank:", queue[0]["source_match_id"])
    print(
        "Required technically eligible matches:",
        protocol["sample_plan"]["required_eligible_matches"],
    )

    print()
    print("IMPORTANT:")
    print(
        "START_TIME_PASSED does not establish that "
        "the match has ended or a demo is available."
    )

    print(
        "No candidate was excluded or assigned "
        "technical eligibility by this preflight."
    )

    print()
    print("================================================")
    print(" V4 8B-5A ACQUISITION PREFLIGHT: PASS")
    print("================================================")

    print("Frozen Queue: UNCHANGED")
    print("HLTV requests: NONE")
    print("Demo downloads: NONE")
    print("Technical exclusions: NONE")
    print("Model predictions: NONE")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(
            "\nPREFLIGHT STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
