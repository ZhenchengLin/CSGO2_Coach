#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo
echo "================================================"
echo " V4 IMPLEMENTATION 8B-4E"
echo " OFFLINE METADATA EXTRACTION AND QUEUE PREVIEW"
echo "================================================"

uv run --with beautifulsoup4 python - <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup


ROOT = Path.cwd()

sys.path.insert(0, str(ROOT / "scripts"))

from build_v4_a_confirm_queue_v1 import (
    build_candidate_queue,
    load_frozen_rules,
    read_historical_match_ids,
    render_queue_csv,
)


SNAPSHOT = ROOT / (
    "docs/v4_a_confirm_hltv_source_snapshot_v1.html"
)

CAPTURE = ROOT / (
    "docs/v4_a_confirm_hltv_source_capture_v1.json"
)

PREVIEW = Path(
    "/tmp/csgo2_v4_8b4e_queue_preview.csv"
)

MATCH_PATH = re.compile(
    r"^/matches/([0-9]+)/[^/?#]+/?$"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_utc(value):
    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    require(
        parsed.tzinfo is not None,
        "Timestamp must include timezone.",
    )

    return parsed.astimezone(timezone.utc)


def utc_text(value):
    return value.isoformat().replace("+00:00", "Z")


def match_id_from_href(href):
    parsed = urlparse(href)

    match = MATCH_PATH.fullmatch(
        parsed.path
    )

    if match is None:
        return None

    return int(match.group(1))


def match_url(href, match_id):
    parsed = urlparse(href)

    require(
        parsed.scheme in ("", "https"),
        f"Unexpected URL scheme for {match_id}.",
    )

    require(
        parsed.hostname in (
            None,
            "hltv.org",
            "www.hltv.org",
        ),
        f"Unexpected URL host for {match_id}.",
    )

    require(
        match_id_from_href(href) == match_id,
        f"Source URL mismatch for {match_id}.",
    )

    return (
        "https://www.hltv.org"
        + parsed.path
    )


def direct_match_links(wrapper, match_id):
    links = []

    for anchor in wrapper.find_all(
        "a",
        href=True,
    ):
        # Ignore links belonging to nested match wrappers.
        if (
            anchor.find_parent(
                "div",
                class_="match-wrapper",
            )
            is not wrapper
        ):
            continue

        if match_id_from_href(
            anchor["href"]
        ) == match_id:
            links.append(anchor)

    return links


print()
print("=== VERIFY FROZEN RULES ===")

protocol, amendment = load_frozen_rules()

print("Original Acquisition Protocol: VERIFIED")
print("Prospective Amendment V1: VERIFIED")


print()
print("=== VERIFY SAVED SOURCE SNAPSHOT ===")

raw = SNAPSHOT.read_bytes()

capture = json.loads(
    CAPTURE.read_text(encoding="utf-8")
)

actual_sha = hashlib.sha256(raw).hexdigest()

require(
    actual_sha == capture["snapshot_sha256"],
    "Saved HLTV snapshot SHA256 mismatch.",
)

require(
    capture["amendment_commit"] == "511f689",
    "Unexpected Amendment commit identity.",
)

require(
    capture["source_url"]
    == amendment["metadata_source"]["primary_listing"],
    "Unexpected metadata source.",
)

capture_started = parse_utc(
    capture["capture_started_utc"]
)

capture_completed = parse_utc(
    capture["capture_completed_utc"]
)

require(
    capture_started <= capture_completed,
    "Capture timestamp ordering mismatch.",
)

print("Snapshot SHA256:", actual_sha)

print(
    "Capture completed UTC:",
    utc_text(capture_completed),
)


print()
print("=== LOAD HISTORICAL MATCH IDS ===")

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

print("Historical provisional IDs:", len(historical))


print()
print("=== EXTRACT TIMED MATCH WRAPPERS ===")

soup = BeautifulSoup(
    raw,
    "html.parser",
)

time_elements = soup.select(
    "div.match-time[data-unix]"
)

# Every appearance is retained during extraction.
# Duplicates are merged by their HLTV Match ID later.
appearances = []

for element in time_elements:

    wrapper = element.find_parent(
        "div",
        class_="match-wrapper",
    )

    require(
        wrapper is not None,
        "Timed match element has no match-wrapper.",
    )

    raw_id = wrapper.get(
        "data-match-id",
        "",
    )

    require(
        bool(re.fullmatch(r"[0-9]+", raw_id)),
        f"Invalid match-wrapper ID: {raw_id!r}",
    )

    match_id = int(raw_id)

    links = direct_match_links(
        wrapper,
        match_id,
    )

    require(
        links,
        f"Match {match_id} has no matching detail link.",
    )

    urls = {
        match_url(
            anchor["href"],
            match_id,
        )
        for anchor in links
    }

    require(
        len(urls) == 1,
        f"Match {match_id} has conflicting detail URLs.",
    )

    unix_ms = element.get(
        "data-unix",
        "",
    )

    require(
        bool(re.fullmatch(
            r"[0-9]{13}",
            unix_ms,
        )),
        f"Match {match_id}: invalid Unix timestamp.",
    )

    start = datetime.fromtimestamp(
        int(unix_ms) / 1000,
        tz=timezone.utc,
    )

    event_element = wrapper.select_one(
        "div.match-event[data-event-headline]"
    )

    event = ""

    if event_element is not None:
        event = (
            event_element.get(
                "data-event-headline",
                "",
            )
            .strip()
        )

    team1_element = wrapper.select_one(
        "div.match-team.team1 "
        "div.match-teamname"
    )

    team2_element = wrapper.select_one(
        "div.match-team.team2 "
        "div.match-teamname"
    )

    team1 = (
        team1_element.get_text(
            " ",
            strip=True,
        )
        if team1_element is not None
        else ""
    )

    team2 = (
        team2_element.get_text(
            " ",
            strip=True,
        )
        if team2_element is not None
        else ""
    )

    unknown_element = wrapper.select_one(
        "a.match-no-info"
    )

    placeholder = (
        unknown_element.get_text(
            " ",
            strip=True,
        )
        if unknown_element is not None
        else ""
    )

    appearances.append(
        {
            "match_id": match_id,
            "start": start,
            "url": next(iter(urls)),
            "event": event,
            "team1": team1,
            "team2": team2,
            "placeholder": placeholder,
            "pinned": (
                wrapper.get("data-pinned")
                == "true"
            ),
        }
    )

print(
    "Timed match appearances:",
    len(appearances),
)


print()
print("=== MERGE APPEARANCES BY MATCH ID ===")

groups = {}

for appearance in appearances:
    groups.setdefault(
        appearance["match_id"],
        [],
    ).append(appearance)

records = []

unresolved_ids = []

for match_id, entries in groups.items():

    timestamps = {
        entry["start"]
        for entry in entries
    }

    require(
        len(timestamps) == 1,
        f"Conflicting start times for {match_id}.",
    )

    urls = {
        entry["url"]
        for entry in entries
    }

    require(
        len(urls) == 1,
        f"Conflicting URLs for {match_id}.",
    )

    events = {
        entry["event"]
        for entry in entries
        if entry["event"]
    }

    require(
        len(events) <= 1,
        f"Conflicting event names for {match_id}.",
    )

    event = next(iter(events), "")

    team_pairs = {
        (entry["team1"], entry["team2"])
        for entry in entries
        if entry["team1"] and entry["team2"]
    }

    require(
        len(team_pairs) <= 1,
        f"Conflicting team names for {match_id}.",
    )

    placeholders = {
        entry["placeholder"]
        for entry in entries
        if entry["placeholder"]
    }

    require(
        len(placeholders) <= 1,
        f"Conflicting participant placeholders for {match_id}.",
    )

    if team_pairs:

        team1, team2 = next(iter(team_pairs))
        participants_resolved = True

    elif placeholders:

        # Preserve the source's exact match-stage
        # description. Neither team is identified.
        placeholder = next(iter(placeholders))

        team1 = placeholder
        team2 = placeholder

        participants_resolved = False
        unresolved_ids.append(match_id)

    else:

        # The original source provides no participant
        # names or placeholder for this appearance.
        # Keep it in the complete selection population.
        # If selected, the Queue Builder will reject it.
        team1 = ""
        team2 = ""

        participants_resolved = False
        unresolved_ids.append(match_id)

    records.append(
        {
            "source_match_id": str(match_id),
            "scheduled_start_utc": utc_text(
                next(iter(timestamps))
            ),
            "event": event,
            "team1": team1,
            "team2": team2,
            "source_url": next(iter(urls)),
            "participants_resolved": participants_resolved,
        }
    )

print("Unique timed Match IDs:", len(records))

print(
    "Matches with unresolved participants:",
    len(unresolved_ids),
)


print()
print("=== PRE-SELECTION METADATA CHECK ===")

earliest = parse_utc(
    amendment[
        "new_prospective_sampling_window"
    ]["earliest_scheduled_start_utc"]
)

pool = [
    row
    for row in records
    if (
        int(row["source_match_id"])
        not in historical
        and parse_utc(
            row["scheduled_start_utc"]
        ) >= earliest
        and parse_utc(
            row["scheduled_start_utc"]
        ) > capture_completed
    )
]

missing_events = [
    row["source_match_id"]
    for row in pool
    if not row["event"]
]

print("Prospective pool:", len(pool))

print(
    "Pool records missing Event:",
    len(missing_events),
)

if missing_events:
    print(
        "Missing Event IDs (first 15):",
        missing_events[:15],
    )

# An Event must not be invented from the match URL.
# If a relevant source record lacks Event metadata,
# stop and inspect the original source evidence.
# An unrelated later match may have no Event metadata.
# The Builder must reject missing Event if that match
# belongs to the deterministically selected first 25.

print()
print("=== CALL EXISTING 8B-4A QUEUE BUILDER ===")

# Pass the full extracted listing, not a manually
# selected or pre-trimmed group of 25 matches.
queue = build_candidate_queue(
    snapshot_rows=records,
    captured_at_utc=(
        capture["capture_completed_utc"]
    ),
    historical_match_ids=historical,
    earliest_start_utc=(
        amendment[
            "new_prospective_sampling_window"
        ]["earliest_scheduled_start_utc"]
    ),
)

require(
    len(queue) == 25,
    "Queue Builder returned an unexpected count.",
)

print("Queue Builder output:", len(queue))


print()
print("=== FIRST 25 CANDIDATES ===")

for row in queue:

    match_id = row["source_match_id"]

    source_record = next(
        record
        for record in records
        if record["source_match_id"] == match_id
    )

    status = (
        "KNOWN"
        if source_record["participants_resolved"]
        else "UNRESOLVED"
    )

    print(
        f'{row["candidate_rank"]:02d}',
        row["match_date"],
        match_id,
        "|",
        row["event"],
        "|",
        row["team1"],
        "vs",
        row["team2"],
        "|",
        status,
    )


print()
print("=== WRITE PREVIEW OUTSIDE REPOSITORY ===")

preview_bytes = render_queue_csv(queue)

if PREVIEW.exists():

    require(
        PREVIEW.is_file()
        and not PREVIEW.is_symlink(),
        "Existing preview path is not a regular file.",
    )

    require(
        PREVIEW.read_bytes() == preview_bytes,
        "An existing preview differs. "
        "Refusing to overwrite it.",
    )

    print("Existing identical Preview: VERIFIED")

else:

    with PREVIEW.open("xb") as handle:
        handle.write(preview_bytes)

    print("Preview created:", PREVIEW)

print(
    "Preview SHA256:",
    hashlib.sha256(
        preview_bytes
    ).hexdigest(),
)


print()
print("================================================")
print(" V4 8B-4E EXTRACTION AND PREVIEW COMPLETE")
print("================================================")

print("Original HLTV HTML: UNCHANGED")
print("Frozen Protocol / Amendment: UNCHANGED")
print("Official Candidate Queue: NOT CREATED")
print("Confirmation Demos: NOT DOWNLOADED")
print("Model Scoring: NOT PERFORMED")
print()
print(
    "NEXT GATE: Source listing completeness "
    "and frozen Queue evidence review."
)

PY

echo
echo "=== FINAL GIT STATUS ==="

git status --short --untracked-files=all
