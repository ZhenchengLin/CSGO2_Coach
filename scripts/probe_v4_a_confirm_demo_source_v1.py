#!/usr/bin/env python3

"""
V4-A Confirmation: Rank-Ordered Demo Source Probe.

This script:
    - verifies the frozen queue identity;
    - reads a candidate's original scheduled start;
    - optionally inspects its HLTV match page;
    - reports whether a Demo download link is visible.

It never:
    - changes candidate_rank;
    - downloads a Demo or archive;
    - selects a map;
    - assigns technical eligibility or exclusion;
    - writes a Confirmation Manifest;
    - runs a model or computes confirmation metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from build_v4_a_confirm_queue_v1 import (
    QUEUE_COLUMNS,
    load_frozen_rules,
    validate_source_url,
)

from preflight_v4_a_confirm_acquisition_v1 import (
    scheduled_times_from_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]

QUEUE = ROOT / "docs/v4_a_confirm_acquisition_queue_v1.csv"

SNAPSHOT = ROOT / (
    "docs/v4_a_confirm_hltv_source_snapshot_v1.html"
)

CAPTURE = ROOT / (
    "docs/v4_a_confirm_hltv_source_capture_v1.json"
)

EVIDENCE = ROOT / (
    "docs/v4_a_confirm_queue_freeze_evidence_v1.json"
)

MANIFEST = ROOT / (
    "docs/v4_a_confirm_manifest_v1.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def iso_utc(value):
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_frozen_queue():
    load_frozen_rules()

    capture = json.loads(
        CAPTURE.read_text(encoding="utf-8")
    )

    evidence = json.loads(
        EVIDENCE.read_text(encoding="utf-8")
    )

    require(
        sha256(SNAPSHOT.read_bytes())
        == capture["snapshot_sha256"]
        == evidence["source_snapshot_sha256"],
        "Frozen source snapshot SHA256 mismatch.",
    )

    require(
        sha256(QUEUE.read_bytes())
        == evidence["queue_sha256"],
        "Frozen candidate queue SHA256 mismatch.",
    )

    with QUEUE.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        require(
            reader.fieldnames == QUEUE_COLUMNS,
            "Frozen queue schema mismatch.",
        )

        rows = list(reader)

    require(
        len(rows) == 25,
        "Expected exactly 25 frozen candidates.",
    )

    require(
        [int(row["candidate_rank"]) for row in rows]
        == list(range(1, 26)),
        "Frozen candidate_rank sequence mismatch.",
    )

    require(
        len({
            row["source_match_id"]
            for row in rows
        }) == 25,
        "Duplicate frozen Match ID.",
    )

    starts = scheduled_times_from_snapshot(
        SNAPSHOT,
        {
            row["source_match_id"]
            for row in rows
        },
    )

    return rows, starts


def find_demo_links(html, match_url):
    """
    Find source-visible Demo download links.

    Multiple distinct links are not automatically resolved:
    their meaning requires inspection before acquisition.
    """

    soup = BeautifulSoup(html, "html.parser")

    result = set()

    for anchor in soup.find_all("a", href=True):

        href = anchor["href"]

        absolute = urljoin(
            match_url,
            href,
        )

        parsed = urlparse(absolute)

        if (
            parsed.scheme != "https"
            or parsed.hostname not in {
                "hltv.org",
                "www.hltv.org",
            }
        ):
            continue

        if not re.fullmatch(
            r"/download/demo/[^/?#]+/?",
            parsed.path,
        ):
            continue

        result.add(
            "https://www.hltv.org"
            + parsed.path
        )

    return sorted(result)



MAX_MATCH_PAGE_REDIRECTS = 5


def validate_match_page_hop_url(url, match_id):
    """
    Validate a match-page destination before contacting it.

    A changed URL slug is acceptable, but the original
    Match ID must remain unchanged.
    """

    require(
        isinstance(url, str)
        and "\\\\" not in url,
        "SOURCE_REVIEW_REQUIRED: invalid match-page URL.",
    )

    parsed = urlparse(url)

    require(
        parsed.scheme == "https"
        and parsed.netloc.lower()
        in {"hltv.org", "www.hltv.org"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and parsed.path.startswith(
            f"/matches/{match_id}/"
        ),
        "SOURCE_REVIEW_REQUIRED: unapproved match-page "
        "destination or different Match ID.",
    )

    return url


def open_checked_match_page(session, match_url, match_id):
    """
    Follow only same-Match-ID, same-site HTTPS redirects.

    Returns (response, final_url). The caller must
    close the returned response.
    """

    current_url = validate_match_page_hop_url(
        match_url, match_id
    )

    visited = set()

    for _ in range(MAX_MATCH_PAGE_REDIRECTS + 1):

        require(
            current_url not in visited,
            "SOURCE_REVIEW_REQUIRED: match-page "
            "redirect loop.",
        )

        visited.add(current_url)

        response = session.get(
            current_url,
            timeout=60,
            allow_redirects=False,
        )

        if 300 <= response.status_code < 400:

            location = response.headers.get("Location")
            response.close()

            require(
                isinstance(location, str)
                and bool(location.strip()),
                "SOURCE_REVIEW_REQUIRED: match-page "
                "redirect has no Location.",
            )

            next_url = urljoin(current_url, location)

            current_url = validate_match_page_hop_url(
                next_url, match_id
            )

            continue

        return response, current_url

    raise RuntimeError(
        "SOURCE_REVIEW_REQUIRED: too many match-page redirects."
    )


def inspect_source(row, original_start):

    rank = int(row["candidate_rank"])

    match_id = row["source_match_id"]

    match_url = row["source_url"]

    validate_source_url(
        match_url,
        int(match_id),
    )

    now = datetime.now(timezone.utc)

    print()
    print("=== SELECTED FROZEN CANDIDATE ===")
    print("Candidate Rank:", rank)
    print("Match ID:", match_id)
    print("Original scheduled UTC:", iso_utc(original_start))
    print("Inspection UTC:", iso_utc(now))
    print("Frozen match URL:", match_url)

    if now < original_start:

        print()
        print("SOURCE STATUS: SCHEDULED")
        print("HLTV request: NOT PERFORMED")
        print("Demo availability: NOT YET ASSESSED")

        return

    # A passed scheduled start does not prove that the
    # match has finished. We only inspect source metadata.
    from curl_cffi import requests

    session = requests.Session(
        impersonate="chrome"
    )

    try:
        response, final_url = open_checked_match_page(
            session, match_url, match_id
        )

        try:
            require(
                response.status_code == 200,
                "HLTV match-page request did not return "
                "HTTP 200. Do not classify this as "
                "a technical exclusion.",
            )

            html = response.content

        finally:
            response.close()

    finally:
        session.close()

    page_text = html.decode(
        "utf-8",
        errors="replace",
    )

    require(
        "Access denied" not in page_text[:5000],
        "HLTV returned an access-denied page. "
        "Do not classify the match.",
    )

    require(
        match_id in final_url,
        "Match-page identity could not be established.",
    )

    links = find_demo_links(
        page_text,
        final_url,
    )

    print()
    print("=== SOURCE INSPECTION ===")
    print("HTTP status:", response.status_code)
    print("Final URL:", final_url)
    print("Match-page SHA256:", sha256(html))
    print("Visible Demo links:", len(links))

    if not links:

        print()
        print("SOURCE STATUS: AWAITING_DEMO_OR_MATCH_RESULT")
        print(
            "No Demo link is visible in this inspection."
        )
        print(
            "This is NOT a DOWNLOAD_FAILURE "
            "or a technical exclusion."
        )

        return

    if len(links) > 1:

        print()
        print("SOURCE STATUS: REVIEW_REQUIRED")
        print("Multiple distinct Demo links found:")

        for link in links:
            print(" ", link)

        print(
            "No link was selected automatically."
        )

        return

    print()
    print("SOURCE STATUS: DEMO_LINK_VISIBLE")
    print("Source Demo URL:", links[0])
    print(
        "A visible link does not establish the "
        "Demo's map, tick rate or technical eligibility."
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Frozen candidate_rank to inspect (default: 1).",
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show all frozen candidates without HTTP requests.",
    )

    args = parser.parse_args()

    rows, starts = read_frozen_queue()

    require(
        not MANIFEST.exists(),
        "Final V4 Confirmation Manifest already exists. "
        "Do not restart intake blindly.",
    )

    now = datetime.now(timezone.utc)

    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)

    print("Frozen Queue: VERIFIED")
    print(f"Available disk: {free_gib:.2f} GiB")

    require(
        free_gib >= 12,
        "Available disk is below the frozen "
        "12 GiB threshold.",
    )

    if args.plan:

        print()
        print("=== ORIGINAL FROZEN SCHEDULE ===")

        for row in rows:

            rank = int(row["candidate_rank"])
            match_id = row["source_match_id"]
            start = starts[match_id]

            status = (
                "SCHEDULED"
                if now < start
                else "START_TIME_PASSED"
            )

            print(
                f"{rank:02d}"
                f" | {match_id}"
                f" | {iso_utc(start)}"
                f" | {status}"
            )

        print()
        print("HTTP requests: NONE")
        print("Demo downloads: NONE")
        print("Technical decisions: NONE")

        return

    require(
        1 <= args.rank <= 25,
        "candidate_rank must be between 1 and 25.",
    )

    row = rows[args.rank - 1]

    require(
        int(row["candidate_rank"]) == args.rank,
        "Frozen Queue rank mismatch.",
    )

    inspect_source(
        row,
        starts[row["source_match_id"]],
    )

    print()
    print("Frozen Queue: UNCHANGED")
    print("Demo downloads: NONE")
    print("Technical exclusions: NONE")
    print("Model scoring: NONE")


if __name__ == "__main__":
    main()
