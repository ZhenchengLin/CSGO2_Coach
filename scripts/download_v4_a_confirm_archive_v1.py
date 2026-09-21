#!/usr/bin/env python3

"""
V4-A Confirmation: Single-Candidate Archive Downloader.

Default mode:
    Offline readiness inspection.

Explicit --download mode:
    Acquire one source-linked archive for Rank 1 only.

This script NEVER:
    modifies the frozen Queue;
    downloads multiple candidates;
    selects a map;
    extracts a Demo;
    assigns technical eligibility;
    creates the final Confirmation Manifest;
    loads frozen models;
    performs model scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from probe_v4_a_confirm_demo_source_v1 import (
    ROOT,
    MANIFEST,
    find_demo_links,
    iso_utc,
    open_checked_match_page,
    read_frozen_queue,
)

from build_v4_a_confirm_queue_v1 import (
    validate_source_url,
)


STAGING_ROOT = (
    ROOT / "data/raw/v4_a_confirm_download_staging"
)

GIB = 1024 ** 3

MIB = 1024 ** 2

MINIMUM_FREE_BYTES = 12 * GIB

MAX_ARCHIVE_BYTES = 1 * GIB

SAFETY_RESERVE_BYTES = 64 * MIB

CHUNK_SIZE = 1 * MIB


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def free_bytes():
    return shutil.disk_usage(ROOT).free


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def archive_extension(header):
    """
    Identify an archive from its actual file signature.

    Do not trust Content-Type or URL filename alone.
    """

    if header.startswith(b"PK\x03\x04"):
        return ".zip"

    if header.startswith(b"Rar!\x1a\x07"):
        return ".rar"

    if header.startswith(b"7z\xbc\xaf\x27\x1c"):
        return ".7z"

    return None


MAX_SAFE_REDIRECTS = 5


def validate_download_hop_url(url):
    """
    Check an HTTP destination BEFORE sending a request.

    External CDN destinations require a separate source
    review; they are not silently approved here.
    """
    require(
        isinstance(url, str)
        and "\\\\" not in url,
        "SOURCE_REDIRECT_REVIEW_REQUIRED: invalid URL.",
    )

    parsed = urlparse(url)

    require(
        parsed.scheme == "https"
        and parsed.netloc.lower() in {
            "hltv.org",
            "www.hltv.org",
        }
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment,
        "SOURCE_REDIRECT_REVIEW_REQUIRED: "
        "unapproved download destination.",
    )

    return url


def open_checked_demo_stream(session, download_url):
    """
    Follow only approved same-site HTTPS redirects.

    Returns (response, final_url). The caller owns the
    returned response and must close it.

    A redirect to an external host is NOT requested.
    """
    current_url = validate_download_hop_url(
        download_url
    )

    visited = set()

    for _ in range(MAX_SAFE_REDIRECTS + 1):

        require(
            current_url not in visited,
            "SOURCE_REDIRECT_REVIEW_REQUIRED: "
            "redirect loop detected.",
        )

        visited.add(current_url)

        response = session.get(
            current_url,
            stream=True,
            allow_redirects=False,
            timeout=300,
        )

        if 300 <= response.status_code < 400:

            location = response.headers.get("Location")
            response.close()

            require(
                isinstance(location, str)
                and bool(location.strip()),
                "SOURCE_REDIRECT_REVIEW_REQUIRED: "
                "redirect has no Location.",
            )

            next_url = urljoin(
                current_url,
                location,
            )

            current_url = validate_download_hop_url(
                next_url
            )

            continue

        return response, current_url

    raise RuntimeError(
        "SOURCE_REDIRECT_REVIEW_REQUIRED: "
        "too many redirects."
    )


def safe_download_budget():

    free = free_bytes()

    require(
        free >= MINIMUM_FREE_BYTES,
        "STORAGE_BLOCK: free disk is below 12 GiB.",
    )

    budget = min(
        MAX_ARCHIVE_BYTES,
        free
        - MINIMUM_FREE_BYTES
        - SAFETY_RESERVE_BYTES,
    )

    require(
        budget > CHUNK_SIZE,
        "STORAGE_BLOCK: insufficient space above "
        "the required 12 GiB reserve.",
    )

    return budget


def verify_selected_candidate(rank):

    rows, scheduled_times = read_frozen_queue()

    require(
        not MANIFEST.exists(),
        "Final Confirmation Manifest already exists. "
        "Do not restart acquisition.",
    )

    require(
        1 <= rank <= 25,
        "candidate_rank must be between 1 and 25.",
    )

    row = rows[rank - 1]

    require(
        int(row["candidate_rank"]) == rank,
        "Frozen candidate_rank mismatch.",
    )

    match_id = row["source_match_id"]

    validate_source_url(
        row["source_url"],
        int(match_id),
    )

    original_start = scheduled_times[match_id]

    require(
        row["match_date"]
        == original_start.date().isoformat(),
        "Frozen match-date identity mismatch.",
    )

    return row, original_start


def inspect_match_page(row):
    """
    Inspect the frozen Match ID's current page.

    No demo acquisition occurs inside this function.
    """

    from curl_cffi import requests

    match_id = row["source_match_id"]

    session = requests.Session(
        impersonate="chrome"
    )

    try:

        response, final_url = open_checked_match_page(
            session,
            row["source_url"],
            match_id,
        )

        require(
            response.status_code == 200,
            "SOURCE_REVIEW_REQUIRED: match page "
            "did not return HTTP 200.",
        )

        parsed = urlparse(final_url)

        require(
            parsed.scheme == "https"
            and parsed.hostname in {
                "hltv.org",
                "www.hltv.org",
            },
            "SOURCE_REVIEW_REQUIRED: unexpected "
            "match-page redirect.",
        )

        require(
            parsed.path.startswith(
                f"/matches/{match_id}/"
            ),
            "SOURCE_REVIEW_REQUIRED: match page "
            "redirected to another Match ID.",
        )

        html = response.content

        require(
            b"Access denied" not in html[:5000],
            "SOURCE_REVIEW_REQUIRED: "
            "HLTV returned an access-denied page.",
        )

        links = find_demo_links(
            html.decode(
                "utf-8",
                errors="replace",
            ),
            final_url,
        )

        return {
            "page_url": final_url,
            "page_sha256": sha256(html),
            "demo_links": links,
        }

    finally:
        session.close()


def download_one_archive(
    *,
    row,
    source_record,
):
    """
    Stream exactly one archive into an exclusive
    V4 staging directory.

    The file remains outside the repository's
    frozen metadata and historical Demo directories.
    """

    from curl_cffi import requests

    rank = int(row["candidate_rank"])

    match_id = row["source_match_id"]

    links = source_record["demo_links"]

    require(
        len(links) == 1,
        "Exactly one verified Demo link is required.",
    )

    download_url = links[0]

    parsed = urlparse(download_url)

    require(
        parsed.scheme == "https"
        and parsed.hostname in {
            "hltv.org",
            "www.hltv.org",
        }
        and parsed.path.startswith("/download/demo/"),
        "Unexpected Demo source URL.",
    )

    budget = safe_download_budget()

    rank_directory = (
        STAGING_ROOT
        / f"rank_{rank:02d}_{match_id}"
    )

    require(
        not rank_directory.exists(),
        "STAGING_ALREADY_EXISTS: this candidate "
        "already has a staging directory. "
        "Inspect it before retrying.",
    )

    print()
    print("=== DOWNLOAD AUTHORIZATION ===")

    print("Candidate Rank:", rank)
    print("Match ID:", match_id)
    print("Demo URL:", download_url)

    print(
        "Available disk:",
        f"{free_bytes() / GIB:.2f} GiB",
    )

    print(
        "Maximum archive budget:",
        f"{budget / MIB:.1f} MiB",
    )

    print(
        "Minimum remaining disk:",
        "12 GiB",
    )

    # Exclusive directory creation prevents accidentally
    # overwriting a previous download or partial result.
    rank_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    part_path = rank_directory / "archive.part"

    final_path = None

    session = requests.Session(
        impersonate="chrome"
    )

    bytes_written = 0

    digest = hashlib.sha256()

    first_bytes = bytearray()

    try:

        response, final_url = open_checked_demo_stream(
            session,
            download_url,
        )

        try:

            require(
                response.status_code == 200,
                "SOURCE_REVIEW_REQUIRED: Demo download "
                "did not return HTTP 200.",
            )

            raw_length = response.headers.get(
                "Content-Length"
            )

            if raw_length is not None:

                try:
                    expected_length = int(raw_length)

                except ValueError as exc:
                    raise RuntimeError(
                        "SOURCE_REVIEW_REQUIRED: invalid "
                        "Content-Length."
                    ) from exc

                require(
                    expected_length > 0,
                    "SOURCE_REVIEW_REQUIRED: "
                    "empty download response.",
                )

                require(
                    expected_length <= budget,
                    "STORAGE_BLOCK: declared archive size "
                    "exceeds the available download budget.",
                )

            with part_path.open("xb") as output:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if not chunk:
                        continue

                    next_total = (
                        bytes_written + len(chunk)
                    )

                    require(
                        next_total <= budget,
                        "STORAGE_BLOCK: archive exceeded "
                        "the authorized download budget.",
                    )

                    # Check BEFORE every write.
                    require(
                        free_bytes()
                        >= (
                            MINIMUM_FREE_BYTES
                            + SAFETY_RESERVE_BYTES
                            + len(chunk)
                        ),
                        "STORAGE_BLOCK: insufficient "
                        "remaining disk space.",
                    )

                    output.write(chunk)

                    digest.update(chunk)

                    bytes_written = next_total

                    if len(first_bytes) < 8:

                        needed = 8 - len(first_bytes)

                        first_bytes.extend(
                            chunk[:needed]
                        )

                output.flush()

                os.fsync(output.fileno())

            require(
                bytes_written > 0,
                "SOURCE_REVIEW_REQUIRED: "
                "empty downloaded archive.",
            )

            if raw_length is not None:

                require(
                    bytes_written == expected_length,
                    "SOURCE_REVIEW_REQUIRED: "
                    "download size differs from "
                    "Content-Length.",
                )

        finally:
            response.close()

        extension = archive_extension(
            bytes(first_bytes)
        )

        require(
            extension is not None,
            "ARCHIVE_REVIEW_REQUIRED: downloaded "
            "content is not recognized as ZIP/RAR/7z. "
            "No technical exclusion has been assigned.",
        )

        require(
            free_bytes() >= MINIMUM_FREE_BYTES,
            "STORAGE_BLOCK: disk fell below 12 GiB. "
            "The partial download must not be retained.",
        )

        final_path = (
            rank_directory
            / f"archive{extension}"
        )

        require(
            not final_path.exists(),
            "Final archive path already exists.",
        )

        part_path.replace(final_path)

        record = {
            "version": (
                "V4_A_CONFIRM_ARCHIVE_ACQUISITION_V1"
            ),

            "candidate_rank": rank,

            "source_match_id": match_id,

            "frozen_match_date": row["match_date"],

            "frozen_match_url": row["source_url"],

            "inspected_match_page_url": (
                source_record["page_url"]
            ),

            "match_page_sha256": (
                source_record["page_sha256"]
            ),

            "source_download_url": download_url,

            "resolved_download_url": final_url,

            "archive_path": str(
                final_path.relative_to(ROOT)
            ),

            "archive_size_bytes": bytes_written,

            "archive_sha256": digest.hexdigest(),

            "acquired_utc": iso_utc(
                datetime.now(timezone.utc)
            ),

            "archive_signature_extension": (
                extension
            ),

            "technical_eligibility": "NOT_EVALUATED",

            "demo_extracted": False,

            "model_scoring_performed": False,
        }

        record_path = (
            rank_directory
            / "acquisition_record.json"
        )

        with record_path.open(
            "x",
            encoding="utf-8",
        ) as output:

            json.dump(
                record,
                output,
                indent=2,
                ensure_ascii=False,
            )

            output.write("\n")

        print()
        print("=== ARCHIVE ACQUISITION COMPLETE ===")

        print("Archive:", final_path)

        print(
            "Archive bytes:",
            bytes_written,
        )

        print(
            "Archive SHA256:",
            digest.hexdigest(),
        )

        print(
            "Acquisition record:",
            record_path,
        )

        print(
            "Remaining disk:",
            f"{free_bytes() / GIB:.2f} GiB",
        )

        print("Technical eligibility: NOT EVALUATED")

    except Exception:

        # Remove only the incomplete file created by
        # this attempt. Never delete an already finalized
        # archive or any historical Demo.
        if part_path.exists():
            part_path.unlink()

        if (
            final_path is None
            or not final_path.exists()
        ):

            if (
                rank_directory.exists()
                and not any(rank_directory.iterdir())
            ):
                rank_directory.rmdir()

        raise

    finally:
        session.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Frozen candidate_rank.",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Explicitly authorize a single archive "
            "download for Rank 1."
        ),
    )

    args = parser.parse_args()

    row, original_start = verify_selected_candidate(
        args.rank
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

    print(
        "Original scheduled UTC:",
        iso_utc(original_start),
    )

    print(
        "Current UTC:",
        iso_utc(datetime.now(timezone.utc)),
    )

    print(
        "Free disk:",
        f"{free_bytes() / GIB:.2f} GiB",
    )

    budget = safe_download_budget()

    print(
        "Current maximum archive budget:",
        f"{budget / MIB:.1f} MiB",
    )

    if not args.download:

        print()
        print("MODE: OFFLINE PLAN")

        print("HLTV requests: NONE")
        print("Archive downloads: NONE")
        print("Technical eligibility decisions: NONE")

        return

    # This implementation has no durable Technical Intake
    # Ledger yet. Do not silently skip a lower-ranked match.
    require(
        args.rank == 1,
        "RANK_ORDER_BLOCK: only Rank 1 is authorized "
        "until the Technical Intake Ledger is implemented.",
    )

    now = datetime.now(timezone.utc)

    if now < original_start:

        print()
        print("SOURCE STATUS: SCHEDULED")

        print(
            "The frozen scheduled start has not passed."
        )

        print("HLTV requests: NONE")
        print("Archive downloads: NONE")

        return

    print()
    print("=== INSPECT CURRENT MATCH SOURCE ===")

    source_record = inspect_match_page(row)

    links = source_record["demo_links"]

    print(
        "Match-page SHA256:",
        source_record["page_sha256"],
    )

    print(
        "Visible Demo links:",
        len(links),
    )

    if not links:

        print()
        print("SOURCE STATUS: AWAITING_DEMO_OR_MATCH_RESULT")

        print(
            "No archive downloaded. "
            "No technical exclusion assigned."
        )

        return

    if len(links) != 1:

        print()
        print("SOURCE STATUS: REVIEW_REQUIRED")

        print(
            "Multiple distinct Demo links were found. "
            "No archive downloaded."
        )

        for link in links:
            print(" ", link)

        return

    download_one_archive(
        row=row,
        source_record=source_record,
    )

    print()
    print("Frozen Queue: UNCHANGED")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NOT PERFORMED")


if __name__ == "__main__":

    try:
        main()

    except RuntimeError as exc:

        print(
            "\nACQUISITION STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
