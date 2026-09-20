#!/usr/bin/env python3

"""
V4-A Confirmation: Initial Technical Intake Ledger.

Create a one-time, immutable starting state for the
25 candidates in the already frozen acquisition queue.

No Demo acquisition, technical exclusion, model prediction,
or confirmation scoring occurs in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from datetime import datetime, timezone
from pathlib import Path

from probe_v4_a_confirm_demo_source_v1 import (
    ROOT,
    EVIDENCE,
    QUEUE,
    MANIFEST,
    read_frozen_queue,
    iso_utc,
)


LEDGER = ROOT / (
    "docs/v4_a_confirm_intake_initial_state_v1.json"
)

STAGING = ROOT / (
    "data/raw/v4_a_confirm_download_staging"
)

VERSION = "V4_A_CONFIRM_INTAKE_INITIAL_STATE_V1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def queue_sha256():
    return hashlib.sha256(
        QUEUE.read_bytes()
    ).hexdigest()


def build_initial_state():
    rows, starts = read_frozen_queue()

    evidence = json.loads(
        EVIDENCE.read_text(encoding="utf-8")
    )

    digest = queue_sha256()

    require(
        digest == evidence["queue_sha256"],
        "Queue SHA256 differs from frozen evidence.",
    )

    require(
        len(rows) == 25,
        "Initial Ledger requires exactly 25 candidates.",
    )

    candidates = []

    for row in rows:
        rank = int(row["candidate_rank"])
        match_id = row["source_match_id"]

        candidates.append(
            {
                "candidate_rank": rank,
                "source_match_id": match_id,
                "frozen_match_date": row["match_date"],
                "frozen_match_url": row["source_url"],
                "original_scheduled_start_utc": iso_utc(
                    starts[match_id]
                ),
                "initial_status": "PENDING",
                "technical_eligibility_evaluated": False,
                "demo_acquired": False,
                "model_scoring_performed": False,
            }
        )

    return {
        "version": VERSION,
        "dataset": "D_V4_A_CONFIRM",
        "frozen_queue_path": str(
            QUEUE.relative_to(ROOT)
        ),
        "frozen_queue_sha256": digest,
        "queue_freeze_commit": "fed0637",
        "downloader_commit": "81798ad",
        "required_eligible_matches": 20,
        "candidate_count": 25,
        "initial_state_only": True,
        "candidates": candidates,
    }


def validate_initial_state(state):
    expected = build_initial_state()

    require(
        state == expected,
        "Initial Ledger differs from the frozen "
        "Queue-derived starting state.",
    )

    require(
        [
            entry["candidate_rank"]
            for entry in state["candidates"]
        ] == list(range(1, 26)),
        "Candidate rank sequence mismatch.",
    )

    require(
        all(
            entry["initial_status"] == "PENDING"
            and entry["technical_eligibility_evaluated"] is False
            and entry["demo_acquired"] is False
            and entry["model_scoring_performed"] is False
            for entry in state["candidates"]
        ),
        "Initial Ledger contains a premature decision.",
    )


def initialize():
    require(
        not MANIFEST.exists(),
        "Final Confirmation Manifest already exists.",
    )

    require(
        not STAGING.exists(),
        "V4 download staging exists. "
        "Do not initialize a clean Ledger over acquired data.",
    )

    require(
        not LEDGER.exists(),
        "Initial Ledger already exists. "
        "Do not overwrite its frozen starting state.",
    )

    state = build_initial_state()

    validate_initial_state(state)

    payload = (
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")

    temporary = LEDGER.with_name(
        LEDGER.name + f".tmp.{os.getpid()}"
    )

    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

        # Create the official Ledger path only if it
        # does not already exist. Never overwrite it.
        os.link(temporary, LEDGER)

    finally:
        if temporary.exists():
            temporary.unlink()

    print("Initial Ledger created:", LEDGER)
    print("Candidate count: 25")
    print("Initial status: PENDING for all candidates")


def check():
    require(
        LEDGER.is_file(),
        "Initial Ledger does not exist.",
    )

    state = json.loads(
        LEDGER.read_text(encoding="utf-8")
    )

    validate_initial_state(state)

    print("Initial Ledger identity: VERIFIED")
    print("Frozen Queue SHA256: VERIFIED")
    print("Candidate ranks: 1 through 25")
    print("Technical eligibility decisions: NONE")

    print()
    print("=== INITIAL CANDIDATE STATES ===")

    for entry in state["candidates"]:
        print(
            f'{entry["candidate_rank"]:02d}'
            f' | {entry["source_match_id"]}'
            f' | {entry["initial_status"]}'
        )


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--init",
        action="store_true",
        help="Create the initial Ledger exactly once.",
    )

    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing initial Ledger.",
    )

    args = parser.parse_args()

    if args.init:
        initialize()
    else:
        check()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileExistsError, OSError) as exc:
        print(
            "INTAKE LEDGER STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
