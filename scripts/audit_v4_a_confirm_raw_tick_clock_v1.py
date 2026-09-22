#!/usr/bin/env python3

"""
V4-A Confirmation: read-only raw tick clock audit.

Reuses the V3 technical measurement precedent:
    median(delta_tick / delta_game_time)

Only positive tick and game-time differences with
delta_game_time < 2 seconds contribute to the estimate.

A configured parser tickrate is not an independent
measurement of the raw Demo clock.

This module does not:
    select a Mirage Demo;
    assign final technical eligibility;
    modify the frozen queue;
    create the final Confirmation Manifest;
    perform model scoring.
"""

from __future__ import annotations

import argparse
import math
import sys

from pathlib import Path
from statistics import median

from manage_v4_a_confirm_intake_state_v1 import (
    STAGING_ROOT,
    load_initial_candidates,
    rank_dir_name,
    read_event,
    require,
)

from extract_v4_a_confirm_zip_v1 import (
    EXTRACT_DIR_NAME,
)

from load_v4_a_confirm_verified_demos_v1 import (
    load_verified_demos,
)

EXPECTED_RAW_TICKS_PER_SECOND = 64.0

TICKRATE_TOLERANCE = 0.25


def estimate_raw_clock(observations):
    """
    observations:
        Iterable of (tick, game_time) pairs.

    Return:
        (median_raw_ticks_per_second, valid_interval_count)

    Input rows must already be sorted by tick.

    The filtering rule matches the V3 confirmation
    raw-clock measurement precedent.
    """

    ratios = []
    previous = None

    for tick, game_time in observations:

        current = (
            int(tick),
            float(game_time),
        )

        if previous is not None:

            dtick = current[0] - previous[0]

            dtime = current[1] - previous[1]

            if (
                dtick > 0
                and math.isfinite(dtime)
                and dtime > 0
                and dtime < 2.0
            ):

                ratio = dtick / dtime

                if math.isfinite(ratio):
                    ratios.append(ratio)

        previous = current

    require(
        bool(ratios),
        "RAW_CLOCK_REVIEW_REQUIRED: "
        "no usable game_time intervals.",
    )

    return float(median(ratios)), len(ratios)


def measure_demo_raw_clock(path):
    """
    Parse actual Demo game_time observations.

    This is deliberately deferred until an extracted
    Demo has passed the existing identity checks.
    """

    from awpy import Demo

    from cs2_tactical_intelligence.v0.timing import (
        V0_DEMO_TICKS_PER_SECOND,
    )

    demo = Demo(
        Path(path),
        tickrate=V0_DEMO_TICKS_PER_SECOND,
        verbose=False,
    )

    clock = (
        demo.parse_ticks(
            other_props=["game_time"]
        )
        .select([
            "tick",
            "game_time",
        ])
        .drop_nulls()
        .unique()
        .sort("tick")
    )

    observations = (
        (row["tick"], row["game_time"])
        for row in clock.iter_rows(named=True)
    )

    measured, interval_count = estimate_raw_clock(
        observations
    )

    return {
        "measured_raw_ticks_per_second": measured,
        "valid_clock_intervals": interval_count,
        "expected_raw_ticks_per_second": (
            EXPECTED_RAW_TICKS_PER_SECOND
        ),
        "absolute_difference": abs(
            measured - EXPECTED_RAW_TICKS_PER_SECOND
        ),
        "within_v3_precedent_tolerance": (
            abs(
                measured - EXPECTED_RAW_TICKS_PER_SECOND
            ) <= TICKRATE_TOLERANCE
        ),
        "parser_configured_tickrate": float(
            demo.tickrate
        ),
        "measurement_method": (
            "median_adjacent_positive_tick_over_game_time"
        ),
    }


def audit_rank(rank, *, measure):

    require(
        rank == 1,
        "RANK_ORDER_BLOCK: only Rank 1 is "
        "currently authorized.",
    )

    row = load_initial_candidates()[rank - 1]

    print()
    print("=== V4 RAW TICK CLOCK AUDIT ===")
    print("Candidate Rank:", rank)
    print("Match ID:", row["source_match_id"])

    event = read_event(row)

    if event is None:

        print("STATUS: WAITING_FOR_VERIFIED_ARCHIVE")
        print("Raw tick clock measured: NO")
        print("Technical eligibility: NOT EVALUATED")

        return

    verified = load_verified_demos(row, event)

    demos = [
        item
        for item in verified
        if item["is_demo_filename"]
    ]

    print("Verified Demo filenames:", len(demos))

    if not measure:

        print("STATUS: VERIFIED_DEMOS_READY_FOR_CLOCK_AUDIT")
        print("Raw tick clock measured: NO")
        print("Technical eligibility: NOT EVALUATED")

        return

    for item in demos:

        print()
        print("Demo:", item["path"])
        print("Verified SHA256:", item["sha256"])

        try:

            result = measure_demo_raw_clock(
                item["absolute_path"]
            )

        except Exception as exc:

            print("CLOCK STATUS: REVIEW_REQUIRED")
            print("Parser/clock error:", repr(exc))
            print(
                "No technical exclusion is assigned "
                "by this preliminary audit."
            )

            continue

        print(
            "Measured raw ticks/second:",
            result["measured_raw_ticks_per_second"],
        )

        print(
            "Valid clock intervals:",
            result["valid_clock_intervals"],
        )

        print(
            "Configured parser tickrate:",
            result["parser_configured_tickrate"],
        )

        print(
            "Within V3 precedent tolerance:",
            result["within_v3_precedent_tolerance"],
        )

    print()
    print("STATUS: RAW_CLOCK_INSPECTED")
    print("Final technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "Explicitly measure the raw clock of "
            "every identity-verified extracted Demo."
        ),
    )

    args = parser.parse_args()

    audit_rank(
        args.rank,
        measure=args.measure,
    )


if __name__ == "__main__":

    try:
        main()

    except (
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:

        print(
            "\nRAW CLOCK AUDIT STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
