from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import polars as pl
from awpy import Demo

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
)

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
)


# ============================================================
# Frozen inputs
# ============================================================

QUEUE = Path(
    "docs/v3_confirm_acquisition_queue_v1.csv"
)

ACQUISITION_PROTOCOL = Path(
    "docs/v3_confirm_acquisition_protocol_v1.json"
)

DEV_MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

V2_CONFIRM_MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

TARGET_BUILDER = Path(
    "scripts/build_v3_confirm_candidate_targets.py"
)

RAW_DIR = Path(
    "data/raw/v3_confirm_incoming"
)

AUDIT_ROOT = Path(
    "docs/v3_confirm_intake_audits"
)

INTERIM_ROOT = Path(
    "data/interim/v3_confirm_intake"
)


EXPECTED_RAW_TICKS_PER_SEC = 64.0
TICKRATE_TOLERANCE = 0.25


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# Exact V2 raw-clock measurement precedent
# ============================================================

def measure_raw_ticks_per_second(
    path: Path,
) -> tuple[float, int]:

    demo = Demo(
        path,
        tickrate=V0_DEMO_TICKS_PER_SECOND,
        verbose=False,
    )

    clock = (
        demo.parse_ticks(
            other_props=[
                "game_time",
            ]
        )
        .select([
            "tick",
            "game_time",
        ])
        .drop_nulls()
        .unique()
        .sort("tick")
    )

    ratios = []
    previous = None

    for current in clock.iter_rows(
        named=True
    ):

        if previous is not None:

            dtick = (
                current["tick"]
                - previous["tick"]
            )

            dtime = (
                current["game_time"]
                - previous["game_time"]
            )

            if (
                dtick > 0
                and dtime > 0
                and dtime < 2.0
            ):
                ratios.append(
                    dtick / dtime
                )

        previous = current


    if not ratios:
        raise RuntimeError(
            "No usable game_time intervals "
            "for raw tick-clock audit."
        )


    return (
        float(
            median(ratios)
        ),
        len(ratios),
    )


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Technical-only intake audit for one frozen "
        "V3 confirmation candidate."
    )
)

parser.add_argument(
    "--candidate-rank",
    type=int,
    required=True,
)

args = parser.parse_args()


require(
    1 <= args.candidate_rank <= 25,
    "candidate-rank must be in [1, 25].",
)


# ============================================================
# Frozen queue row
# ============================================================

queue = pl.read_csv(
    QUEUE,
    infer_schema_length=None,
)


require(
    queue.height == 25,
    f"Expected 25 frozen candidates, found {queue.height}.",
)


candidate = queue.filter(
    pl.col("candidate_rank")
    == args.candidate_rank
)


require(
    candidate.height == 1,
    "Candidate rank did not resolve to exactly one row.",
)


row = candidate.row(
    0,
    named=True,
)


date = str(
    row["date"]
)

team1 = str(
    row["team1"]
)

team2 = str(
    row["team2"]
)


demo_filename = (
    f"{date}_"
    f"{team1}_vs_"
    f"{team2}_mirage.dem"
)

demo_path = (
    RAW_DIR
    / demo_filename
)


require(
    demo_path.exists(),
    (
        "Downloaded confirmation demo is missing: "
        f"{demo_path}"
    ),
)


AUDIT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


audit_output = (
    AUDIT_ROOT
    / (
        f"rank_{args.candidate_rank:02d}.json"
    )
)


require(
    not audit_output.exists(),
    (
        "Candidate intake audit already exists. "
        "Refusing overwrite."
    ),
)


# ============================================================
# Base result
# ============================================================

result = {
    "candidate_rank":
        int(
            args.candidate_rank
        ),

    "match_date":
        date,

    "match_id":
        str(
            row["match_id"]
        ),

    "team1":
        team1,

    "team2":
        team2,

    "demo_filename":
        demo_filename,

    "demo_path":
        str(
            demo_path
        ),

    "audited_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "technical_status":
        None,

    "technical_exclusion_reason":
        None,

    "model_prediction_performed":
        False,

    "confirmation_metric_calculated":
        False,

    "class_distribution_reported":
        False,
}


def finish_excluded(reason: str):
    result[
        "technical_status"
    ] = "EXCLUDED"

    result[
        "technical_exclusion_reason"
    ] = reason

    audit_output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "TECHNICAL STATUS: EXCLUDED"
    )

    print(
        "reason:",
        reason,
    )

    print(
        "model prediction: NO"
    )

    print(
        "confirmation metric: NO"
    )

    print()
    print(
        "V3_CONFIRM_CANDIDATE_AUDIT_COMPLETE"
    )

    raise SystemExit(0)


# ============================================================
# SHA identity and overlap
# ============================================================

digest = sha256_file(
    demo_path
)


result[
    "demo_sha256"
] = digest


dev_manifest = pl.read_csv(
    DEV_MANIFEST,
    infer_schema_length=None,
)

v2_manifest = pl.read_csv(
    V2_CONFIRM_MANIFEST,
    infer_schema_length=None,
)


if digest in set(
    dev_manifest["sha256"].to_list()
):
    finish_excluded(
        "DEVELOPMENT_SHA_OVERLAP"
    )


if digest in set(
    v2_manifest["sha256"].to_list()
):
    finish_excluded(
        "V2_CONFIRM_SHA_OVERLAP"
    )


# Check duplicate SHA among already acquired confirmation demos.
sha_owners = []

for other in sorted(
    RAW_DIR.glob("*.dem")
):

    if other == demo_path:
        continue

    try:
        other_sha = sha256_file(
            other
        )
    except Exception:
        continue

    if other_sha == digest:
        sha_owners.append(
            other.name
        )


if sha_owners:

    result[
        "duplicate_confirmation_sha_owners"
    ] = sha_owners

    finish_excluded(
        "DUPLICATE_SHA"
    )


result[
    "sha_overlap_check"
] = "PASS"


# ============================================================
# Header / configured timing
# ============================================================

try:

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )

except Exception as exc:

    result[
        "parser_error"
    ] = repr(
        exc
    )

    finish_excluded(
        "PARSER_FAILURE"
    )


map_name = demo.header.get(
    "map_name"
)


result[
    "map_name"
] = map_name


if map_name != "de_mirage":

    finish_excluded(
        "WRONG_MAP"
    )


result[
    "configured_tickrate"
] = float(
    demo.tickrate
)


if (
    demo.tickrate
    != V0_DEMO_TICKS_PER_SECOND
):

    finish_excluded(
        "WRONG_TICK_RATE"
    )


# ============================================================
# Independent raw-clock measurement
# ============================================================

try:

    measured_tickrate, n_intervals = (
        measure_raw_ticks_per_second(
            demo_path
        )
    )

except Exception as exc:

    result[
        "raw_tick_clock_error"
    ] = repr(
        exc
    )

    finish_excluded(
        "REQUIRED_TELEMETRY_UNAVAILABLE"
    )


result[
    "measured_raw_ticks_per_sec"
] = measured_tickrate

result[
    "raw_clock_intervals"
] = int(
    n_intervals
)


tick_clock_ok = (
    abs(
        measured_tickrate
        - EXPECTED_RAW_TICKS_PER_SEC
    )
    <= TICKRATE_TOLERANCE
)


result[
    "raw_tick_clock_ok"
] = bool(
    tick_clock_ok
)


if not tick_clock_ok:

    finish_excluded(
        "WRONG_TICK_RATE"
    )


# ============================================================
# Required V3 semantic telemetry
# ============================================================

try:

    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )

except Exception as exc:

    result[
        "parser_error"
    ] = repr(
        exc
    )

    finish_excluded(
        "PARSER_FAILURE"
    )


required_tick_columns = {
    "tick",
    "round_num",
    "steamid",
    "place",
}


missing_tick_columns = sorted(
    required_tick_columns
    - set(
        demo.ticks.columns
    )
)


result[
    "missing_required_tick_columns"
] = missing_tick_columns


if missing_tick_columns:

    finish_excluded(
        "REQUIRED_TELEMETRY_UNAVAILABLE"
    )


result[
    "parser_and_required_telemetry"
] = "PASS"


# ============================================================
# Frozen target semantics
#
# Any unexpected target-builder implementation failure is NOT
# silently converted into a scientific exclusion. Stop instead.
# ============================================================

command = [
    sys.executable,
    str(
        TARGET_BUILDER
    ),
    "--candidate-rank",
    str(
        args.candidate_rank
    ),
]


completed = subprocess.run(
    command,
    check=False,
)


require(
    completed.returncode == 0,
    (
        "Frozen target builder failed. "
        "Stop and investigate implementation; "
        "do not classify this candidate from the failure."
    ),
)


target_summary_path = (
    INTERIM_ROOT
    / (
        f"rank_{args.candidate_rank:02d}"
    )
    / "target_build_summary.json"
)


require(
    target_summary_path.exists(),
    (
        "Target builder returned success but summary "
        "artifact is missing."
    ),
)


target_summary = json.loads(
    target_summary_path.read_text()
)


require(
    target_summary[
        "demo_sha256"
    ]
    == digest,
    "Target-build SHA does not match intake SHA.",
)


plus5 = int(
    target_summary[
        "valid_rows_by_horizon"
    ][
        "5"
    ]
)

plus10 = int(
    target_summary[
        "valid_rows_by_horizon"
    ][
        "10"
    ]
)


result[
    "valid_rows_by_horizon"
] = {
    "5":
        plus5,

    "10":
        plus10,
}


result[
    "drop_semantics"
] = target_summary[
    "drop_semantics"
]


if plus5 <= 0:

    finish_excluded(
        "ZERO_ELIGIBLE_PLUS5_ROWS"
    )


if plus10 <= 0:

    finish_excluded(
        "ZERO_ELIGIBLE_PLUS10_ROWS"
    )


# ============================================================
# Eligible
# ============================================================

result[
    "technical_status"
] = "ELIGIBLE"

result[
    "technical_exclusion_reason"
] = None


result[
    "scientific_boundary"
] = {
    "model_prediction_performed":
        False,

    "confirmation_metric_calculated":
        False,

    "class_distribution_reported":
        False,

    "target_distribution_used_for_selection":
        False,
}


audit_output.write_text(
    json.dumps(
        result,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 88)
print("V3 CONFIRM TECHNICAL INTAKE AUDIT")
print("=" * 88)

print(
    "candidate rank:",
    args.candidate_rank,
)

print(
    "demo SHA:",
    digest,
)

print(
    "map:",
    map_name,
)

print(
    "configured tickrate:",
    demo.tickrate,
)

print(
    "measured raw tickrate:",
    f"{measured_tickrate:.6f}",
)

print(
    "+5 valid rows:",
    plus5,
)

print(
    "+10 valid rows:",
    plus10,
)

print(
    "drop unresolved:",
    result[
        "drop_semantics"
    ][
        "unresolved"
    ],
)

print()
print("TECHNICAL STATUS: ELIGIBLE")
print("model prediction: NO")
print("confirmation metric: NO")
print("class distribution reported: NO")

print()
print(
    "V3_CONFIRM_CANDIDATE_AUDIT_COMPLETE"
)
