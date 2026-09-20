"""Replay one existing development demo against frozen V3 motion.

Read-only research audit. No training or confirmation scoring.
"""

from __future__ import annotations

import csv
import hashlib
import math
import shutil
import os
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo
from cs2_tactical_intelligence.v4_a_confirm_motion_v1 import (
    calculate_motion,
)


MANIFEST = Path("docs/v3_dev_manifest.csv")
MOTION = Path("data/interim/v3_b1_motion_inputs_v1.parquet")

EXPECTED_MOTION_SHA = (
    "412cee7968abbf340e5c49dffebe848b0e6101202b9f7dbbfb5585f115c42289"
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def same_number(actual, expected, label):
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ),
        f"{label}: actual={actual}, expected={expected}",
    )


print("\n=== V4 MOTION DEVELOPMENT DEMO REPLAY ===")

free_gib = shutil.disk_usage(".").free / 1024**3
require(free_gib >= 12, f"Free disk below 12 GiB: {free_gib:.2f}")

require(MANIFEST.is_file(), f"Missing {MANIFEST}")
require(MOTION.is_file(), f"Missing {MOTION}")
require(
    sha256(MOTION) == EXPECTED_MOTION_SHA,
    "Frozen V3 motion cache SHA256 mismatch.",
)

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(len(manifest) == 53, "Unexpected development manifest size.")

# Select by frozen manifest order, not by model performance.
record = manifest[int(os.environ.get("V4_DEV_REPLAY_INDEX", "0"))]
name = record["demo_filename"]
demo_path = Path(record["path"])

require(demo_path.is_file(), f"Missing development demo: {demo_path}")

cache = pl.read_parquet(MOTION).filter(
    pl.col("demo_filename") == name
)

require(cache.height > 0, "No frozen motion rows for the selected demo.")
require(
    cache.filter(pl.col("status") != "RESOLVED").height == 0,
    "Selected demo contains unresolved frozen motion rows.",
)

print("Development demo:", name)
print("Frozen motion rows:", cache.height)

# This is an existing Development demo, not a confirmation candidate.
demo = open_v0_demo(demo_path, verbose=False)

require(
    demo.header.get("map_name") == "de_mirage",
    "Unexpected parsed map.",
)

demo.parse(
    player_props=[
        *V0_PLAYER_PROPS,
        "last_place_name",
    ]
)

required_player_columns = {
    "round_num",
    "tick",
    "steamid",
    "side",
    "health",
    "X",
    "Y",
    "Z",
    "inventory",
}

require(
    required_player_columns <= set(demo.ticks.columns),
    "Parsed player snapshot schema is incomplete.",
)

required_event_columns = {
    "round_num",
    "tick",
    "event",
    "steamid",
    "X",
    "Y",
    "Z",
}

require(
    required_event_columns <= set(demo.bomb.columns),
    "Parsed bomb-event schema is incomplete.",
)

# Exactly the same ordering prerequisite as the frozen V3 implementation.
tick_rows = (
    demo.ticks
    .select(["round_num", "tick"])
    .unique()
    .sort(["round_num", "tick"])
)

ticks_by_round = {}

for row in tick_rows.iter_rows(named=True):
    ticks_by_round.setdefault(
        int(row["round_num"]), []
    ).append(int(row["tick"]))

events_by_round = {}

for row in (
    demo.bomb
    .sort(["round_num", "tick"])
    .iter_rows(named=True)
):
    events_by_round.setdefault(
        int(row["round_num"]), []
    ).append(row)

# Materialize only the historical snapshots requested by the frozen cache.
required_keys = {
    (int(row["round_num"]), int(row["prior_tick"]))
    for row in cache.iter_rows(named=True)
}

required_ticks = sorted({tick for _, tick in required_keys})

snapshots = {}

subset = (
    demo.ticks
    .filter(pl.col("tick").is_in(required_ticks))
    .select(sorted(required_player_columns))
)

for row in subset.iter_rows(named=True):
    key = (int(row["round_num"]), int(row["tick"]))

    if key in required_keys:
        snapshots.setdefault(key, []).append(row)

require(
    set(snapshots) == required_keys,
    "Historical snapshot coverage differs from frozen cache.",
)

print("Historical player snapshots: PASS")

checked = 0

for row in cache.iter_rows(named=True):
    round_num = int(row["round_num"])
    current_tick = int(row["current_tick"])

    result = calculate_motion(
        current_tick=current_tick,
        current_xyz=(
            row["current_bomb_X"],
            row["current_bomb_Y"],
            row["current_bomb_Z"],
        ),
        available_ticks=ticks_by_round.get(round_num, []),
        snapshots={
            int(row["prior_tick"]): snapshots[
                (round_num, int(row["prior_tick"]))
            ]
        },
        events=events_by_round.get(round_num, []),
    )

    require(
        result["status"] == "RESOLVED",
        f"New module returned unresolved motion at {round_num}:{current_tick}",
    )

    exact_fields = [
        "prior_nominal_tick",
        "prior_tick",
        "elapsed_ticks",
        "prior_source",
        "status",
        "reason",
    ]

    for field in exact_fields:
        require(
            result[field] == row[field],
            f"{round_num}:{current_tick} — {field} mismatch: "
            f"{result[field]!r} != {row[field]!r}",
        )

    numerical_fields = [
        "prior_bomb_X",
        "prior_bomb_Y",
        "prior_bomb_Z",
        "elapsed_sec",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ]

    for field in numerical_fields:
        same_number(
            result[field],
            row[field],
            f"{round_num}:{current_tick} — {field}",
        )

    checked += 1

require(checked == cache.height, "Not all frozen motion rows were checked.")

print("\n=== DEVELOPMENT REPLAY SUMMARY ===")
print("Replayed rows:", checked)
print("Historical snapshot selection: PASS")
print("Historical bomb XYZ/source: PASS")
print("Elapsed time and XYZ velocity: PASS")
print("\nV4_MOTION_SINGLE_DEV_DEMO_REPLAY_PASS")
print("No confirmation data was accessed or scored.")
