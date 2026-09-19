"""V4 Gate 0C: inspect the one missing player place in development."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo


NAME = (
    "2026-09-12_iowa-stormboar_vs_"
    "sportsbetexpert_mirage.dem"
)

MANIFEST = Path("docs/v3_dev_manifest.csv")
TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
AUDIT = Path("docs/v4_gate0b_development_audit_v1.json")

MIN_FREE_BYTES = 12 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def free_gib():
    return shutil.disk_usage(".").free / 1024**3


print("\n=== V4 GATE 0C: MISSING PLACE INVESTIGATION ===")

require(
    shutil.disk_usage(".").free >= MIN_FREE_BYTES,
    "Disk space below 12 GiB.",
)

for path in (MANIFEST, TARGETS, AUDIT):
    require(path.is_file(), f"Missing file: {path}")

report = json.loads(AUDIT.read_text())

require(
    report["status"] == "GATE_0B_NEEDS_REVIEW",
    "Unexpected Gate 0B status.",
)

require(
    report["quality_issues"] == {"unmapped_player_rows": 1},
    "Gate 0B contains additional quality issues.",
)

affected = [
    row
    for row in report["per_match"]
    if row["counts"].get("unmapped_player_rows", 0) > 0
]

require(
    len(affected) == 1
    and affected[0]["demo_filename"] == NAME
    and affected[0]["unmapped_places"] == {"": 1},
    "The observed missing-place record differs from Gate 0B.",
)

with MANIFEST.open(newline="") as file:
    matches = list(csv.DictReader(file))

selected = [
    row for row in matches
    if row["demo_filename"] == NAME
]

require(
    len(selected) == 1
    and selected[0]["dev_source_role"] == "V2_RESERVE",
    "Selected demo is not the expected development match.",
)

path = Path(selected[0]["path"])

require(path.is_file(), f"Demo missing: {path}")

targets = pl.read_parquet(TARGETS).filter(
    pl.col("demo_filename") == NAME
)

current = targets.select(
    "round_num",
    "current_tick",
    "horizon_sec",
).unique()

keys = {
    (int(row["round_num"]), int(row["current_tick"]))
    for row in current.iter_rows(named=True)
}

ticks = {tick for _, tick in keys}

print("Development demo:", NAME)
print("Expected unmapped rows: 1")
print(f"Free disk: {free_gib():.2f} GiB")
print("\nParsing this one development demo...", flush=True)

demo = open_v0_demo(path, verbose=False)

demo.parse(
    player_props=[
        *V0_PLAYER_PROPS,
        "last_place_name",
    ]
)

required = {
    "round_num",
    "tick",
    "side",
    "health",
    "place",
    "X",
    "Y",
    "Z",
}

require(
    required <= set(demo.ticks.columns),
    "Player snapshot schema is incomplete.",
)

snapshots = demo.ticks.filter(
    pl.col("tick").is_in(list(ticks))
).select(sorted(required))

missing = snapshots.filter(
    pl.col("side").is_in(["t", "ct"])
    & (pl.col("health") > 0)
    & (pl.col("place").fill_null("").str.strip_chars() == "")
)

records = [
    row
    for row in missing.iter_rows(named=True)
    if (
        int(row["round_num"]),
        int(row["tick"]),
    ) in keys
]

require(
    len(records) == 1,
    f"Expected one missing-place row, found {len(records)}.",
)

row = records[0]

key = (
    int(row["round_num"]),
    int(row["tick"]),
)

snapshot = snapshots.filter(
    (pl.col("round_num") == key[0])
    & (pl.col("tick") == key[1])
)

alive = snapshot.filter(
    pl.col("side").is_in(["t", "ct"])
    & (pl.col("health") > 0)
)

t_count = alive.filter(
    pl.col("side") == "t"
).height

ct_count = alive.filter(
    pl.col("side") == "ct"
).height

xyz = {
    axis: row[axis]
    for axis in ("X", "Y", "Z")
}

xyz_valid = all(
    value is not None and math.isfinite(float(value))
    for value in xyz.values()
)

horizons = sorted(
    targets.filter(
        (pl.col("round_num") == key[0])
        & (pl.col("current_tick") == key[1])
    )["horizon_sec"].unique().to_list()
)

print("\n=== EXACT MISSING-PLACE RECORD ===")
print("Demo:", NAME)
print("Round:", key[0])
print("Current tick:", key[1])
print("Player side:", row["side"])
print("Player health:", row["health"])
print("Raw place:", repr(row["place"]))
print("XYZ:", xyz)
print("XYZ valid:", xyz_valid)
print("Living T players in snapshot:", t_count)
print("Living CT players in snapshot:", ct_count)
print("Affected target horizons:", horizons)

print("\n=== INTERPRETATION ===")

if xyz_valid and 1 <= t_count <= 5 and 1 <= ct_count <= 5:
    print("GATE_0C_TRANSIENT_PLACE_MISSINGNESS_CONFIRMED")
    print(
        "The player is alive and has valid coordinates, "
        "but the current place label is unavailable."
    )
    print(
        "Do not infer a macro-zone from future information."
    )
else:
    print("GATE_0C_NEEDS_FURTHER_REVIEW")

print(
    "\nNo frozen mapping, targets, models or "
    "confirmation artifacts were modified."
)
print(f"Free disk: {free_gib():.2f} GiB")
