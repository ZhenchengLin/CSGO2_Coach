"""V4 Gate 0A: read-only, single-development-demo team-state feasibility."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo


MANIFEST = Path("docs/v3_dev_manifest.csv")
TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")
SPATIAL = Path(
    "data/interim/v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)

MIN_FREE_BYTES = 12 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def gib_free():
    return shutil.disk_usage(".").free / 1024**3


print("=== V4 GATE 0A: READ-ONLY FEASIBILITY ===")
print(f"Free disk: {gib_free():.2f} GiB")

require(
    shutil.disk_usage(".").free >= MIN_FREE_BYTES,
    "Disk space below the existing 12 GiB safety threshold.",
)

for path in (MANIFEST, TARGETS, MAPPING, SPATIAL):
    require(path.is_file(), f"Missing existing development file: {path}")

mapping = json.loads(MAPPING.read_text())

require(
    mapping["status"] == "FROZEN"
    and len(mapping["zones"]) == 15,
    "Unexpected frozen macro-zone mapping.",
)

place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate mapped place: {place}",
        )
        place_to_zone[place] = zone


# The spatial cache is inspected only, not modified.
spatial_columns = set(pl.read_parquet_schema(SPATIAL))

print("\n=== EXISTING SPATIAL CACHE ===")
print("Has XYZ:", {"X", "Y", "Z"} <= spatial_columns)
print("Has place:", "place" in spatial_columns)
print("Has player side:", "side" in spatial_columns)
print("Has raw tick:", "tick" in spatial_columns)

# Select the first locally available V3 development demo.
with MANIFEST.open(newline="") as file:
    development = list(csv.DictReader(file))

require(
    len(development) == 53,
    "Expected exactly 53 entries in V3 development manifest.",
)

selected = next(
    (
        row
        for row in development
        if Path(row["path"]).is_file()
    ),
    None,
)

require(
    selected is not None,
    "No existing V3 development demo is available locally.",
)

name = selected["demo_filename"]
demo_path = Path(selected["path"])

require(
    selected["map_name"] == "de_mirage"
    and int(selected["raw_ticks_per_sec"]) == 64,
    "Selected development demo violates frozen map/tick contract.",
)

targets = pl.read_parquet(TARGETS).filter(
    pl.col("demo_filename") == name
)

require(
    targets.height > 0,
    f"No existing V3 development targets for {name}.",
)

required_target_columns = {
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
    "horizon_sec",
}

require(
    required_target_columns <= set(targets.columns),
    "Existing V3 target dataset lacks required time keys.",
)

current = targets.select(
    "round_num",
    "current_nominal_tick",
    "current_tick",
).unique()

require(
    current.height > 0,
    "No current-time observations in selected development demo.",
)

lateness = (
    current["current_tick"]
    - current["current_nominal_tick"]
)

require(
    ((lateness >= 0) & (lateness <= 1)).all(),
    "Current snapshot tick violates the frozen lateness policy.",
)

requested = {
    (int(row["round_num"]), int(row["current_tick"]))
    for row in current.iter_rows(named=True)
}

required_ticks = {
    tick
    for _, tick in requested
}

print("\n=== SELECTED DEVELOPMENT DEMO ===")
print("Demo:", name)
print("Existing target rows:", targets.height)
print("Unique current-time snapshots:", len(requested))

print("\nParsing one EXISTING development demo...", flush=True)

demo = open_v0_demo(demo_path, verbose=False)

require(
    demo.header.get("map_name") == "de_mirage",
    "Parsed demo map mismatch.",
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
    "side",
    "health",
    "X",
    "Y",
    "Z",
    "place",
}

require(
    required_player_columns <= set(demo.ticks.columns),
    "Parser lacks required player snapshot columns: "
    + str(
        sorted(required_player_columns - set(demo.ticks.columns))
    ),
)

snapshots = demo.ticks.filter(
    pl.col("tick").is_in(list(required_ticks))
).select(sorted(required_player_columns))

groups = {
    (int(group["round_num"][0]), int(group["tick"][0])): group
    for group in snapshots.partition_by(
        ["round_num", "tick"],
        maintain_order=False,
    )
    if (
        int(group["round_num"][0]),
        int(group["tick"][0]),
    ) in requested
}

missing_snapshots = requested - set(groups)

total_alive = 0
mapped_alive = 0
invalid_places = Counter()
zone_counts = Counter()
side_counts = Counter()
empty_t = 0
empty_ct = 0
invalid_xyz = 0

for key in sorted(requested):
    snapshot = groups.get(key)

    if snapshot is None:
        continue

    alive = snapshot.filter(
        pl.col("side").is_in(["t", "ct"])
        & (pl.col("health") > 0)
    )

    side_counts["t"] += alive.filter(
        pl.col("side") == "t"
    ).height

    side_counts["ct"] += alive.filter(
        pl.col("side") == "ct"
    ).height

    if alive.filter(pl.col("side") == "t").height == 0:
        empty_t += 1

    if alive.filter(pl.col("side") == "ct").height == 0:
        empty_ct += 1

    for row in alive.iter_rows(named=True):
        total_alive += 1

        place = row["place"]

        if place in place_to_zone:
            mapped_alive += 1
            zone_counts[place_to_zone[place]] += 1
        else:
            invalid_places[str(place)] += 1

        for axis in ("X", "Y", "Z"):
            value = row[axis]

            if value is None:
                invalid_xyz += 1
            else:
                import math
                if not math.isfinite(float(value)):
                    invalid_xyz += 1

require(
    total_alive > 0,
    "No living T/CT player observations were recovered.",
)

print("\n=== GATE 0A OBSERVATIONS ===")
print("Requested current-time snapshots:", len(requested))
print("Recovered exact (round, tick) snapshots:", len(groups))
print("Missing exact snapshots:", len(missing_snapshots))
print("Living T player rows:", side_counts["t"])
print("Living CT player rows:", side_counts["ct"])
print("All living player rows:", total_alive)
print("Rows mapped to a frozen macro-zone:", mapped_alive)
print(
    "Macro-zone mapping coverage:",
    f"{100 * mapped_alive / total_alive:.2f}%",
)
print("Snapshots without a living T player:", empty_t)
print("Snapshots without a living CT player:", empty_ct)
print("Invalid XYZ coordinate values:", invalid_xyz)
print("Unmapped places:", dict(invalid_places.most_common(10)))
print("Observed macro-zones:", len(zone_counts), "of 15")

print("\n=== INTERPRETATION ===")

if missing_snapshots:
    print(
        "NEEDS_REVIEW: Some exact current-time snapshots were missing."
    )
elif invalid_xyz:
    print(
        "NEEDS_REVIEW: Some living player coordinates were invalid."
    )
elif invalid_places:
    print(
        "NEEDS_REVIEW: Some living player place labels were unmapped."
    )
else:
    print(
        "GATE_0A_SINGLE_DEMO_FEASIBILITY_PASS"
    )

print(
    "This checks one development demo only; "
    "it does not establish 53-match coverage."
)
print(
    "No confirmation data, model inference, training, "
    "or scoring was performed."
)
print(f"Free disk after audit: {gib_free():.2f} GiB")
