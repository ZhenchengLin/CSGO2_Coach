from __future__ import annotations

from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
    has_c4,
)

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
)


INCOMING = Path(
    "data/raw/v2_incoming"
)

MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Select first V2 reserve demo only
# ============================================================

manifest = pl.read_csv(
    MANIFEST
)

confirm = set(
    manifest[
        "demo_filename"
    ].to_list()
)

reserve = sorted(
    path
    for path in INCOMING.glob("*.dem")
    if path.name not in confirm
)

require(
    len(reserve) == 13,
    f"Expected 13 reserve demos, found {len(reserve)}",
)

demo_path = reserve[0]


print("=" * 100)
print("V3 GATE 2A — FUTURE TARGET SOURCE AUDIT")
print("=" * 100)

print()
print("Audit demo:")
print(" ", demo_path.name)


# ============================================================
# Parse with existing V0 contract + semantic field
# ============================================================

demo = open_v0_demo(
    demo_path,
    verbose=False,
)

require(
    demo.tickrate
    == V0_DEMO_TICKS_PER_SECOND,
    "Historical tickrate contract violated.",
)


player_props = [
    *V0_PLAYER_PROPS,
    "last_place_name",
]


print()
print(
    "Requested player props:",
    player_props,
)


demo.parse(
    player_props=player_props
)


print()
print("Map:")
print(
    " ",
    demo.header.get(
        "map_name"
    ),
)

print(
    "Raw ticks/sec:",
    demo.tickrate,
)


# ============================================================
# Tick schema
# ============================================================

print()
print("=" * 100)
print("PLAYER TICK SCHEMA")
print("=" * 100)

for column in demo.ticks.columns:
    print(" ", column)


require(
    "place"
    in demo.ticks.columns,
    (
        "Awpy did not expose normalized place column. "
        "Do not build Gate 2 targets yet."
    ),
)


# ============================================================
# last_place_name availability
# ============================================================

alive = demo.ticks.filter(
    (pl.col("health") > 0)
    &
    pl.col("side").is_in([
        "t",
        "ct",
    ])
)


labelled_alive = alive.filter(
    pl.col(
        "place"
    ).is_not_null()
    &
    (
        pl.col(
            "place"
        )
        != ""
    )
)


alive_coverage = (
    labelled_alive.height
    / alive.height
    if alive.height
    else 0.0
)


print()
print("=" * 100)
print("SEMANTIC FIELD AVAILABILITY")
print("=" * 100)

print(
    "Alive competitive rows:",
    f"{alive.height:,}",
)

print(
    "Rows with Awpy place:",
    f"{labelled_alive.height:,}",
)

print(
    "Coverage:",
    f"{alive_coverage:.6%}",
)


# ============================================================
# Bomb schema
# ============================================================

print()
print("=" * 100)
print("BOMB EVENT SCHEMA")
print("=" * 100)

print(
    "Bomb rows:",
    demo.bomb.height,
)

print()
print("Columns:")

for column in demo.bomb.columns:
    print(" ", column)


print()
print("Event counts:")

event_counts = (
    demo.bomb
    .group_by(
        "event"
    )
    .len()
    .sort(
        "len",
        descending=True,
    )
)

print(
    event_counts
)


# ============================================================
# Show representative bomb events
# ============================================================

candidate_columns = [
    "round_num",
    "tick",
    "event",
    "bombsite",
    "steamid",
    "X",
    "Y",
    "Z",
]

existing_columns = [
    column
    for column
    in candidate_columns
    if column
    in demo.bomb.columns
]


for event_name in [
    "pickup",
    "drop",
    "plant",
]:

    subset = (
        demo.bomb
        .filter(
            pl.col("event")
            == event_name
        )
        .select(
            existing_columns
        )
        .head(5)
    )

    print()
    print(
        f"{event_name.upper()} examples:"
    )

    print(
        subset
    )


# ============================================================
# Find a carried-bomb snapshot with semantic place
# ============================================================

print()
print("=" * 100)
print("CARRIED BOMB + SEMANTIC PLACE CHECK")
print("=" * 100)


required_carrier_columns = [
    "round_num",
    "tick",
    "steamid",
    "name",
    "side",
    "health",
    "X",
    "Y",
    "Z",
    "inventory",
    "place",
]


for column in required_carrier_columns:

    require(
        column in demo.ticks.columns,
        f"Missing required carrier column: {column}",
    )


carrier_example = None


for row in (
    demo.ticks
    .select(
        required_carrier_columns
    )
    .filter(
        (pl.col("side") == "t")
        &
        (pl.col("health") > 0)
    )
    .iter_rows(
        named=True
    )
):

    if not has_c4(
        row["inventory"]
    ):
        continue

    if (
        row[
            "place"
        ]
        in [
            None,
            "",
        ]
    ):
        continue

    carrier_example = row
    break


require(
    carrier_example
    is not None,
    (
        "Could not find a carried-bomb snapshot "
        "with last_place_name."
    ),
)


print(
    "round:",
    carrier_example[
        "round_num"
    ],
)

print(
    "tick:",
    carrier_example[
        "tick"
    ],
)

print(
    "player:",
    carrier_example[
        "name"
    ],
)

print(
    "steamid:",
    carrier_example[
        "steamid"
    ],
)

print(
    "XYZ:",
    (
        carrier_example["X"],
        carrier_example["Y"],
        carrier_example["Z"],
    ),
)

print(
    "place:",
    carrier_example[
        "place"
    ],
)


# ============================================================
# Plant-source check
# ============================================================

plants = demo.bomb.filter(
    pl.col("event")
    == "plant"
)


print()
print("=" * 100)
print("PLANT TARGET SOURCE CHECK")
print("=" * 100)

print(
    "Plant events:",
    plants.height,
)


if "bombsite" in plants.columns:

    print()
    print(
        plants
        .group_by(
            "bombsite"
        )
        .len()
        .sort(
            "bombsite"
        )
    )


plant_xyz_available = all(
    column in plants.columns
    for column in [
        "X",
        "Y",
        "Z",
    ]
)


print()
print(
    "Plant XYZ columns available:",
    plant_xyz_available,
)


# ============================================================
# Final feasibility result
# ============================================================

print()
print("=" * 100)
print("GATE 2A RESULT")
print("=" * 100)

print(
    "64-tick historical contract: PASS"
)

print(
    "last_place_name -> Awpy place normalization: PASS"
)

print(
    "C4 carrier + semantic place: PASS"
)

print(
    "Bomb event stream available: PASS"
)

print(
    "Plant site available:",
    "PASS"
    if "bombsite" in plants.columns
    else "FAIL",
)

print()

if (
    alive_coverage >= 0.95
    and "bombsite"
    in plants.columns
):

    print(
        "✅ V3 TARGET SOURCES ARE VIABLE"
    )

    print(
        "NEXT_ACTION=FREEZE_FUTURE_TARGET_CONTRACT"
    )

else:

    print(
        "⚠️ V3 TARGET SOURCE AUDIT "
        "REQUIRES INVESTIGATION"
    )

print("=" * 100)
