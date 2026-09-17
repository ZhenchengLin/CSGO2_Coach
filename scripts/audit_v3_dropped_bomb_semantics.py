from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
)

from cs2_tactical_intelligence.v0.timing import (
    open_v0_demo,
)


INCOMING = Path(
    "data/raw/v2_incoming"
)

MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

OUTPUT = Path(
    "data/interim/v3_drop_semantic_recovery_audit.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Frozen macro mapping
# ============================================================

require(
    MAPPING.exists(),
    "Frozen Gate 1 mapping is missing.",
)

mapping_record = json.loads(
    MAPPING.read_text()
)

require(
    mapping_record["status"]
    == "FROZEN",
    "Macro-zone mapping is not frozen.",
)

place_to_zone = {}

for zone, places in (
    mapping_record["zones"].items()
):

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate fine place: {place}",
        )

        place_to_zone[
            place
        ] = zone


# ============================================================
# Select only V2 reserves
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


print("=" * 100)
print("V3 GATE 2B — DROPPED BOMB SEMANTIC RECOVERY AUDIT")
print("=" * 100)

print()
print("Reserve demos:", len(reserve))
print("V2 D_CONFIRM excluded:", len(confirm))


# ============================================================
# Audit
# ============================================================

rows = []

for demo_index, demo_path in enumerate(
    reserve,
    start=1,
):

    print()
    print(
        f"[{demo_index:02d}/13] "
        f"{demo_path.name}"
    )

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )

    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )

    require(
        "place" in demo.ticks.columns,
        "Awpy normalized place column missing.",
    )

    drops = (
        demo.bomb
        .filter(
            pl.col("event")
            == "drop"
        )
        .sort(
            [
                "round_num",
                "tick",
            ]
        )
    )

    resolved_count = 0

    for drop in drops.iter_rows(
        named=True
    ):

        round_num = int(
            drop["round_num"]
        )

        drop_tick = int(
            drop["tick"]
        )

        steamid = (
            drop["steamid"]
        )

        result = {
            "demo_filename":
                demo_path.name,

            "round_num":
                round_num,

            "drop_tick":
                drop_tick,

            "steamid":
                steamid,

            "drop_X":
                drop["X"],

            "drop_Y":
                drop["Y"],

            "drop_Z":
                drop["Z"],

            "resolved":
                False,

            "resolved_tick":
                None,

            "tick_lateness":
                None,

            "player_X":
                None,

            "player_Y":
                None,

            "player_Z":
                None,

            "xyz_distance":
                None,

            "place":
                None,

            "macro_zone":
                None,

            "reason":
                None,
        }


        # ----------------------------------------------------
        # Event must identify the player who dropped the C4.
        # ----------------------------------------------------

        if steamid is None:

            result[
                "reason"
            ] = "DROP_MISSING_STEAMID"

            rows.append(
                result
            )

            continue


        # ----------------------------------------------------
        # Historical snapshot contract:
        #
        # first player snapshot at or after the event,
        # with at most +1 raw tick lateness.
        # ----------------------------------------------------

        candidates = (
            demo.ticks
            .filter(
                (
                    pl.col("round_num")
                    == round_num
                )
                &
                (
                    pl.col("steamid")
                    == steamid
                )
                &
                (
                    pl.col("tick")
                    >= drop_tick
                )
                &
                (
                    pl.col("tick")
                    <= drop_tick + 1
                )
            )
            .sort(
                "tick"
            )
        )


        if candidates.height == 0:

            result[
                "reason"
            ] = "DROP_PLAYER_SNAPSHOT_UNRESOLVED"

            rows.append(
                result
            )

            continue


        player = candidates.row(
            0,
            named=True,
        )

        place = player[
            "place"
        ]


        if (
            place is None
            or str(place) == ""
        ):

            result[
                "reason"
            ] = "DROP_PLACE_UNRESOLVED"

            rows.append(
                result
            )

            continue


        place = str(
            place
        )


        if place not in place_to_zone:

            result[
                "reason"
            ] = (
                "DROP_PLACE_OUTSIDE_FROZEN_SEMANTIC_SEED"
            )

            result[
                "place"
            ] = place

            rows.append(
                result
            )

            continue


        # ----------------------------------------------------
        # Geometry sanity diagnostic.
        #
        # This is NOT used to create the label.
        # ----------------------------------------------------

        dx = float(
            player["X"]
            - drop["X"]
        )

        dy = float(
            player["Y"]
            - drop["Y"]
        )

        dz = float(
            player["Z"]
            - drop["Z"]
        )

        distance = math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )


        result.update({
            "resolved":
                True,

            "resolved_tick":
                int(
                    player[
                        "tick"
                    ]
                ),

            "tick_lateness":
                int(
                    player[
                        "tick"
                    ]
                    - drop_tick
                ),

            "player_X":
                float(
                    player["X"]
                ),

            "player_Y":
                float(
                    player["Y"]
                ),

            "player_Z":
                float(
                    player["Z"]
                ),

            "xyz_distance":
                distance,

            "place":
                place,

            "macro_zone":
                place_to_zone[
                    place
                ],

            "reason":
                "RESOLVED",
        })


        resolved_count += 1

        rows.append(
            result
        )


    print(
        "  drop events:",
        drops.height,
    )

    print(
        "  resolved:",
        resolved_count,
    )


# ============================================================
# Save
# ============================================================

audit = pl.DataFrame(
    rows
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.write_csv(
    OUTPUT
)


# ============================================================
# Summary
# ============================================================

total = audit.height

resolved = audit.filter(
    pl.col("resolved")
    == True
).height

unresolved = (
    total
    - resolved
)

coverage = (
    resolved / total
    if total
    else 0.0
)


print()
print("=" * 100)
print("GLOBAL DROP RECOVERY SUMMARY")
print("=" * 100)

print(
    "Drop events:",
    total,
)

print(
    "Resolved:",
    resolved,
)

print(
    "Unresolved:",
    unresolved,
)

print(
    "Recovery coverage:",
    f"{coverage:.6%}",
)


if resolved:

    resolved_rows = audit.filter(
        pl.col("resolved")
        == True
    )

    print()
    print(
        "Tick lateness:"
    )

    print(
        resolved_rows
        .group_by(
            "tick_lateness"
        )
        .len()
        .sort(
            "tick_lateness"
        )
    )


    print()
    print(
        "Recovered macro-zone counts:"
    )

    print(
        resolved_rows
        .group_by(
            "macro_zone"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )


    print()
    print(
        "XYZ distance diagnostics:"
    )

    print(
        resolved_rows
        .select([
            pl.col(
                "xyz_distance"
            ).mean().alias(
                "mean"
            ),

            pl.col(
                "xyz_distance"
            ).median().alias(
                "median"
            ),

            pl.col(
                "xyz_distance"
            ).quantile(
                0.95
            ).alias(
                "p95"
            ),

            pl.col(
                "xyz_distance"
            ).max().alias(
                "max"
            ),
        ])
    )


if unresolved:

    print()
    print(
        "Unresolved reasons:"
    )

    print(
        audit
        .filter(
            pl.col("resolved")
            == False
        )
        .group_by(
            "reason"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )


print()
print("=" * 100)
print("GATE 2B RESULT")
print("=" * 100)


if (
    total > 0
    and unresolved == 0
):

    print(
        "✅ DROPPED BOMB SEMANTIC RECOVERY PASS"
    )

    print(
        "All audited drop events resolve through "
        "the Valve semantic source."
    )

    print(
        "No XYZ classifier is required for "
        "dropped-bomb target labels."
    )

    print(
        "NEXT_ACTION=FREEZE_FUTURE_TARGET_CONTRACT"
    )

else:

    print(
        "⚠️ DROPPED BOMB SEMANTIC RECOVERY "
        "REQUIRES INVESTIGATION"
    )

    print(
        "No fallback should be added yet."
    )


print("=" * 100)

print()
print("Artifact:")
print(" ", OUTPUT)
