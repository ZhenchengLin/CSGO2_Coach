from __future__ import annotations

from bisect import bisect_left
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


EXCLUSIONS = Path(
    "data/interim/v3_dev_target_exclusions.csv"
)

ROUND_EXCLUSIONS = Path(
    "data/interim/v3_dev_round_exclusions.csv"
)

MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)


BOUNDARY_REASON = (
    "TARGET_AFTER_ROUND_END"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


for path in [
    EXCLUSIONS,
    ROUND_EXCLUSIONS,
    MANIFEST,
]:
    require(
        path.exists(),
        f"Missing artifact: {path}",
    )


exclusions = pl.read_csv(
    EXCLUSIONS,
    infer_schema_length=None,
)

round_exclusions = pl.read_csv(
    ROUND_EXCLUSIONS,
    infer_schema_length=None,
)

manifest = pl.read_csv(
    MANIFEST,
    infer_schema_length=None,
)


nonboundary = (
    exclusions
    .filter(
        pl.col("reason")
        != BOUNDARY_REASON
    )
    .sort([
        "demo_filename",
        "round_num",
        "current_nominal_tick",
        "horizon_sec",
    ])
)


print("=" * 108)
print("V3 GATE 3C.1 — NON-BOUNDARY EXCLUSION AUDIT")
print("=" * 108)

print()
print("GLOBAL COUNTS")
print("-" * 108)

print(
    "All horizon exclusions:",
    exclusions.height,
)

print(
    "Round-boundary exclusions:",
    exclusions
    .filter(
        pl.col("reason")
        == BOUNDARY_REASON
    )
    .height,
)

print(
    "Non-boundary horizon exclusions:",
    nonboundary.height,
)


print()
print("NON-BOUNDARY REASON COUNTS")
print("-" * 108)

print(
    nonboundary
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
print("NON-BOUNDARY ROWS")
print("-" * 108)

print(
    nonboundary
)


# ============================================================
# Collapse repeated +5/+10 exclusions at same current point
# ============================================================

problem_points = (
    nonboundary
    .group_by([
        "demo_filename",
        "round_num",
        "current_nominal_tick",
        "reason",
    ])
    .agg([
        pl.col(
            "horizon_sec"
        )
        .sort()
        .alias(
            "horizons"
        ),

        pl.len().alias(
            "n_horizon_rows"
        ),
    ])
    .sort([
        "demo_filename",
        "round_num",
        "current_nominal_tick",
        "reason",
    ])
)


print()
print("UNIQUE PROBLEM POINTS")
print("-" * 108)

print(
    problem_points
)


# ============================================================
# Manifest lookup
# ============================================================

path_by_demo = {
    row[
        "demo_filename"
    ]:
        Path(
            row[
                "path"
            ]
        )
    for row in manifest.iter_rows(
        named=True
    )
}


# ============================================================
# Helpers
# ============================================================

def round_ticks(
    demo,
    round_num,
):
    return (
        demo.ticks
        .filter(
            pl.col("round_num")
            == round_num
        )[
            "tick"
        ]
        .unique()
        .sort()
        .to_list()
    )


def nearest_tick_report(
    ticks,
    nominal,
):
    if not ticks:
        return {
            "before":
                None,

            "after":
                None,

            "before_gap":
                None,

            "after_gap":
                None,
        }

    index = bisect_left(
        ticks,
        nominal,
    )


    before = (
        ticks[
            index - 1
        ]
        if index > 0
        else None
    )


    after = (
        ticks[
            index
        ]
        if index
        < len(
            ticks
        )
        else None
    )


    return {
        "before":
            before,

        "after":
            after,

        "before_gap":
            (
                nominal - before
                if before is not None
                else None
            ),

        "after_gap":
            (
                after - nominal
                if after is not None
                else None
            ),
    }


def resolve_current_tick(
    ticks,
    nominal,
):
    info = nearest_tick_report(
        ticks,
        nominal,
    )

    after = info[
        "after"
    ]

    if after is None:
        return None

    gap = (
        after
        - nominal
    )

    if (
        gap < 0
        or gap > 1
    ):
        return None

    return int(
        after
    )


def latest_bomb_event(
    demo,
    round_num,
    tick,
):
    events = (
        demo.bomb
        .filter(
            (
                pl.col(
                    "round_num"
                )
                == round_num
            )
            &
            (
                pl.col(
                    "tick"
                )
                <= tick
            )
        )
        .sort(
            "tick"
        )
    )


    if events.height == 0:
        return None


    return events.row(
        -1,
        named=True,
    )


# ============================================================
# Deep diagnostics, demo by demo
# ============================================================

anomaly_demos = (
    nonboundary[
        "demo_filename"
    ]
    .unique()
    .sort()
    .to_list()
)


print()
print("=" * 108)
print("DEEP DIAGNOSTICS")
print("=" * 108)


for demo_filename in anomaly_demos:

    demo_path = (
        path_by_demo.get(
            demo_filename
        )
    )

    require(
        demo_path is not None,
        (
            "Demo not found in manifest: "
            f"{demo_filename}"
        ),
    )


    print()
    print("#" * 108)
    print(
        demo_filename
    )
    print("#" * 108)


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


    rows = (
        nonboundary
        .filter(
            pl.col(
                "demo_filename"
            )
            == demo_filename
        )
        .sort([
            "round_num",
            "current_nominal_tick",
            "horizon_sec",
        ])
        .to_dicts()
    )


    seen = set()


    for row in rows:

        key = (
            int(
                row[
                    "round_num"
                ]
            ),
            int(
                row[
                    "current_nominal_tick"
                ]
            ),
            str(
                row[
                    "reason"
                ]
            ),
        )


        if key in seen:
            continue

        seen.add(
            key
        )


        round_num = key[
            0
        ]

        current_nominal = key[
            1
        ]

        reason = key[
            2
        ]


        ticks = round_ticks(
            demo,
            round_num,
        )


        current_tick = (
            resolve_current_tick(
                ticks,
                current_nominal,
            )
        )


        current_tick_info = (
            nearest_tick_report(
                ticks,
                current_nominal,
            )
        )


        matching_horizons = sorted({
            int(
                item[
                    "horizon_sec"
                ]
            )
            for item in rows
            if (
                int(
                    item[
                        "round_num"
                    ]
                )
                == round_num
                and int(
                    item[
                        "current_nominal_tick"
                    ]
                )
                == current_nominal
                and str(
                    item[
                        "reason"
                    ]
                )
                == reason
            )
        })


        print()
        print("-" * 108)

        print(
            "round:",
            round_num,
        )

        print(
            "reason:",
            reason,
        )

        print(
            "horizons:",
            matching_horizons,
        )

        print(
            "current nominal tick:",
            current_nominal,
        )

        print(
            "nearest tick info:",
            current_tick_info,
        )

        print(
            "resolved current tick:",
            current_tick,
        )


        # ----------------------------------------------------
        # Snapshot availability
        # ----------------------------------------------------

        if reason == (
            "CURRENT_SNAPSHOT_UNAVAILABLE"
        ):

            continue


        # ----------------------------------------------------
        # Need a valid current snapshot beyond here
        # ----------------------------------------------------

        if current_tick is None:

            print(
                "⚠️ Could not resolve current tick."
            )

            continue


        snapshot = (
            demo.ticks
            .filter(
                (
                    pl.col(
                        "round_num"
                    )
                    == round_num
                )
                &
                (
                    pl.col(
                        "tick"
                    )
                    == current_tick
                )
            )
        )


        print(
            "snapshot player rows:",
            snapshot.height,
        )


        # ----------------------------------------------------
        # Target snapshot diagnostic
        # ----------------------------------------------------

        if reason == (
            "TARGET_SNAPSHOT_UNAVAILABLE"
        ):

            for horizon in (
                matching_horizons
            ):

                target_nominal = (
                    current_tick
                    + horizon
                    * V0_DEMO_TICKS_PER_SECOND
                )


                info = (
                    nearest_tick_report(
                        ticks,
                        target_nominal,
                    )
                )


                print(
                    f"+{horizon}s target nominal:",
                    target_nominal,
                )

                print(
                    f"+{horizon}s nearest tick:",
                    info,
                )


            continue


        # ----------------------------------------------------
        # Bomb-state diagnostic
        # ----------------------------------------------------

        if reason == (
            "CURRENT_BOMB_STATE_UNRESOLVED"
        ):

            carriers = []


            for player in snapshot.iter_rows(
                named=True
            ):

                if (
                    player[
                        "side"
                    ]
                    == "t"
                    and has_c4(
                        player[
                            "inventory"
                        ]
                    )
                ):

                    carriers.append(
                        player
                    )


            print(
                "inventory C4 carriers:",
                len(
                    carriers
                ),
            )


            for carrier in carriers:

                print(
                    "  carrier:",
                    {
                        "steamid":
                            carrier[
                                "steamid"
                            ],

                        "name":
                            carrier[
                                "name"
                            ],

                        "health":
                            carrier[
                                "health"
                            ],

                        "place":
                            carrier[
                                "place"
                            ],

                        "X":
                            carrier[
                                "X"
                            ],

                        "Y":
                            carrier[
                                "Y"
                            ],

                        "Z":
                            carrier[
                                "Z"
                            ],
                    },
                )


            latest = (
                latest_bomb_event(
                    demo,
                    round_num,
                    current_tick,
                )
            )


            print(
                "latest bomb event <= current tick:",
                latest,
            )


            alive_t = (
                snapshot
                .filter(
                    (
                        pl.col(
                            "side"
                        )
                        == "t"
                    )
                    &
                    (
                        pl.col(
                            "health"
                        )
                        > 0
                    )
                )
                .select([
                    "steamid",
                    "name",
                    "health",
                    "place",
                    "X",
                    "Y",
                    "Z",
                ])
            )


            print(
                "alive T players:"
            )

            print(
                alive_t
            )


# ============================================================
# Missing freeze-end rounds
# ============================================================

print()
print("=" * 108)
print("ROUND-LEVEL EXCLUSIONS")
print("=" * 108)

print(
    round_exclusions
    .sort([
        "demo_filename",
        "round_num",
    ])
)


print()
print("=" * 108)
print(
    "✅ NON-BOUNDARY EXCLUSION DIAGNOSTIC COMPLETE"
)
print(
    "NEXT_ACTION=CLASSIFY_ANOMALIES_BEFORE_CV_FREEZE"
)
print("=" * 108)
