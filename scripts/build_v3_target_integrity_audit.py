from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
    build_plant_lookup,
    has_c4,
)

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
    seconds_to_demo_ticks,
)


INCOMING = Path(
    "data/raw/v2_incoming"
)

MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

CONTRACT = Path(
    "docs/v3_future_target_contract.json"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

DROP_AUDIT = Path(
    "data/interim/v3_drop_semantic_recovery_audit.csv"
)

OBS_OUTPUT = Path(
    "data/interim/v3_target_integrity_observations.csv"
)

ROUND_OUTPUT = Path(
    "data/interim/v3_target_integrity_round_exclusions.csv"
)

SUMMARY_OUTPUT = Path(
    "docs/v3_target_integrity_summary.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Load frozen contracts
# ============================================================

contract = json.loads(
    CONTRACT.read_text()
)

mapping_record = json.loads(
    MAPPING.read_text()
)

require(
    contract["status"] == "FROZEN",
    "V3 target contract is not frozen.",
)

require(
    mapping_record["status"] == "FROZEN",
    "V3 macro-zone mapping is not frozen.",
)


CURRENT_START_SEC = int(
    contract[
        "observation_grid"
    ][
        "first_current_observation_sec"
    ]
)

CURRENT_STRIDE_SEC = int(
    contract[
        "observation_grid"
    ][
        "stride_sec"
    ]
)

TARGET_HORIZONS_SEC = list(
    contract[
        "prediction_horizons_sec"
    ]
)

MAX_LATENESS = int(
    contract[
        "tick_clock"
    ][
        "max_snapshot_lateness_ticks"
    ]
)


require(
    TARGET_HORIZONS_SEC == [5, 10],
    (
        "Unexpected frozen target horizons: "
        f"{TARGET_HORIZONS_SEC}"
    ),
)


# ============================================================
# Fine place -> frozen macro-zone
# ============================================================

place_to_zone = {}

for zone, places in (
    mapping_record[
        "zones"
    ].items()
):

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate mapped place: {place}",
        )

        place_to_zone[
            place
        ] = zone


require(
    len(place_to_zone) == 23,
    (
        "Expected 23 fine semantic places, "
        f"found {len(place_to_zone)}."
    ),
)


macro_zones = set(
    mapping_record[
        "zones"
    ]
)


# ============================================================
# V2 reserve identity
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


# ============================================================
# Frozen exact-tick dropped-bomb semantics
# ============================================================

drop_audit = pl.read_csv(
    DROP_AUDIT
)

require(
    drop_audit.filter(
        pl.col("resolved") != True
    ).height == 0,
    "Drop audit contains unresolved rows.",
)


drop_lookup = {}

for row in drop_audit.iter_rows(
    named=True
):

    key = (
        str(
            row[
                "demo_filename"
            ]
        ),
        int(
            row[
                "round_num"
            ]
        ),
        int(
            row[
                "drop_tick"
            ]
        ),
    )

    require(
        key not in drop_lookup,
        f"Duplicate drop semantic key: {key}",
    )

    macro_zone = str(
        row[
            "macro_zone"
        ]
    )

    place = str(
        row[
            "place"
        ]
    )

    require(
        macro_zone in macro_zones,
        (
            "Drop audit contains unknown macro-zone: "
            f"{macro_zone}"
        ),
    )

    require(
        place in place_to_zone,
        (
            "Drop audit contains unknown semantic place: "
            f"{place}"
        ),
    )

    require(
        place_to_zone[
            place
        ] == macro_zone,
        (
            "Drop audit semantic mapping disagrees "
            "with frozen Gate 1 mapping."
        ),
    )

    drop_lookup[
        key
    ] = {
        "place":
            place,

        "macro_zone":
            macro_zone,
    }


# ============================================================
# Helpers
# ============================================================

def build_ticks_by_round(demo):
    result = {}

    rows = (
        demo.ticks
        .select([
            "round_num",
            "tick",
        ])
        .unique()
        .sort([
            "round_num",
            "tick",
        ])
    )

    for row in rows.iter_rows(
        named=True
    ):

        round_num = int(
            row[
                "round_num"
            ]
        )

        tick = int(
            row[
                "tick"
            ]
        )

        result.setdefault(
            round_num,
            [],
        ).append(
            tick
        )

    return result


def resolve_snapshot_tick(
    available_ticks,
    nominal_tick,
    *,
    upper_bound_exclusive=None,
):
    """
    Frozen V3/V0 historical timing rule:

    Use first available snapshot at or after nominal tick.
    Maximum lateness = +1 raw tick.
    """

    if not available_ticks:
        return None

    index = bisect_left(
        available_ticks,
        nominal_tick,
    )

    if index >= len(
        available_ticks
    ):
        return None

    actual_tick = int(
        available_ticks[
            index
        ]
    )

    lateness = (
        actual_tick
        - nominal_tick
    )

    if (
        lateness < 0
        or lateness > MAX_LATENESS
    ):
        return None

    if (
        upper_bound_exclusive
        is not None
        and actual_tick
        >= upper_bound_exclusive
    ):
        return None

    return {
        "tick":
            actual_tick,

        "lateness":
            int(
                lateness
            ),
    }


def build_events_by_round(demo):
    result = {}

    rows = (
        demo.bomb
        .sort([
            "round_num",
            "tick",
        ])
    )

    for row in rows.iter_rows(
        named=True
    ):

        round_num = int(
            row[
                "round_num"
            ]
        )

        result.setdefault(
            round_num,
            [],
        ).append(
            row
        )

    return result


def latest_event_at_or_before(
    events,
    tick,
):
    if not events:
        return None

    event_ticks = [
        int(
            event[
                "tick"
            ]
        )
        for event in events
    ]

    index = (
        bisect_right(
            event_ticks,
            tick,
        )
        - 1
    )

    if index < 0:
        return None

    return events[
        index
    ]


def snapshot_rows_for_keys(
    demo,
    required_keys,
):
    """
    Materialize only the snapshots needed by Gate 2D.

    key = (round_num, tick)
    """

    if not required_keys:
        return {}

    required_ticks = sorted({
        tick
        for _, tick
        in required_keys
    })

    subset = (
        demo.ticks
        .filter(
            pl.col(
                "tick"
            ).is_in(
                required_ticks
            )
        )
        .select([
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
        ])
    )

    snapshots = {}

    for row in subset.iter_rows(
        named=True
    ):

        key = (
            int(
                row[
                    "round_num"
                ]
            ),
            int(
                row[
                    "tick"
                ]
            ),
        )

        if key not in required_keys:
            continue

        snapshots.setdefault(
            key,
            [],
        ).append(
            row
        )

    return snapshots


def map_place(
    place,
    *,
    unresolved_reason,
):
    if (
        place is None
        or str(
            place
        ) == ""
    ):
        return {
            "ok":
                False,

            "error":
                unresolved_reason,
        }

    place = str(
        place
    )

    if place not in place_to_zone:
        return {
            "ok":
                False,

            "error":
                unresolved_reason,
        }

    return {
        "ok":
            True,

        "place":
            place,

        "macro_zone":
            place_to_zone[
                place
            ],
    }


def resolve_unplanted_bomb_semantic(
    *,
    demo_filename,
    round_num,
    tick,
    snapshot,
    events,
    place_error_reason,
    state_error_reason,
):
    """
    Resolve bomb semantic zone without using future evidence.

    Priority:
      1. Current inventory carrier
      2. Latest pickup event identifies current carrier
      3. Latest drop event uses frozen exact-tick drop semantic

    Plant state is handled by the caller before this function.
    """

    carriers = []

    for player in snapshot:

        if (
            player[
                "side"
            ] == "t"
            and has_c4(
                player[
                    "inventory"
                ]
            )
        ):

            carriers.append(
                player
            )


    # --------------------------------------------------------
    # Inventory carrier
    # --------------------------------------------------------

    if len(carriers) > 1:
        raise RuntimeError(
            "Invariant violation: multiple C4 carriers "
            f"demo={demo_filename} "
            f"round={round_num} "
            f"tick={tick}"
        )

    if len(carriers) == 1:

        carrier = carriers[
            0
        ]

        mapped = map_place(
            carrier[
                "place"
            ],
            unresolved_reason=
                place_error_reason,
        )

        if not mapped[
            "ok"
        ]:

            return mapped

        return {
            "ok":
                True,

            "source":
                "CARRIED_INVENTORY",

            "place":
                mapped[
                    "place"
                ],

            "macro_zone":
                mapped[
                    "macro_zone"
                ],
        }


    # --------------------------------------------------------
    # No inventory carrier:
    # use latest event at or before this state tick
    # --------------------------------------------------------

    event = latest_event_at_or_before(
        events,
        tick,
    )

    if event is None:

        return {
            "ok":
                False,

            "error":
                state_error_reason,
        }


    event_type = str(
        event[
            "event"
        ]
    )


    # --------------------------------------------------------
    # Pickup fallback
    # --------------------------------------------------------

    if event_type == "pickup":

        pickup_steamid = (
            event[
                "steamid"
            ]
        )

        if pickup_steamid is None:

            return {
                "ok":
                    False,

                "error":
                    state_error_reason,
            }


        matching = [
            player
            for player in snapshot
            if (
                player[
                    "side"
                ] == "t"
                and player[
                    "steamid"
                ]
                == pickup_steamid
            )
        ]


        if len(matching) != 1:

            return {
                "ok":
                    False,

                "error":
                    state_error_reason,
            }


        carrier = matching[
            0
        ]


        if (
            carrier[
                "health"
            ] is None
            or carrier[
                "health"
            ] <= 0
        ):

            return {
                "ok":
                    False,

                "error":
                    state_error_reason,
            }


        mapped = map_place(
            carrier[
                "place"
            ],
            unresolved_reason=
                place_error_reason,
        )


        if not mapped[
            "ok"
        ]:

            return mapped


        return {
            "ok":
                True,

            "source":
                "CARRIED_PICKUP_FALLBACK",

            "place":
                mapped[
                    "place"
                ],

            "macro_zone":
                mapped[
                    "macro_zone"
                ],
        }


    # --------------------------------------------------------
    # Dropped bomb
    # --------------------------------------------------------

    if event_type == "drop":

        drop_key = (
            demo_filename,
            int(
                round_num
            ),
            int(
                event[
                    "tick"
                ]
            ),
        )

        recovered = (
            drop_lookup.get(
                drop_key
            )
        )


        if recovered is None:

            return {
                "ok":
                    False,

                "error":
                    place_error_reason,
            }


        return {
            "ok":
                True,

            "source":
                "DROPPED",

            "place":
                recovered[
                    "place"
                ],

            "macro_zone":
                recovered[
                    "macro_zone"
                ],
        }


    # --------------------------------------------------------
    # A canonical plant should have been handled before
    # calling this resolver.
    # --------------------------------------------------------

    return {
        "ok":
            False,

        "error":
            state_error_reason,
    }


# ============================================================
# Main audit
# ============================================================

observation_rows = []
round_exclusions = []

print("=" * 104)
print("V3 GATE 2D — FUTURE TARGET INTEGRITY AUDIT")
print("=" * 104)

print()
print(
    "Reserve demos:",
    len(reserve),
)

print(
    "V2 D_CONFIRM demos excluded:",
    len(confirm),
)

print(
    "Current observation grid:",
    (
        f"freeze_end + {CURRENT_START_SEC}s, "
        f"stride {CURRENT_STRIDE_SEC}s"
    ),
)

print(
    "Target horizons:",
    TARGET_HORIZONS_SEC,
)


for demo_index, demo_path in enumerate(
    reserve,
    start=1,
):

    demo_filename = (
        demo_path.name
    )

    print()
    print(
        f"[{demo_index:02d}/13] "
        f"{demo_filename}"
    )


    # --------------------------------------------------------
    # Parse with frozen V0 timing + V3 semantic property
    # --------------------------------------------------------

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
        demo.header.get(
            "map_name"
        )
        == "de_mirage",
        (
            "Unexpected map in V3 reserve: "
            f"{demo_filename}"
        ),
    )


    require(
        "place"
        in demo.ticks.columns,
        (
            "Awpy normalized place column missing "
            f"in {demo_filename}"
        ),
    )


    ticks_by_round = (
        build_ticks_by_round(
            demo
        )
    )

    events_by_round = (
        build_events_by_round(
            demo
        )
    )


    # --------------------------------------------------------
    # Canonical plant semantics
    # --------------------------------------------------------

    (
        plant_lookup,
        plant_issues,
    ) = build_plant_lookup(
        demo
    )


    unresolved_plant_rounds = {
        int(
            issue[
                "round_num"
            ]
        )
        for issue in plant_issues
        if issue[
            "reason"
        ]
        == "PLANT_LABEL_UNRESOLVED"
    }


    # --------------------------------------------------------
    # Stage 1:
    # resolve only timing.
    #
    # We collect required snapshot keys first so that we do
    # not repeatedly scan millions of player rows.
    # --------------------------------------------------------

    timing_records = []

    required_snapshot_keys = set()


    for round_row in (
        demo.rounds
        .sort(
            "round_num"
        )
        .iter_rows(
            named=True
        )
    ):

        round_num = int(
            round_row[
                "round_num"
            ]
        )

        freeze_end = (
            round_row[
                "freeze_end"
            ]
        )

        round_end = (
            round_row[
                "end"
            ]
        )


        # ----------------------------------------------------
        # Frozen round-level exclusions
        # ----------------------------------------------------

        if freeze_end is None:

            round_exclusions.append({
                "demo_filename":
                    demo_filename,

                "round_num":
                    round_num,

                "reason":
                    "MISSING_FREEZE_END",
            })

            continue


        if round_end is None:

            round_exclusions.append({
                "demo_filename":
                    demo_filename,

                "round_num":
                    round_num,

                "reason":
                    "MISSING_ROUND_END",
            })

            continue


        freeze_end = int(
            freeze_end
        )

        round_end = int(
            round_end
        )


        if freeze_end >= round_end:

            round_exclusions.append({
                "demo_filename":
                    demo_filename,

                "round_num":
                    round_num,

                "reason":
                    "INVALID_TIMING_ORDER",
            })

            continue


        if (
            round_num
            in unresolved_plant_rounds
        ):

            round_exclusions.append({
                "demo_filename":
                    demo_filename,

                "round_num":
                    round_num,

                "reason":
                    "PLANT_LABEL_UNRESOLVED",
            })

            continue


        plant_info = (
            plant_lookup.get(
                round_num
            )
        )


        if plant_info is None:

            plant_tick = None
            plant_label = None

        else:

            plant_tick = int(
                plant_info[
                    "plant_tick"
                ]
            )

            plant_label = str(
                plant_info[
                    "label"
                ]
            )


        available_ticks = (
            ticks_by_round.get(
                round_num,
                [],
            )
        )


        # ----------------------------------------------------
        # Frozen current observation grid
        # ----------------------------------------------------

        current_nominal = (
            freeze_end
            + seconds_to_demo_ticks(
                CURRENT_START_SEC
            )
        )

        stride_ticks = (
            seconds_to_demo_ticks(
                CURRENT_STRIDE_SEC
            )
        )


        while (
            current_nominal
            < round_end
            and (
                plant_tick is None
                or current_nominal
                < plant_tick
            )
        ):

            current_resolution = (
                resolve_snapshot_tick(
                    available_ticks,
                    current_nominal,
                    upper_bound_exclusive=
                        round_end,
                )
            )


            # ----------------------------------------------
            # One audit row per horizon even when current
            # snapshot cannot be resolved.
            # ----------------------------------------------

            if current_resolution is None:

                for horizon_sec in (
                    TARGET_HORIZONS_SEC
                ):

                    timing_records.append({
                        "demo_filename":
                            demo_filename,

                        "round_num":
                            round_num,

                        "freeze_end":
                            freeze_end,

                        "round_end":
                            round_end,

                        "plant_tick":
                            plant_tick,

                        "plant_label":
                            plant_label,

                        "current_nominal_tick":
                            current_nominal,

                        "current_tick":
                            None,

                        "current_lateness":
                            None,

                        "horizon_sec":
                            horizon_sec,

                        "target_nominal_tick":
                            None,

                        "target_tick":
                            None,

                        "target_lateness":
                            None,

                        "timing_valid":
                            False,

                        "exclusion_reason":
                            "CURRENT_SNAPSHOT_UNAVAILABLE",
                    })

                current_nominal += (
                    stride_ticks
                )

                continue


            current_tick = (
                current_resolution[
                    "tick"
                ]
            )

            current_lateness = (
                current_resolution[
                    "lateness"
                ]
            )


            # ----------------------------------------------
            # A nominal pre-plant observation can resolve
            # +1 tick onto or after the actual plant.
            # Fail closed.
            # ----------------------------------------------

            if (
                plant_tick is not None
                and current_tick
                >= plant_tick
            ):

                for horizon_sec in (
                    TARGET_HORIZONS_SEC
                ):

                    timing_records.append({
                        "demo_filename":
                            demo_filename,

                        "round_num":
                            round_num,

                        "freeze_end":
                            freeze_end,

                        "round_end":
                            round_end,

                        "plant_tick":
                            plant_tick,

                        "plant_label":
                            plant_label,

                        "current_nominal_tick":
                            current_nominal,

                        "current_tick":
                            current_tick,

                        "current_lateness":
                            current_lateness,

                        "horizon_sec":
                            horizon_sec,

                        "target_nominal_tick":
                            None,

                        "target_tick":
                            None,

                        "target_lateness":
                            None,

                        "timing_valid":
                            False,

                        "exclusion_reason":
                            "CURRENT_STATE_AT_OR_AFTER_PLANT",
                    })

                current_nominal += (
                    stride_ticks
                )

                continue


            current_key = (
                round_num,
                current_tick,
            )

            required_snapshot_keys.add(
                current_key
            )


            # ----------------------------------------------
            # Future horizons
            # ----------------------------------------------

            for horizon_sec in (
                TARGET_HORIZONS_SEC
            ):

                target_nominal = (
                    current_tick
                    + seconds_to_demo_ticks(
                        horizon_sec
                    )
                )


                if target_nominal >= round_end:

                    timing_records.append({
                        "demo_filename":
                            demo_filename,

                        "round_num":
                            round_num,

                        "freeze_end":
                            freeze_end,

                        "round_end":
                            round_end,

                        "plant_tick":
                            plant_tick,

                        "plant_label":
                            plant_label,

                        "current_nominal_tick":
                            current_nominal,

                        "current_tick":
                            current_tick,

                        "current_lateness":
                            current_lateness,

                        "horizon_sec":
                            horizon_sec,

                        "target_nominal_tick":
                            target_nominal,

                        "target_tick":
                            None,

                        "target_lateness":
                            None,

                        "timing_valid":
                            False,

                        "exclusion_reason":
                            "TARGET_AFTER_ROUND_END",
                    })

                    continue


                target_resolution = (
                    resolve_snapshot_tick(
                        available_ticks,
                        target_nominal,
                        upper_bound_exclusive=
                            round_end,
                    )
                )


                if target_resolution is None:

                    timing_records.append({
                        "demo_filename":
                            demo_filename,

                        "round_num":
                            round_num,

                        "freeze_end":
                            freeze_end,

                        "round_end":
                            round_end,

                        "plant_tick":
                            plant_tick,

                        "plant_label":
                            plant_label,

                        "current_nominal_tick":
                            current_nominal,

                        "current_tick":
                            current_tick,

                        "current_lateness":
                            current_lateness,

                        "horizon_sec":
                            horizon_sec,

                        "target_nominal_tick":
                            target_nominal,

                        "target_tick":
                            None,

                        "target_lateness":
                            None,

                        "timing_valid":
                            False,

                        "exclusion_reason":
                            "TARGET_SNAPSHOT_UNAVAILABLE",
                    })

                    continue


                target_tick = (
                    target_resolution[
                        "tick"
                    ]
                )

                target_lateness = (
                    target_resolution[
                        "lateness"
                    ]
                )


                timing_records.append({
                    "demo_filename":
                        demo_filename,

                    "round_num":
                        round_num,

                    "freeze_end":
                        freeze_end,

                    "round_end":
                        round_end,

                    "plant_tick":
                        plant_tick,

                    "plant_label":
                        plant_label,

                    "current_nominal_tick":
                        current_nominal,

                    "current_tick":
                        current_tick,

                    "current_lateness":
                        current_lateness,

                    "horizon_sec":
                        horizon_sec,

                    "target_nominal_tick":
                        target_nominal,

                    "target_tick":
                        target_tick,

                    "target_lateness":
                        target_lateness,

                    "timing_valid":
                        True,

                    "exclusion_reason":
                        None,
                })


                required_snapshot_keys.add(
                    (
                        round_num,
                        target_tick,
                    )
                )


            current_nominal += (
                stride_ticks
            )


    # --------------------------------------------------------
    # Materialize only needed player snapshots
    # --------------------------------------------------------

    snapshots = (
        snapshot_rows_for_keys(
            demo,
            required_snapshot_keys,
        )
    )


    # --------------------------------------------------------
    # Stage 2:
    # semantic bomb-state resolution.
    # --------------------------------------------------------

    demo_valid = 0
    demo_excluded = 0


    for record in timing_records:

        base = dict(
            record
        )

        base.update({
            "current_source":
                None,

            "current_place":
                None,

            "current_macro_zone":
                None,

            "target_source":
                None,

            "target_place":
                None,

            "target_macro_zone":
                None,

            "stay_same_zone":
                None,

            "valid":
                False,
        })


        if not record[
            "timing_valid"
        ]:

            observation_rows.append(
                base
            )

            demo_excluded += 1
            continue


        round_num = int(
            record[
                "round_num"
            ]
        )

        current_tick = int(
            record[
                "current_tick"
            ]
        )

        target_tick = int(
            record[
                "target_tick"
            ]
        )


        current_snapshot = (
            snapshots.get(
                (
                    round_num,
                    current_tick,
                )
            )
        )

        target_snapshot = (
            snapshots.get(
                (
                    round_num,
                    target_tick,
                )
            )
        )


        if not current_snapshot:

            base[
                "exclusion_reason"
            ] = (
                "CURRENT_SNAPSHOT_UNAVAILABLE"
            )

            observation_rows.append(
                base
            )

            demo_excluded += 1
            continue


        if not target_snapshot:

            base[
                "exclusion_reason"
            ] = (
                "TARGET_SNAPSHOT_UNAVAILABLE"
            )

            observation_rows.append(
                base
            )

            demo_excluded += 1
            continue


        events = (
            events_by_round.get(
                round_num,
                [],
            )
        )


        # ----------------------------------------------------
        # Current bomb semantic
        # ----------------------------------------------------

        current_semantic = (
            resolve_unplanted_bomb_semantic(
                demo_filename=
                    demo_filename,

                round_num=
                    round_num,

                tick=
                    current_tick,

                snapshot=
                    current_snapshot,

                events=
                    events,

                place_error_reason=
                    "CURRENT_BOMB_PLACE_UNRESOLVED",

                state_error_reason=
                    "CURRENT_BOMB_STATE_UNRESOLVED",
            )
        )


        if not current_semantic[
            "ok"
        ]:

            base[
                "exclusion_reason"
            ] = current_semantic[
                "error"
            ]

            observation_rows.append(
                base
            )

            demo_excluded += 1
            continue


        base.update({
            "current_source":
                current_semantic[
                    "source"
                ],

            "current_place":
                current_semantic[
                    "place"
                ],

            "current_macro_zone":
                current_semantic[
                    "macro_zone"
                ],
        })


        # ----------------------------------------------------
        # Target semantic.
        #
        # Plant gets priority if the canonical valid plant
        # occurred by the resolved future target tick.
        # ----------------------------------------------------

        plant_tick = record[
            "plant_tick"
        ]

        plant_label = record[
            "plant_label"
        ]


        if (
            plant_tick is not None
            and int(
                plant_tick
            )
            <= target_tick
        ):

            if plant_label == "A_PLANT":

                target_semantic = {
                    "ok":
                        True,

                    "source":
                        "PLANTED",

                    "place":
                        "BombsiteA",

                    "macro_zone":
                        "A_SITE_COMPLEX",
                }

            elif plant_label == "B_PLANT":

                target_semantic = {
                    "ok":
                        True,

                    "source":
                        "PLANTED",

                    "place":
                        "BombsiteB",

                    "macro_zone":
                        "B_SITE_COMPLEX",
                }

            else:

                raise RuntimeError(
                    "Unexpected canonical plant label: "
                    f"{plant_label}"
                )


        else:

            target_semantic = (
                resolve_unplanted_bomb_semantic(
                    demo_filename=
                        demo_filename,

                    round_num=
                        round_num,

                    tick=
                        target_tick,

                    snapshot=
                        target_snapshot,

                    events=
                        events,

                    place_error_reason=
                        "TARGET_BOMB_PLACE_UNRESOLVED",

                    state_error_reason=
                        "TARGET_BOMB_STATE_UNRESOLVED",
                )
            )


        if not target_semantic[
            "ok"
        ]:

            base[
                "exclusion_reason"
            ] = target_semantic[
                "error"
            ]

            observation_rows.append(
                base
            )

            demo_excluded += 1
            continue


        base.update({
            "target_source":
                target_semantic[
                    "source"
                ],

            "target_place":
                target_semantic[
                    "place"
                ],

            "target_macro_zone":
                target_semantic[
                    "macro_zone"
                ],

            "stay_same_zone":
                (
                    current_semantic[
                        "macro_zone"
                    ]
                    ==
                    target_semantic[
                        "macro_zone"
                    ]
                ),

            "valid":
                True,

            "exclusion_reason":
                None,
        })


        require(
            base[
                "current_macro_zone"
            ]
            in macro_zones,
            "Unknown current macro-zone.",
        )

        require(
            base[
                "target_macro_zone"
            ]
            in macro_zones,
            "Unknown target macro-zone.",
        )


        observation_rows.append(
            base
        )

        demo_valid += 1


    print(
        "  candidate horizon rows:",
        len(
            timing_records
        ),
    )

    print(
        "  valid:",
        demo_valid,
    )

    print(
        "  excluded:",
        demo_excluded,
    )


# ============================================================
# Dataset-level invariants
# ============================================================

require(
    len(
        observation_rows
    ) > 0,
    "No target observations were generated.",
)


candidate_keys = [
    (
        row[
            "demo_filename"
        ],
        row[
            "round_num"
        ],
        row[
            "current_nominal_tick"
        ],
        row[
            "horizon_sec"
        ],
    )
    for row in observation_rows
]


require(
    len(
        candidate_keys
    )
    == len(
        set(
            candidate_keys
        )
    ),
    "Duplicate target-integrity candidate keys detected.",
)


allowed_observation_exclusions = {
    "CURRENT_SNAPSHOT_UNAVAILABLE",
    "CURRENT_STATE_AT_OR_AFTER_PLANT",
    "CURRENT_BOMB_STATE_UNRESOLVED",
    "CURRENT_BOMB_PLACE_UNRESOLVED",
    "TARGET_AFTER_ROUND_END",
    "TARGET_SNAPSHOT_UNAVAILABLE",
    "TARGET_BOMB_STATE_UNRESOLVED",
    "TARGET_BOMB_PLACE_UNRESOLVED",
}


observed_exclusions = {
    row[
        "exclusion_reason"
    ]
    for row in observation_rows
    if row[
        "exclusion_reason"
    ]
    is not None
}


unexpected_exclusions = (
    observed_exclusions
    - allowed_observation_exclusions
)


require(
    not unexpected_exclusions,
    (
        "Unexpected exclusion reasons: "
        f"{sorted(unexpected_exclusions)}"
    ),
)


# ============================================================
# Save audit artifacts
# ============================================================

observations = pl.DataFrame(
    observation_rows,
    infer_schema_length=None,
)

OBS_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

observations.write_csv(
    OBS_OUTPUT
)


if round_exclusions:

    round_df = pl.DataFrame(
        round_exclusions,
        infer_schema_length=None,
    )

else:

    round_df = pl.DataFrame(
        schema={
            "demo_filename":
                pl.String,

            "round_num":
                pl.Int64,

            "reason":
                pl.String,
        }
    )


round_df.write_csv(
    ROUND_OUTPUT
)


# ============================================================
# Global summary
# ============================================================

total_rows = (
    observations.height
)

valid_rows = (
    observations
    .filter(
        pl.col("valid")
        == True
    )
)

n_valid = (
    valid_rows.height
)

n_excluded = (
    total_rows
    - n_valid
)

coverage = (
    n_valid / total_rows
)


print()
print("=" * 104)
print("GLOBAL TARGET INTEGRITY SUMMARY")
print("=" * 104)

print(
    "Candidate horizon rows:",
    f"{total_rows:,}",
)

print(
    "Valid:",
    f"{n_valid:,}",
)

print(
    "Excluded:",
    f"{n_excluded:,}",
)

print(
    "Integrity coverage:",
    f"{coverage:.4%}",
)


print()
print("VALID BY HORIZON")
print("-" * 104)

valid_by_horizon = (
    observations
    .group_by(
        "horizon_sec"
    )
    .agg([
        pl.len().alias(
            "candidate_rows"
        ),

        pl.col(
            "valid"
        )
        .sum()
        .alias(
            "valid_rows"
        ),
    ])
    .with_columns(
        (
            pl.col(
                "valid_rows"
            )
            /
            pl.col(
                "candidate_rows"
            )
        ).alias(
            "coverage"
        )
    )
    .sort(
        "horizon_sec"
    )
)

print(
    valid_by_horizon
)


print()
print("EXCLUSION REASONS")
print("-" * 104)

exclusion_counts = (
    observations
    .filter(
        pl.col(
            "valid"
        )
        == False
    )
    .group_by([
        "horizon_sec",
        "exclusion_reason",
    ])
    .len()
    .sort([
        "horizon_sec",
        "len",
    ], descending=[
        False,
        True,
    ])
)

print(
    exclusion_counts
)


print()
print("TARGET SOURCE COUNTS")
print("-" * 104)

target_source_counts = (
    valid_rows
    .group_by([
        "horizon_sec",
        "target_source",
    ])
    .len()
    .sort([
        "horizon_sec",
        "len",
    ], descending=[
        False,
        True,
    ])
)

print(
    target_source_counts
)


print()
print("TARGET MACRO-ZONE COUNTS")
print("-" * 104)

target_zone_counts = (
    valid_rows
    .group_by([
        "horizon_sec",
        "target_macro_zone",
    ])
    .len()
    .sort([
        "horizon_sec",
        "len",
    ], descending=[
        False,
        True,
    ])
)

print(
    target_zone_counts
)


print()
print("STAY VS MOVE")
print("-" * 104)

stay_move = (
    valid_rows
    .group_by([
        "horizon_sec",
        "stay_same_zone",
    ])
    .len()
    .sort([
        "horizon_sec",
        "stay_same_zone",
    ])
)

print(
    stay_move
)


print()
print("CURRENT SOURCE COUNTS")
print("-" * 104)

current_sources = (
    valid_rows
    .group_by(
        "current_source"
    )
    .len()
    .sort(
        "len",
        descending=True,
    )
)

print(
    current_sources
)


print()
print("ROUND EXCLUSIONS")
print("-" * 104)

if round_df.height:

    print(
        round_df
        .group_by(
            "reason"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

else:

    print(
        "None"
    )


# ============================================================
# Machine-readable summary
# ============================================================

summary = {
    "version":
        "V3",

    "gate":
        "2D",

    "status":
        "AUDIT_COMPLETE",

    "training_performed":
        False,

    "v2_d_confirm_used":
        False,

    "candidate_horizon_rows":
        total_rows,

    "valid_rows":
        n_valid,

    "excluded_rows":
        n_excluded,

    "integrity_coverage":
        coverage,

    "n_round_exclusions":
        round_df.height,

    "observation_output":
        str(
            OBS_OUTPUT
        ),

    "round_exclusion_output":
        str(
            ROUND_OUTPUT
        ),

    "contract":
        str(
            CONTRACT
        ),

    "notes": [
        (
            "Coverage is an audit result, not a "
            "post-hoc optimization target."
        ),
        (
            "No target labels use the XYZ classifier."
        ),
        (
            "No model training or hyperparameter "
            "selection occurred."
        ),
        (
            "The 13 reserve demos are V3 development "
            "evidence only."
        ),
    ],
}


SUMMARY_OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 104)

print(
    "✅ V3 GATE 2D TARGET INTEGRITY TABLE BUILT"
)

print(
    "NEXT_ACTION=INTERPRET_TARGET_DISTRIBUTION_AND_EXCLUSIONS"
)

print("=" * 104)

print()
print("Artifacts:")
print(" ", OBS_OUTPUT)
print(" ", ROUND_OUTPUT)
print(" ", SUMMARY_OUTPUT)
