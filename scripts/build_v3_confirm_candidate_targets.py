from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import gc
import hashlib
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


# ============================================================
# Frozen confirmation-candidate inputs
# ============================================================

QUEUE = Path(
    "docs/v3_confirm_acquisition_queue_v1.csv"
)

DEV_MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

V2_CONFIRM_MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

TARGET_CONTRACT = Path(
    "docs/v3_future_target_contract.json"
)

MACRO_MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

CONFIRM_RAW = Path(
    "data/raw/v3_confirm_incoming"
)


EXPECTED_MATCHES = 1

HORIZONS = [
    5,
    10,
]


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


parser = argparse.ArgumentParser(
    description=(
        "Build frozen V3 target semantics for exactly one "
        "confirmation candidate. No model prediction or "
        "confirmation metric is calculated."
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


match_date = str(
    row["date"]
)

team1 = str(
    row["team1"]
)

team2 = str(
    row["team2"]
)


demo_filename = (
    f"{match_date}_"
    f"{team1}_vs_"
    f"{team2}_mirage.dem"
)

demo_path = (
    CONFIRM_RAW
    / demo_filename
)


require(
    demo_path.exists(),
    f"Missing confirmation demo: {demo_path}",
)


demo_sha = sha256_file(
    demo_path
)


dev_manifest = pl.read_csv(
    DEV_MANIFEST,
    infer_schema_length=None,
)

v2_confirm_manifest = pl.read_csv(
    V2_CONFIRM_MANIFEST,
    infer_schema_length=None,
)


require(
    demo_sha
    not in set(
        dev_manifest["sha256"].to_list()
    ),
    "Confirmation candidate overlaps D_V3_DEV SHA.",
)

require(
    demo_sha
    not in set(
        v2_confirm_manifest["sha256"].to_list()
    ),
    "Confirmation candidate overlaps V2 D_CONFIRM SHA.",
)


# One-row compatibility manifest.
#
# Only the fields consumed by the frozen generalized target
# builder carry scientific meaning here. This temporary
# in-memory manifest is NOT D_V3_CONFIRM itself.

manifest = pl.DataFrame({
    "dev_rank": [
        int(
            args.candidate_rank
        )
    ],

    "demo_filename": [
        demo_filename
    ],

    "match_date": [
        match_date
    ],

    "dev_source_role": [
        "V3_CONFIRM_INTAKE"
    ],

    "prior_scientific_role": [
        "UNTOUCHED_CONFIRMATION_CANDIDATE"
    ],

    "path": [
        str(
            demo_path
        )
    ],

    "sha256": [
        demo_sha
    ],

    "map_name": [
        "PENDING_TECHNICAL_AUDIT"
    ],

    "raw_ticks_per_sec": [
        64
    ],

    "validation_status": [
        "PRE_MANIFEST_TECHNICAL_AUDIT"
    ],
})


OUT_DIR = Path(
    "data/interim/v3_confirm_intake"
) / (
    f"rank_{args.candidate_rank:02d}"
)

DATASET_OUTPUT = (
    OUT_DIR
    / "targets.parquet"
)

EXCLUSION_OUTPUT = (
    OUT_DIR
    / "target_exclusions.csv"
)

ROUND_EXCLUSION_OUTPUT = (
    OUT_DIR
    / "round_exclusions.csv"
)

DROP_AUDIT_OUTPUT = (
    OUT_DIR
    / "drop_semantic_audit.csv"
)

RESULT_OUTPUT = (
    OUT_DIR
    / "target_build_summary.json"
)


# ============================================================
# Frozen target contract
# ============================================================

contract = json.loads(
    TARGET_CONTRACT.read_text()
)

require(
    contract[
        "status"
    ] == "FROZEN",
    "Target contract is not frozen.",
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

TARGET_HORIZONS = list(
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
    TARGET_HORIZONS == HORIZONS,
    (
        "Frozen horizons changed: "
        f"{TARGET_HORIZONS}"
    ),
)

require(
    V0_DEMO_TICKS_PER_SECOND
    == 64,
    "Expected frozen 64 tick/sec historical clock.",
)


# ============================================================
# Frozen macro-zone representation
# ============================================================

mapping_record = json.loads(
    MACRO_MAPPING.read_text()
)

require(
    mapping_record[
        "status"
    ] == "FROZEN",
    "Macro-zone mapping is not frozen.",
)


class_order = list(
    mapping_record[
        "zones"
    ].keys()
)


require(
    len(
        class_order
    )
    == 15,
    (
        "Expected 15 frozen macro-zones, "
        f"found {len(class_order)}."
    ),
)


class_to_index = {
    zone:
        index
    for index, zone
    in enumerate(
        class_order
    )
}


place_to_zone = {}


for zone, places in (
    mapping_record[
        "zones"
    ].items()
):

    for place in places:

        require(
            place
            not in place_to_zone,
            (
                "Fine semantic place appears "
                f"in multiple zones: {place}"
            ),
        )

        place_to_zone[
            place
        ] = zone


require(
    len(
        place_to_zone
    )
    == 23,
    (
        "Expected 23 Valve fine places, "
        f"found {len(place_to_zone)}."
    ),
)


# ============================================================
# Timing helpers
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

        result.setdefault(
            int(
                row[
                    "round_num"
                ]
            ),
            [],
        ).append(
            int(
                row[
                    "tick"
                ]
            )
        )

    return result


def resolve_snapshot_tick(
    available_ticks,
    nominal_tick,
    *,
    upper_bound_exclusive=None,
):
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


# ============================================================
# Bomb-event helpers
# ============================================================

def build_events_by_round(demo):
    result = {}

    for row in (
        demo.bomb
        .sort([
            "round_num",
            "tick",
        ])
        .iter_rows(
            named=True
        )
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


# ============================================================
# Snapshot materialization
# ============================================================

def materialize_snapshots(
    demo,
    required_keys,
):
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


# ============================================================
# Semantic mapping
# ============================================================

def map_place(
    place,
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

    zone = place_to_zone.get(
        place
    )

    if zone is None:

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
            zone,
    }


# ============================================================
# Exact-tick dropped-bomb semantic audit
# ============================================================

def build_drop_lookup(
    demo,
    demo_filename,
):
    lookup = {}

    audit_rows = []

    drops = (
        demo.bomb
        .filter(
            pl.col("event")
            == "drop"
        )
        .sort([
            "round_num",
            "tick",
        ])
    )


    for drop in drops.iter_rows(
        named=True
    ):

        round_num = int(
            drop[
                "round_num"
            ]
        )

        drop_tick = int(
            drop[
                "tick"
            ]
        )

        steamid = (
            drop[
                "steamid"
            ]
        )


        audit = {
            "demo_filename":
                demo_filename,

            "round_num":
                round_num,

            "drop_tick":
                drop_tick,

            "steamid":
                steamid,

            "resolved":
                False,

            "place":
                None,

            "macro_zone":
                None,

            "reason":
                None,
        }


        if steamid is None:

            audit[
                "reason"
            ] = "DROP_MISSING_STEAMID"

            audit_rows.append(
                audit
            )

            continue


        candidates = (
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
                    == drop_tick
                )
                &
                (
                    pl.col(
                        "steamid"
                    )
                    == steamid
                )
            )
        )


        if candidates.height != 1:

            audit[
                "reason"
            ] = (
                "DROP_EXACT_TICK_PLAYER_UNRESOLVED"
            )

            audit_rows.append(
                audit
            )

            continue


        player = candidates.row(
            0,
            named=True,
        )

        mapped = map_place(
            player[
                "place"
            ],
            "DROP_PLACE_UNRESOLVED",
        )


        if not mapped[
            "ok"
        ]:

            audit[
                "reason"
            ] = mapped[
                "error"
            ]

            audit_rows.append(
                audit
            )

            continue


        key = (
            round_num,
            drop_tick,
        )


        require(
            key not in lookup,
            (
                "Duplicate drop-event semantic key: "
                f"{demo_filename} {key}"
            ),
        )


        lookup[
            key
        ] = {
            "place":
                mapped[
                    "place"
                ],

            "macro_zone":
                mapped[
                    "macro_zone"
                ],

            "X":
                float(
                    drop[
                        "X"
                    ]
                ),

            "Y":
                float(
                    drop[
                        "Y"
                    ]
                ),

            "Z":
                float(
                    drop[
                        "Z"
                    ]
                ),
        }


        audit.update({
            "resolved":
                True,

            "place":
                mapped[
                    "place"
                ],

            "macro_zone":
                mapped[
                    "macro_zone"
                ],

            "reason":
                "RESOLVED",
        })


        audit_rows.append(
            audit
        )


    return lookup, audit_rows


# ============================================================
# Resolve current/future unplanted bomb state
# ============================================================

def resolve_unplanted_bomb(
    *,
    round_num,
    tick,
    snapshot,
    events,
    drop_lookup,
    state_error,
    place_error,
):
    carriers = [
        player
        for player in snapshot
        if (
            player[
                "side"
            ] == "t"
            and has_c4(
                player[
                    "inventory"
                ]
            )
        )
    ]


    if len(
        carriers
    ) > 1:

        raise RuntimeError(
            "Invariant violation: multiple C4 carriers "
            f"round={round_num} tick={tick}"
        )


    # --------------------------------------------------------
    # Inventory carrier
    # --------------------------------------------------------

    if len(
        carriers
    ) == 1:

        carrier = carriers[
            0
        ]

        mapped = map_place(
            carrier[
                "place"
            ],
            place_error,
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

            "X":
                float(
                    carrier[
                        "X"
                    ]
                ),

            "Y":
                float(
                    carrier[
                        "Y"
                    ]
                ),

            "Z":
                float(
                    carrier[
                        "Z"
                    ]
                ),
        }


    # --------------------------------------------------------
    # Latest causal bomb event
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
                state_error,
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

        steamid = (
            event[
                "steamid"
            ]
        )


        if steamid is None:

            return {
                "ok":
                    False,

                "error":
                    state_error,
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
                == steamid
            )
        ]


        if len(
            matching
        ) != 1:

            return {
                "ok":
                    False,

                "error":
                    state_error,
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
                    state_error,
            }


        mapped = map_place(
            carrier[
                "place"
            ],
            place_error,
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

            "X":
                float(
                    carrier[
                        "X"
                    ]
                ),

            "Y":
                float(
                    carrier[
                        "Y"
                    ]
                ),

            "Z":
                float(
                    carrier[
                        "Z"
                    ]
                ),
        }


    # --------------------------------------------------------
    # Dropped bomb
    # --------------------------------------------------------

    if event_type == "drop":

        key = (
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
                key
            )
        )


        if recovered is None:

            return {
                "ok":
                    False,

                "error":
                    place_error,
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

            "X":
                recovered[
                    "X"
                ],

            "Y":
                recovered[
                    "Y"
                ],

            "Z":
                recovered[
                    "Z"
                ],
        }


    return {
        "ok":
            False,

        "error":
            state_error,
    }


# ============================================================
# Global accumulators
# ============================================================

dataset_rows = []
exclusion_rows = []
round_exclusion_rows = []
drop_audit_rows = []


print("=" * 108)
print("V3 GATE 3C — BUILD D_V3_DEV TARGET DATASET")
print("=" * 108)

print()
print(
    "Development matches:",
    manifest.height,
)

print(
    "Current observation start:",
    f"freeze_end + {CURRENT_START_SEC}s",
)

print(
    "Stride:",
    f"{CURRENT_STRIDE_SEC}s",
)

print(
    "Horizons:",
    TARGET_HORIZONS,
)

print(
    "Raw clock:",
    f"{V0_DEMO_TICKS_PER_SECOND} ticks/sec",
)

print(
    "No model training:",
    True,
)


# ============================================================
# Process one demo at a time
# ============================================================

for demo_index, manifest_row in enumerate(
    manifest
    .sort(
        "dev_rank"
    )
    .iter_rows(
        named=True
    ),
    start=1,
):

    demo_filename = str(
        manifest_row[
            "demo_filename"
        ]
    )

    demo_path = Path(
        manifest_row[
            "path"
        ]
    )

    match_date = str(
        manifest_row[
            "match_date"
        ]
    )

    dev_source_role = str(
        manifest_row[
            "dev_source_role"
        ]
    )


    print()
    print(
        f"[{demo_index:02d}/01]",
        dev_source_role,
        demo_filename,
    )


    require(
        demo_path.exists(),
        f"Missing demo: {demo_path}",
    )


    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )


    require(
        demo.header.get(
            "map_name"
        )
        == "de_mirage",
        (
            "Unexpected map during target build: "
            f"{demo_filename}"
        ),
    )


    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )


    require(
        "place"
        in demo.ticks.columns,
        (
            "Awpy normalized semantic column "
            f"'place' missing in {demo_filename}."
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
    # Canonical plant lookup
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
    # Exact-tick drop semantics
    # --------------------------------------------------------

    (
        drop_lookup,
        demo_drop_audit,
    ) = build_drop_lookup(
        demo,
        demo_filename,
    )


    drop_audit_rows.extend(
        demo_drop_audit
    )


    drop_total = len(
        demo_drop_audit
    )

    drop_resolved = sum(
        1
        for row in demo_drop_audit
        if row[
            "resolved"
        ]
    )


    # --------------------------------------------------------
    # Stage 1:
    # candidate timing resolution
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


        if freeze_end is None:

            round_exclusion_rows.append({
                "demo_filename":
                    demo_filename,

                "round_num":
                    round_num,

                "reason":
                    "MISSING_FREEZE_END",
            })

            continue


        if round_end is None:

            round_exclusion_rows.append({
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

            round_exclusion_rows.append({
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

            round_exclusion_rows.append({
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


            if current_resolution is None:

                for horizon in (
                    TARGET_HORIZONS
                ):

                    exclusion_rows.append({
                        "demo_filename":
                            demo_filename,

                        "match_date":
                            match_date,

                        "dev_source_role":
                            dev_source_role,

                        "round_num":
                            round_num,

                        "current_nominal_tick":
                            current_nominal,

                        "horizon_sec":
                            horizon,

                        "reason":
                            "CURRENT_SNAPSHOT_UNAVAILABLE",
                    })

                current_nominal += (
                    stride_ticks
                )

                continue


            current_tick = int(
                current_resolution[
                    "tick"
                ]
            )

            current_lateness = int(
                current_resolution[
                    "lateness"
                ]
            )


            if (
                plant_tick is not None
                and current_tick
                >= plant_tick
            ):

                for horizon in (
                    TARGET_HORIZONS
                ):

                    exclusion_rows.append({
                        "demo_filename":
                            demo_filename,

                        "match_date":
                            match_date,

                        "dev_source_role":
                            dev_source_role,

                        "round_num":
                            round_num,

                        "current_nominal_tick":
                            current_nominal,

                        "horizon_sec":
                            horizon,

                        "reason":
                            "CURRENT_STATE_AT_OR_AFTER_PLANT",
                    })

                current_nominal += (
                    stride_ticks
                )

                continue


            required_snapshot_keys.add(
                (
                    round_num,
                    current_tick,
                )
            )


            for horizon in (
                TARGET_HORIZONS
            ):

                target_nominal = (
                    current_tick
                    + seconds_to_demo_ticks(
                        horizon
                    )
                )


                if target_nominal >= round_end:

                    exclusion_rows.append({
                        "demo_filename":
                            demo_filename,

                        "match_date":
                            match_date,

                        "dev_source_role":
                            dev_source_role,

                        "round_num":
                            round_num,

                        "current_nominal_tick":
                            current_nominal,

                        "horizon_sec":
                            horizon,

                        "reason":
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

                    exclusion_rows.append({
                        "demo_filename":
                            demo_filename,

                        "match_date":
                            match_date,

                        "dev_source_role":
                            dev_source_role,

                        "round_num":
                            round_num,

                        "current_nominal_tick":
                            current_nominal,

                        "horizon_sec":
                            horizon,

                        "reason":
                            "TARGET_SNAPSHOT_UNAVAILABLE",
                    })

                    continue


                target_tick = int(
                    target_resolution[
                        "tick"
                    ]
                )

                target_lateness = int(
                    target_resolution[
                        "lateness"
                    ]
                )


                required_snapshot_keys.add(
                    (
                        round_num,
                        target_tick,
                    )
                )


                timing_records.append({
                    "demo_filename":
                        demo_filename,

                    "match_date":
                        match_date,

                    "dev_source_role":
                        dev_source_role,

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
                        horizon,

                    "target_nominal_tick":
                        target_nominal,

                    "target_tick":
                        target_tick,

                    "target_lateness":
                        target_lateness,
                })


            current_nominal += (
                stride_ticks
            )


    # --------------------------------------------------------
    # Only materialize player snapshots we actually need
    # --------------------------------------------------------

    snapshots = materialize_snapshots(
        demo,
        required_snapshot_keys,
    )


    demo_valid = 0
    demo_semantic_excluded = 0


    # --------------------------------------------------------
    # Stage 2:
    # resolve bomb semantics
    # --------------------------------------------------------

    for record in timing_records:

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

            exclusion_rows.append({
                "demo_filename":
                    demo_filename,

                "match_date":
                    match_date,

                "dev_source_role":
                    dev_source_role,

                "round_num":
                    round_num,

                "current_nominal_tick":
                    record[
                        "current_nominal_tick"
                    ],

                "horizon_sec":
                    record[
                        "horizon_sec"
                    ],

                "reason":
                    "CURRENT_SNAPSHOT_UNAVAILABLE",
            })

            demo_semantic_excluded += 1
            continue


        if not target_snapshot:

            exclusion_rows.append({
                "demo_filename":
                    demo_filename,

                "match_date":
                    match_date,

                "dev_source_role":
                    dev_source_role,

                "round_num":
                    round_num,

                "current_nominal_tick":
                    record[
                        "current_nominal_tick"
                    ],

                "horizon_sec":
                    record[
                        "horizon_sec"
                    ],

                "reason":
                    "TARGET_SNAPSHOT_UNAVAILABLE",
            })

            demo_semantic_excluded += 1
            continue


        events = events_by_round.get(
            round_num,
            [],
        )


        # ----------------------------------------------------
        # Current bomb state
        # ----------------------------------------------------

        current_state = (
            resolve_unplanted_bomb(
                round_num=
                    round_num,

                tick=
                    current_tick,

                snapshot=
                    current_snapshot,

                events=
                    events,

                drop_lookup=
                    drop_lookup,

                state_error=
                    "CURRENT_BOMB_STATE_UNRESOLVED",

                place_error=
                    "CURRENT_BOMB_PLACE_UNRESOLVED",
            )
        )


        if not current_state[
            "ok"
        ]:

            exclusion_rows.append({
                "demo_filename":
                    demo_filename,

                "match_date":
                    match_date,

                "dev_source_role":
                    dev_source_role,

                "round_num":
                    round_num,

                "current_nominal_tick":
                    record[
                        "current_nominal_tick"
                    ],

                "horizon_sec":
                    record[
                        "horizon_sec"
                    ],

                "reason":
                    current_state[
                        "error"
                    ],
            })

            demo_semantic_excluded += 1
            continue


        # ----------------------------------------------------
        # Target bomb state
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

                target_state = {
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

                target_state = {
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

            target_state = (
                resolve_unplanted_bomb(
                    round_num=
                        round_num,

                    tick=
                        target_tick,

                    snapshot=
                        target_snapshot,

                    events=
                        events,

                    drop_lookup=
                        drop_lookup,

                    state_error=
                        "TARGET_BOMB_STATE_UNRESOLVED",

                    place_error=
                        "TARGET_BOMB_PLACE_UNRESOLVED",
                )
            )


        if not target_state[
            "ok"
        ]:

            exclusion_rows.append({
                "demo_filename":
                    demo_filename,

                "match_date":
                    match_date,

                "dev_source_role":
                    dev_source_role,

                "round_num":
                    round_num,

                "current_nominal_tick":
                    record[
                        "current_nominal_tick"
                    ],

                "horizon_sec":
                    record[
                        "horizon_sec"
                    ],

                "reason":
                    target_state[
                        "error"
                    ],
            })

            demo_semantic_excluded += 1
            continue


        current_zone = (
            current_state[
                "macro_zone"
            ]
        )

        target_zone = (
            target_state[
                "macro_zone"
            ]
        )


        require(
            current_zone
            in class_to_index,
            (
                "Unknown current zone: "
                f"{current_zone}"
            ),
        )

        require(
            target_zone
            in class_to_index,
            (
                "Unknown target zone: "
                f"{target_zone}"
            ),
        )


        dataset_rows.append({
            "demo_filename":
                demo_filename,

            "match_date":
                match_date,

            "dev_source_role":
                dev_source_role,

            "round_num":
                round_num,

            "current_nominal_tick":
                int(
                    record[
                        "current_nominal_tick"
                    ]
                ),

            "current_tick":
                current_tick,

            "current_lateness":
                int(
                    record[
                        "current_lateness"
                    ]
                ),

            "horizon_sec":
                int(
                    record[
                        "horizon_sec"
                    ]
                ),

            "target_nominal_tick":
                int(
                    record[
                        "target_nominal_tick"
                    ]
                ),

            "target_tick":
                target_tick,

            "target_lateness":
                int(
                    record[
                        "target_lateness"
                    ]
                ),

            "current_source":
                current_state[
                    "source"
                ],

            "current_place":
                current_state[
                    "place"
                ],

            "current_macro_zone":
                current_zone,

            "current_class_index":
                int(
                    class_to_index[
                        current_zone
                    ]
                ),

            "current_bomb_X":
                float(
                    current_state[
                        "X"
                    ]
                ),

            "current_bomb_Y":
                float(
                    current_state[
                        "Y"
                    ]
                ),

            "current_bomb_Z":
                float(
                    current_state[
                        "Z"
                    ]
                ),

            "target_source":
                target_state[
                    "source"
                ],

            "target_place":
                target_state[
                    "place"
                ],

            "target_macro_zone":
                target_zone,

            "target_class_index":
                int(
                    class_to_index[
                        target_zone
                    ]
                ),

            "stay_same_zone":
                (
                    current_zone
                    == target_zone
                ),
        })


        demo_valid += 1


    print(
        "  valid target rows:",
        demo_valid,
    )

    print(
        "  semantic exclusions:",
        demo_semantic_excluded,
    )

    print(
        "  drop semantics:",
        f"{drop_resolved}/{drop_total}",
    )


    # Release one parsed demo before loading next.
    del snapshots
    del demo

    gc.collect()


# ============================================================
# Convert datasets
# ============================================================

require(
    len(
        dataset_rows
    ) > 0,
    "No valid target rows generated.",
)


dataset = pl.DataFrame(
    dataset_rows,
    infer_schema_length=None,
)


if exclusion_rows:

    exclusions = pl.DataFrame(
        exclusion_rows,
        infer_schema_length=None,
    )

else:

    exclusions = pl.DataFrame(
        schema={
            "demo_filename":
                pl.String,

            "match_date":
                pl.String,

            "dev_source_role":
                pl.String,

            "round_num":
                pl.Int64,

            "current_nominal_tick":
                pl.Int64,

            "horizon_sec":
                pl.Int64,

            "reason":
                pl.String,
        }
    )


if round_exclusion_rows:

    round_exclusions = pl.DataFrame(
        round_exclusion_rows,
        infer_schema_length=None,
    )

else:

    round_exclusions = pl.DataFrame(
        schema={
            "demo_filename":
                pl.String,

            "round_num":
                pl.Int64,

            "reason":
                pl.String,
        }
    )


if drop_audit_rows:

    drop_audit = pl.DataFrame(
        drop_audit_rows,
        infer_schema_length=None,
    )

else:

    drop_audit = pl.DataFrame(
        schema={
            "demo_filename":
                pl.String,

            "round_num":
                pl.Int64,

            "drop_tick":
                pl.Int64,

            "steamid":
                pl.UInt64,

            "resolved":
                pl.Boolean,

            "place":
                pl.String,

            "macro_zone":
                pl.String,

            "reason":
                pl.String,
        }
    )


# ============================================================
# Dataset invariants
# ============================================================

duplicate_keys = (
    dataset
    .group_by([
        "demo_filename",
        "round_num",
        "current_nominal_tick",
        "horizon_sec",
    ])
    .len()
    .filter(
        pl.col("len")
        != 1
    )
)


require(
    duplicate_keys.height == 0,
    "Duplicate V3 target rows detected.",
)


require(
    dataset[
        "demo_filename"
    ].n_unique()
    in {
        0,
        1,
    },
    "Unexpected number of candidate demos in target rows.",
)


observed_horizons = set(
    dataset[
        "horizon_sec"
    ].unique().to_list()
)


require(
    observed_horizons.issubset(
        {
            5,
            10,
        }
    ),
    "Unexpected horizon values.",
)


total_nulls = sum(
    dataset.null_count().row(0)
)

require(
    total_nulls == 0,
    (
        "Valid target dataset contains null values: "
        f"{total_nulls}"
    ),
)


# ============================================================
# Write compressed artifacts
# ============================================================

DATASET_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

EXCLUSION_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


dataset.write_parquet(
    DATASET_OUTPUT,
    compression="zstd",
)


exclusions.write_csv(
    EXCLUSION_OUTPUT
)


round_exclusions.write_csv(
    ROUND_EXCLUSION_OUTPUT
)


drop_audit.write_csv(
    DROP_AUDIT_OUTPUT
)


# ============================================================
# Technical-only confirmation intake summary
# ============================================================

valid_by_horizon = {
    str(horizon):
        int(
            dataset
            .filter(
                pl.col("horizon_sec")
                == horizon
            )
            .height
        )
    for horizon in HORIZONS
}


drop_total = int(
    drop_audit.height
)

drop_resolved = int(
    drop_audit
    .filter(
        pl.col("resolved")
        == True
    )
    .height
)

drop_unresolved = (
    drop_total
    - drop_resolved
)


summary = {
    "candidate_rank":
        int(
            args.candidate_rank
        ),

    "demo_filename":
        demo_filename,

    "demo_path":
        str(
            demo_path
        ),

    "demo_sha256":
        demo_sha,

    "target_contract":
        str(
            TARGET_CONTRACT
        ),

    "target_contract_sha256":
        sha256_file(
            TARGET_CONTRACT
        ),

    "macro_mapping":
        str(
            MACRO_MAPPING
        ),

    "macro_mapping_sha256":
        sha256_file(
            MACRO_MAPPING
        ),

    "valid_rows_by_horizon":
        valid_by_horizon,

    "drop_semantics": {
        "events":
            drop_total,

        "resolved_exact_tick":
            drop_resolved,

        "unresolved":
            drop_unresolved,
    },

    "artifacts": {
        "targets":
            str(
                DATASET_OUTPUT
            ),

        "target_exclusions":
            str(
                EXCLUSION_OUTPUT
            ),

        "round_exclusions":
            str(
                ROUND_EXCLUSION_OUTPUT
            ),

        "drop_semantic_audit":
            str(
                DROP_AUDIT_OUTPUT
            ),
    },

    "model_training_performed":
        False,

    "model_prediction_performed":
        False,

    "confirmation_metric_calculated":
        False,

    "class_distribution_reported":
        False,

    "persistence_distribution_reported":
        False,
}


RESULT_OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 96)
print("V3 CONFIRM CANDIDATE TARGET BUILD")
print("=" * 96)

print(
    "candidate rank:",
    args.candidate_rank,
)

print(
    "demo:",
    demo_filename,
)

print(
    "+5 valid rows:",
    valid_by_horizon["5"],
)

print(
    "+10 valid rows:",
    valid_by_horizon["10"],
)

print(
    "drop semantic unresolved:",
    drop_unresolved,
)

print("model prediction: NO")
print("confirmation metric: NO")
print("class distribution reported: NO")

print()
print(
    "V3_CONFIRM_CANDIDATE_TARGET_BUILD_COMPLETE"
)
