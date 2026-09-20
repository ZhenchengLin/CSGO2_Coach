"""Frozen V3 target semantics reused for V4-A confirmation.

The functions below are copied verbatim from the frozen V3
confirmation target builder. Do not import the V3 builder directly:
its module-level code executes the historical dataset pipeline.

This module provides semantic helpers only. It does not select
confirmation matches, produce target rows, load models, or score.
"""

from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import has_c4

MAX_LATENESS = 1

EXPECTED_MAPPING_SHA256 = "557bd091789e43e463bdd85251cd7d581a912723f38b5989ce6045e76692d9b3"
EXPECTED_TARGET_CONTRACT_SHA256 = "ae51a6149086b95c71ce8a7a58c88d3993bce845fdad00f438ff1791c0e6943e"

_MAPPING_PATH = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)
_TARGET_CONTRACT_PATH = Path(
    "docs/v3_future_target_contract.json"
)

for _path, _expected in [
    (_MAPPING_PATH, EXPECTED_MAPPING_SHA256),
    (_TARGET_CONTRACT_PATH, EXPECTED_TARGET_CONTRACT_SHA256),
]:
    _actual = hashlib.sha256(_path.read_bytes()).hexdigest()
    if _actual != _expected:
        raise RuntimeError(
            f"Frozen target dependency changed: {_path}"
        )

_mapping = json.loads(_MAPPING_PATH.read_text())

if _mapping["status"] != "FROZEN":
    raise RuntimeError("Macro-zone mapping is not frozen.")

place_to_zone = {}

for _zone, _places in _mapping["zones"].items():
    for _place in _places:
        if _place in place_to_zone:
            raise RuntimeError(
                f"Duplicate fine semantic place: {_place}"
            )
        place_to_zone[_place] = _zone

if len(place_to_zone) != 23:
    raise RuntimeError(
        "Expected 23 frozen fine semantic places."
    )




def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
