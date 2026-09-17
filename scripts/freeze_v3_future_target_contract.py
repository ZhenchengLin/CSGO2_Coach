from __future__ import annotations

import hashlib
import json
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

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

DROP_AUDIT = Path(
    "data/interim/v3_drop_semantic_recovery_audit.csv"
)

OUTPUT = Path(
    "docs/v3_future_target_contract.json"
)


EXPECTED_RESERVE_DEMOS = 13
EXPECTED_DROP_EVENTS = 657

CURRENT_START_SEC = 5
CURRENT_STRIDE_SEC = 5

TARGET_HORIZONS_SEC = [
    5,
    10,
]

MAX_SNAPSHOT_LATENESS_TICKS = 1


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
# Immutable inputs
# ============================================================

require(
    MAPPING.exists(),
    "Frozen Gate 1 macro-zone mapping is missing.",
)

require(
    DROP_AUDIT.exists(),
    "Gate 2B drop audit is missing.",
)

require(
    not OUTPUT.exists(),
    (
        "Refusing to overwrite frozen target contract: "
        f"{OUTPUT}"
    ),
)


mapping = json.loads(
    MAPPING.read_text()
)

require(
    mapping["status"]
    == "FROZEN",
    "Macro-zone mapping is not frozen.",
)

require(
    mapping[
        "n_macro_zones"
    ]
    == 15,
    "Expected frozen 15-zone mapping.",
)


# ============================================================
# Verify Gate 2B evidence
# ============================================================

drop_audit = pl.read_csv(
    DROP_AUDIT
)

require(
    drop_audit.height
    == EXPECTED_DROP_EVENTS,
    (
        "Gate 2B drop-event count changed. "
        f"Expected {EXPECTED_DROP_EVENTS}, "
        f"found {drop_audit.height}."
    ),
)

require(
    drop_audit.filter(
        pl.col("resolved")
        != True
    ).height
    == 0,
    "Gate 2B contains unresolved drop events.",
)

require(
    drop_audit.filter(
        pl.col(
            "tick_lateness"
        )
        != 0
    ).height
    == 0,
    (
        "Expected all audited drop semantics "
        "to resolve at the exact event tick."
    ),
)


# ============================================================
# Reconfirm Gate 2A source schema
#
# We intentionally rerun a small source check here so the
# frozen contract does not depend only on terminal output.
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
    len(reserve)
    == EXPECTED_RESERVE_DEMOS,
    (
        f"Expected {EXPECTED_RESERVE_DEMOS} reserve demos, "
        f"found {len(reserve)}."
    ),
)

audit_demo = reserve[0]

demo = open_v0_demo(
    audit_demo,
    verbose=False,
)

require(
    demo.tickrate
    == V0_DEMO_TICKS_PER_SECOND,
    "64 raw tick/sec contract violated.",
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
    "Unexpected audit map.",
)


required_tick_columns = {
    "round_num",
    "tick",
    "steamid",
    "side",
    "health",
    "X",
    "Y",
    "Z",
    "inventory",
    "place",
}

missing_tick_columns = (
    required_tick_columns
    - set(
        demo.ticks.columns
    )
)

require(
    not missing_tick_columns,
    (
        "Missing required player columns: "
        f"{sorted(missing_tick_columns)}"
    ),
)


required_bomb_columns = {
    "round_num",
    "tick",
    "event",
    "steamid",
    "X",
    "Y",
    "Z",
    "bombsite",
}

missing_bomb_columns = (
    required_bomb_columns
    - set(
        demo.bomb.columns
    )
)

require(
    not missing_bomb_columns,
    (
        "Missing required bomb columns: "
        f"{sorted(missing_bomb_columns)}"
    ),
)


plant_sites = set(
    demo.bomb
    .filter(
        pl.col("event")
        == "plant"
    )[
        "bombsite"
    ]
    .drop_nulls()
    .to_list()
)

require(
    plant_sites
    <= {
        "BombsiteA",
        "BombsiteB",
    },
    (
        "Unexpected plant site semantics: "
        f"{sorted(plant_sites)}"
    ),
)


carrier_semantic_example_found = False

for row in (
    demo.ticks
    .filter(
        (pl.col("side") == "t")
        &
        (pl.col("health") > 0)
    )
    .select([
        "inventory",
        "place",
    ])
    .iter_rows(
        named=True
    )
):

    if not has_c4(
        row[
            "inventory"
        ]
    ):
        continue

    place = row[
        "place"
    ]

    if (
        place is None
        or str(place) == ""
    ):
        continue

    carrier_semantic_example_found = True
    break


require(
    carrier_semantic_example_found,
    (
        "Could not reconfirm carried C4 "
        "semantic-place source."
    ),
)


# ============================================================
# Frozen contract
# ============================================================

contract = {
    "version":
        "V3",

    "artifact":
        "Near-future bomb macro-zone target contract",

    "status":
        "FROZEN",

    "map":
        "de_mirage",

    "tick_clock": {
        "raw_ticks_per_second":
            V0_DEMO_TICKS_PER_SECOND,

        "max_snapshot_lateness_ticks":
            MAX_SNAPSHOT_LATENESS_TICKS,

        "snapshot_resolution":
            (
                "First available player snapshot at or after "
                "the nominal tick, with at most +1 raw tick "
                "lateness."
            ),
    },

    "observation_grid": {
        "anchor":
            "round.freeze_end",

        "first_current_observation_sec":
            CURRENT_START_SEC,

        "stride_sec":
            CURRENT_STRIDE_SEC,

        "current_observation_policy":
            (
                "Generate nominal current observations at "
                "freeze_end + 5s, +10s, +15s, ... ."
            ),

        "current_must_be_preplant":
            True,

        "current_must_be_before_round_end":
            True,

        "reason_for_not_using_freeze_end_itself":
            (
                "Avoid overrepresenting trivial spawn-state "
                "persistence immediately at live-round start."
            ),
    },

    "prediction_horizons_sec":
        TARGET_HORIZONS_SEC,

    "target_tick_policy": {
        "definition":
            (
                "For a resolved current snapshot at tick t, "
                "the nominal target is t + horizon_seconds * 64."
            ),

        "resolution":
            (
                "Resolve the target onto the first available "
                "player snapshot at or after that nominal "
                "target, with at most +1 raw tick lateness."
            ),

        "target_must_be_before_round_end":
            True,

        "horizons_evaluated_separately":
            True,
    },

    "target_semantic_priority": [
        {
            "state":
                "PLANTED",

            "rule":
                (
                    "If a canonical valid plant event occurred "
                    "at or before the resolved target tick, "
                    "use its bombsite."
                ),

            "mapping": {
                "BombsiteA":
                    "A_SITE_COMPLEX",

                "BombsiteB":
                    "B_SITE_COMPLEX",
            },
        },

        {
            "state":
                "CARRIED_INVENTORY",

            "rule":
                (
                    "If exactly one alive T player carries C4 "
                    "in the target snapshot, use that player's "
                    "Awpy-normalized Valve place and map it "
                    "through the frozen 15-zone mapping."
                ),
        },

        {
            "state":
                "CARRIED_PICKUP_FALLBACK",

            "rule":
                (
                    "If inventory exposes no carrier and the "
                    "latest bomb event at or before the target "
                    "tick is pickup, use that event only to "
                    "identify the carrier steamid. Resolve the "
                    "carrier's current target-snapshot place."
                ),
        },

        {
            "state":
                "DROPPED",

            "rule":
                (
                    "If the latest bomb event is drop, use the "
                    "Valve semantic place recovered for that "
                    "drop event from the exact-tick dropper "
                    "snapshot and map it through the frozen "
                    "15-zone mapping."
                ),

            "xyz_classifier_used":
                False,
        },
    ],

    "label_space": {
        "source":
            "docs/v3_macro_zone_mapping_v1_frozen.json",

        "n_macro_zones":
            15,

        "mapping_version":
            "v1",
    },

    "round_exclusion_policy": [
        "MISSING_FREEZE_END",
        "MISSING_ROUND_END",
        "INVALID_TIMING_ORDER",
        "PLANT_LABEL_UNRESOLVED",
    ],

    "observation_or_horizon_exclusion_policy": [
        "CURRENT_SNAPSHOT_UNAVAILABLE",
        "CURRENT_STATE_AT_OR_AFTER_PLANT",
        "CURRENT_BOMB_STATE_UNRESOLVED",
        "CURRENT_BOMB_PLACE_UNRESOLVED",
        "TARGET_AFTER_ROUND_END",
        "TARGET_SNAPSHOT_UNAVAILABLE",
        "TARGET_BOMB_STATE_UNRESOLVED",
        "TARGET_BOMB_PLACE_UNRESOLVED",
    ],

    "anti_leakage_contract": {
        "features_at_time_t_may_use_future_information":
            False,

        "future_bomb_events_may_be_used_as_features":
            False,

        "future_target_state_is_used_only_as_y":
            True,

        "current_observations_after_plant_are_allowed":
            False,

        "future_targets_may_cross_a_plant_event":
            True,

        "demo_round_bomb_site_metadata_used_as_target":
            False,
    },

    "scientific_data_boundary": {
        "v2_d_confirm_used":
            False,

        "13_v2_reserve_demos_role":
            (
                "V3 design / target-integrity development "
                "evidence only. They are not eligible to "
                "become V3 confirmation data."
            ),

        "v3_confirmation_data":
            (
                "Must be acquired and frozen separately "
                "after V3 development decisions are complete."
            ),
    },

    "evidence": {
        "macro_mapping_sha256":
            sha256_file(
                MAPPING
            ),

        "drop_audit_sha256":
            sha256_file(
                DROP_AUDIT
            ),

        "drop_events_audited":
            EXPECTED_DROP_EVENTS,

        "drop_semantic_recovery":
            1.0,

        "drop_tick_lateness":
            "657/657 exact",

        "gate2a_schema_reconfirmed_on":
            audit_demo.name,
    },
}


OUTPUT.write_text(
    json.dumps(
        contract,
        indent=2,
    )
    + "\n"
)


print("=" * 100)
print("V3 GATE 2C — FUTURE TARGET CONTRACT FREEZE")
print("=" * 100)

print()
print(
    "Observation start:",
    f"freeze_end + {CURRENT_START_SEC}s",
)

print(
    "Observation stride:",
    f"{CURRENT_STRIDE_SEC}s",
)

print(
    "Prediction horizons:",
    TARGET_HORIZONS_SEC,
)

print(
    "Snapshot tolerance:",
    f"+{MAX_SNAPSHOT_LATENESS_TICKS} raw tick",
)

print()
print("Semantic target sources:")
print("  PLANTED  -> canonical plant site")
print("  CARRIED  -> carrier Valve place")
print("  PICKUP   -> current carrier Valve place")
print("  DROPPED  -> exact-tick dropper Valve place")

print()
print(
    "XYZ classifier used for ground truth:",
    False,
)

print(
    "Post-plant current observations:",
    False,
)

print(
    "Future target may cross plant:",
    True,
)

print(
    "V2 D_CONFIRM used:",
    False,
)

print()
print("✅ V3 FUTURE TARGET CONTRACT FROZEN")
print("NEXT_ACTION=BUILD_TARGET_INTEGRITY_AUDIT")

print()
print("Frozen contract:")
print(" ", OUTPUT)
