from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)

OUTPUT = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


for p in [
    TARGETS,
    MOTION,
    MAPPING,
    BASELINE_PROTOCOL,
]:
    require(
        p.exists(),
        f"Missing: {p}",
    )


mapping = json.loads(
    MAPPING.read_text()
)

require(
    mapping["status"] == "FROZEN",
    "Macro mapping not frozen.",
)

zones = list(
    mapping["zones"].keys()
)

require(
    len(zones) == 15,
    "Expected 15 macro-zones.",
)


baseline = json.loads(
    BASELINE_PROTOCOL.read_text()
)

b3 = baseline[
    "baselines"
][
    "B3_tabular_map_aware"
]

require(
    b3["status"] == "FUTURE_STAGE",
    "Unexpected frozen B3 status.",
)

require(
    b3[
        "feature_selection_using_confirmation_data"
    ] is False,
    "Confirmation feature selection must be forbidden.",
)


targets = pl.read_parquet(
    TARGETS
)

motion = pl.read_parquet(
    MOTION
)


motion_key = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]


require(
    motion.unique(
        subset=motion_key
    ).height
    == motion.height,
    "Motion key is not unique.",
)


motion_small = motion.select([
    *motion_key,
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "status",
])


data = targets.join(
    motion_small,
    on=motion_key,
    how="left",
)


require(
    data.height == targets.height,
    "Feature join changed target row count.",
)

require(
    data.height == 29_065,
    f"Expected 29,065 rows, got {data.height}.",
)

require(
    data["demo_filename"].n_unique()
    == 53,
    "Expected 53 development matches.",
)


for col in [
    "current_macro_zone",
    "current_source",
    "current_bomb_X",
    "current_bomb_Y",
    "current_bomb_Z",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
]:
    require(
        data[col].null_count() == 0,
        f"Null feature: {col}",
    )


require(
    data.filter(
        pl.col("status") != "RESOLVED"
    ).height == 0,
    "Unresolved motion row encountered.",
)


source_values = set(
    data[
        "current_source"
    ].unique().to_list()
)

allowed_source_values = {
    "CARRIED_INVENTORY",
    "CARRIED_PICKUP_FALLBACK",
    "DROPPED",
}

require(
    source_values
    <= allowed_source_values,
    (
        "Unexpected current_source values: "
        f"{source_values}"
    ),
)


velocity = np.column_stack([
    data["velocity_X"].to_numpy(),
    data["velocity_Y"].to_numpy(),
    data["velocity_Z"].to_numpy(),
]).astype(
    np.float64,
    copy=False,
)

require(
    np.isfinite(velocity).all(),
    "Non-finite velocity value.",
)

speed = np.linalg.norm(
    velocity,
    axis=1,
)


feature_names = (
    [
        f"zone__{zone}"
        for zone in zones
    ]
    +
    [
        "bomb_state__CARRIED",
        "bomb_state__DROPPED",
        "current_bomb_X",
        "current_bomb_Y",
        "current_bomb_Z",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "speed",
    ]
)

require(
    len(feature_names) == 24,
    f"Expected 24 features, got {len(feature_names)}",
)


record = {
    "version":
        "V3_B3_FEATURE_PROTOCOL_V1",

    "status":
        "FROZEN_BEFORE_FIRST_B3_FIT",

    "parent_baseline":
        "B3_tabular_map_aware",

    "research_question":
        (
            "Can a nonlinear tabular model combine current "
            "map state and recent causal bomb motion to improve "
            "near-future macro-zone forecasting beyond the "
            "frozen B0-B2 development baselines?"
        ),

    "causality_rule":
        (
            "Every feature must be observable at or before "
            "current time t."
        ),

    "models":
        {
            "5":
                "separate multiclass model",

            "10":
                "separate multiclass model",
        },

    "features": {
        "current_macro_zone": {
            "encoding":
                "full one-hot",

            "categories":
                zones,

            "dimensions":
                15,
        },

        "bomb_state": {
            "encoding":
                "full one-hot",

            "categories": [
                "CARRIED",
                "DROPPED",
            ],

            "source_mapping": {
                "CARRIED_INVENTORY":
                    "CARRIED",

                "CARRIED_PICKUP_FALLBACK":
                    "CARRIED",

                "DROPPED":
                    "DROPPED",
            },

            "dimensions":
                2,

            "provenance_not_exposed_as_feature":
                True,
        },

        "current_xyz": {
            "columns": [
                "current_bomb_X",
                "current_bomb_Y",
                "current_bomb_Z",
            ],

            "transform":
                "raw",

            "dimensions":
                3,
        },

        "recent_velocity": {
            "columns": [
                "velocity_X",
                "velocity_Y",
                "velocity_Z",
            ],

            "history_seconds":
                1.0,

            "transform":
                "raw",

            "dimensions":
                3,
        },

        "speed": {
            "definition":
                "sqrt(vx^2 + vy^2 + vz^2)",

            "transform":
                "raw",

            "clipping":
                None,

            "dimensions":
                1,
        },
    },

    "feature_order":
        feature_names,

    "total_dimensions":
        24,

    "explicitly_excluded": [
        "prior_source provenance",
        "current resolver provenance as separate categories",
        "demo identity",
        "team identity",
        "player identity",
        "future target information",
        "future player information",
        "B2 prediction vector",
        "V2 D_CONFIRM",
        "future D_V3_CONFIRM",
    ],

    "speed_audit": {
        "unique_motion_rows":
            int(motion.height),

        "gt_300":
            int(
                (
                    np.linalg.norm(
                        np.column_stack([
                            motion["velocity_X"].to_numpy(),
                            motion["velocity_Y"].to_numpy(),
                            motion["velocity_Z"].to_numpy(),
                        ]),
                        axis=1,
                    )
                    > 300
                ).sum()
            ),

        "maximum":
            float(speed.max()),

        "policy":
            (
                "Retain raw values. No clipping is introduced "
                "before first B3 fit."
            ),
    },

    "development_rows":
        int(data.height),

    "development_matches":
        int(
            data[
                "demo_filename"
            ].n_unique()
        ),

    "target_dataset_sha256":
        sha256(TARGETS),

    "motion_cache_sha256":
        sha256(MOTION),

    "macro_mapping_sha256":
        sha256(MAPPING),

    "baseline_protocol_sha256":
        sha256(BASELINE_PROTOCOL),

    "feature_selection_using_confirmation_data":
        False,

    "v2_d_confirm_used":
        False,

    "v3_confirmation_used":
        False,

    "next_action":
        (
            "Freeze exact B3 model configuration before "
            "first B3 model fit."
        ),
}


if OUTPUT.exists():

    existing = json.loads(
        OUTPUT.read_text()
    )

    require(
        existing == record,
        "Existing B3 feature protocol differs.",
    )

else:

    OUTPUT.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n"
    )


print("B3 V1 FEATURE PROTOCOL")
print()
print("development rows:", data.height)
print(
    "development matches:",
    data["demo_filename"].n_unique(),
)
print("feature dimensions:", len(feature_names))
print("zones:", len(zones))
print("speed >300:", record["speed_audit"]["gt_300"])
print(
    "max speed:",
    f"{record['speed_audit']['maximum']:.3f}",
)
print()
print("bomb-state mapping:")
print("  CARRIED_INVENTORY       -> CARRIED")
print("  CARRIED_PICKUP_FALLBACK -> CARRIED")
print("  DROPPED                 -> DROPPED")
print()
print("speed clipping: NONE")
print("confirmation used: NO")
print()
print("B3_FEATURE_PROTOCOL_FROZEN")
