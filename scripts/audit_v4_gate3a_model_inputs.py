"""V4 Gate 3A: read-only model-input compatibility preflight.

No model fitting, demo parsing, confirmation access,
or modification of frozen research artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import polars as pl


TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

TEAM = Path(
    "data/interim/v4_dev_team_context_v1.parquet"
)

SPLITS = Path(
    "docs/v3_dev_cv_splits_v1.csv"
)

TARGET_ID = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

SPLIT_ID = Path(
    "docs/v3_dev_cv_splits_v1_identity.json"
)

B3_FEATURE = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

V4_FEATURE = Path(
    "docs/v4_team_context_feature_contract_v1_frozen.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text())


print("\n=== V4 GATE 3A: MODEL INPUT PREFLIGHT ===")

free_gib = shutil.disk_usage(".").free / 1024**3

print(f"Free disk: {free_gib:.2f} GiB")

require(
    free_gib >= 12,
    "Free disk is below the 12 GiB safety threshold.",
)

for path in (
    TARGETS,
    MOTION,
    TEAM,
    SPLITS,
    TARGET_ID,
    SPLIT_ID,
    B3_FEATURE,
    V4_FEATURE,
):
    require(
        path.is_file(),
        f"Required input is missing: {path}",
    )


# ------------------------------------------------------------
# Verify frozen artifact identities.
# ------------------------------------------------------------

target_id = load_json(TARGET_ID)
split_id = load_json(SPLIT_ID)

b3_protocol = load_json(B3_FEATURE)
v4_protocol = load_json(V4_FEATURE)

require(
    sha256(TARGETS) == target_id["output_sha256"],
    "Frozen target dataset SHA mismatch.",
)

require(
    sha256(SPLITS) == split_id["split_sha256"],
    "Frozen CV split SHA mismatch.",
)

require(
    sha256(MOTION)
    == b3_protocol["motion_cache_sha256"],
    "Frozen motion cache SHA mismatch.",
)

require(
    sha256(TEAM)
    == v4_protocol["frozen_development_artifact"]["sha256"],
    "Frozen Team Context SHA mismatch.",
)

require(
    v4_protocol["status"]
    == "FROZEN_FOR_V4_A_DEVELOPMENT",
    "V4 feature protocol is not frozen.",
)

require(
    b3_protocol["total_dimensions"] == 24,
    "Unexpected B3 feature dimensions.",
)

require(
    v4_protocol["team_context_dimensions"] == 32,
    "Unexpected Team Context dimensions.",
)

require(
    v4_protocol["total_candidate_dimensions"] == 56,
    "Unexpected V4-A feature dimensions.",
)

print("Frozen artifact identities: PASS")


# ------------------------------------------------------------
# Read existing data only.
# ------------------------------------------------------------

targets = pl.read_parquet(TARGETS)
motion = pl.read_parquet(MOTION)
team = pl.read_parquet(TEAM)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)

observation_key = [
    "demo_filename",
    "round_num",
    "current_tick",
]

motion_key = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]

target_key = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]

require(
    targets.height == 29065,
    "Unexpected development target row count.",
)

require(
    team.height == 14869,
    "Unexpected Team Context observation count.",
)

require(
    team.select(observation_key).unique().height
    == team.height,
    "Team Context observation keys are not unique.",
)

require(
    motion.select(motion_key).unique().height
    == motion.height,
    "Motion observation keys are not unique.",
)

require(
    targets.select(target_key).unique().height
    == targets.height,
    "Target keys are not unique.",
)

require(
    splits["demo_filename"].n_unique()
    == splits.height == 53,
    "Expected one CV assignment per development match.",
)

require(
    set(splits["fold"].unique().to_list())
    == set(range(5)),
    "Expected five CV folds numbered 0–4.",
)


# ------------------------------------------------------------
# Join all current-time information.
# ------------------------------------------------------------

motion_features = motion.select(
    motion_key
    + [
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "status",
    ]
)

team_columns = (
    v4_protocol["team_context_feature_order"]
)

require(
    len(team_columns) == 32,
    "Unexpected Team Context feature order.",
)

require(
    team.columns
    == observation_key + team_columns,
    "Saved Team Context schema differs from frozen protocol.",
)

data = (
    targets
    .join(
        motion_features,
        on=motion_key,
        how="left",
        validate="m:1",
    )
    .join(
        team,
        on=observation_key,
        how="left",
        validate="m:1",
    )
    .join(
        splits.select(
            ["demo_filename", "fold"]
        ),
        on="demo_filename",
        how="left",
        validate="m:1",
    )
)

require(
    data.height == targets.height == 29065,
    "Joining model inputs changed target row count.",
)

require(
    data["fold"].null_count() == 0,
    "Missing grouped CV fold assignment.",
)

require(
    data["status"].null_count() == 0,
    "Missing causal motion feature.",
)

require(
    data.filter(
        pl.col("status") != "RESOLVED"
    ).height == 0,
    "Unresolved causal motion feature.",
)

require(
    all(
        data[column].null_count() == 0
        for column in team_columns
    ),
    "Missing Team Context feature after join.",
)

print("Model input joins: PASS")


# ------------------------------------------------------------
# Confirm row and match coverage for each horizon.
# ------------------------------------------------------------

print("\n=== HORIZON COVERAGE ===")

for horizon, expected_rows in (
    (5, 14869),
    (10, 14196),
):

    part = data.filter(
        pl.col("horizon_sec") == horizon
    )

    require(
        part.height == expected_rows,
        f"Unexpected +{horizon}s target count.",
    )

    require(
        part["demo_filename"].n_unique() == 53,
        f"Missing development match at +{horizon}s.",
    )

    require(
        set(part["fold"].unique().to_list())
        == set(range(5)),
        f"Missing CV fold at +{horizon}s.",
    )

    print(
        f"+{horizon}s:",
        part.height,
        "rows |",
        part["demo_filename"].n_unique(),
        "matches | 5 folds",
    )


# ------------------------------------------------------------
# Verify exact training/validation group separation.
# ------------------------------------------------------------

for fold in range(5):

    train_matches = set(
        data.filter(
            pl.col("fold") != fold
        )["demo_filename"].unique().to_list()
    )

    valid_matches = set(
        data.filter(
            pl.col("fold") == fold
        )["demo_filename"].unique().to_list()
    )

    require(
        not (train_matches & valid_matches),
        f"Match leakage detected in fold {fold}.",
    )

    require(
        train_matches | valid_matches
        == set(splits["demo_filename"].to_list()),
        f"Incomplete match coverage in fold {fold}.",
    )

print("Match-grouped train/validation separation: PASS")

print("\n=== GATE 3A FINAL SUMMARY ===")
print("V3 B3 input dimensions: 24")
print("V4-A input dimensions: 56")
print("Frozen current-time observations: 14,869")
print("Joined development target rows: 29,065")
print("Grouped CV folds: 5")

print("\nGATE_3A_MODEL_INPUT_PREFLIGHT_PASS")
print("No models were trained or scored.")
print("No confirmation data was accessed.")
print("No existing research artifacts were modified.")
