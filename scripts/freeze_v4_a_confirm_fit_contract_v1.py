"""Freeze V4-A full-development fitting rules.

No model fitting, confirmation access, or frozen-file modification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import polars as pl


CLOSURE = Path("docs/v4_a_development_closure_v1.json")
EXPERIMENT = Path("docs/v4_a_model_experiment_protocol_v1_frozen.json")
FEATURE = Path("docs/v4_team_context_feature_contract_v1_frozen.json")
PARENT_MODEL = Path("docs/v3_b3_model_protocol_v1.json")

TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
MOTION = Path("data/interim/v3_b1_motion_inputs_v1.parquet")
TEAM = Path("data/interim/v4_dev_team_context_v1.parquet")

OUTPUT = Path("docs/v4_a_confirm_fit_contract_v1_frozen.json")


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


print("\n=== V4-A GATE 6A: FINAL FIT CONTRACT ===")

free_gib = shutil.disk_usage(".").free / 1024**3

require(
    free_gib >= 12,
    f"Free disk below 12 GiB: {free_gib:.2f}",
)

for path in (
    CLOSURE,
    EXPERIMENT,
    FEATURE,
    PARENT_MODEL,
    TARGETS,
    MOTION,
    TEAM,
):
    require(path.is_file(), f"Missing input: {path}")

require(
    not OUTPUT.exists(),
    f"Contract already exists: {OUTPUT}",
)

closure = read_json(CLOSURE)
experiment = read_json(EXPERIMENT)
feature = read_json(FEATURE)
parent_model = read_json(PARENT_MODEL)

require(
    closure["status"]
    == "DEVELOPMENT_CLOSED_NOT_INDEPENDENTLY_CONFIRMED",
    "V4-A development is not closed.",
)

require(
    experiment["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "V4-A experiment protocol is not frozen.",
)

require(
    feature["status"] == "FROZEN_FOR_V4_A_DEVELOPMENT",
    "V4-A feature contract is not frozen.",
)

require(
    sha256(EXPERIMENT)
    == closure["source_artifacts"]["experiment_protocol"]["sha256"],
    "Closure/experiment protocol SHA mismatch.",
)

require(
    sha256(FEATURE)
    == closure["source_artifacts"]["feature_contract"]["sha256"],
    "Closure/feature contract SHA mismatch.",
)

for name, path in (
    ("target", TARGETS),
    ("motion", MOTION),
    ("team_context", TEAM),
):
    require(
        sha256(path)
        == experiment["provenance"][f"{name}_sha256"],
        f"Frozen {name} SHA mismatch.",
    )

require(
    sha256(PARENT_MODEL)
    == experiment["model_configuration"]["source_sha256"],
    "Parent XGBoost protocol SHA mismatch.",
)

require(
    experiment["model_configuration"]["xgboost_config"]
    == parent_model["xgboost_config"],
    "Frozen XGBoost configuration mismatch.",
)

require(
    experiment["control"]["dimensions"] == 24
    and experiment["candidate"]["dimensions"] == 56,
    "Unexpected feature dimensions.",
)

require(
    experiment["candidate"]["feature_order"]
    == experiment["control"]["feature_order"]
    + feature["team_context_feature_order"],
    "Candidate feature ordering mismatch.",
)

targets = pl.read_parquet(
    TARGETS,
    columns=["demo_filename", "horizon_sec", "target_class_index"],
)

require(
    targets.height == 29065
    and targets["demo_filename"].n_unique() == 53,
    "Unexpected full-development training population.",
)

for horizon, expected_rows in ((5, 14869), (10, 14196)):
    part = targets.filter(
        pl.col("horizon_sec") == horizon
    )

    require(
        part.height == expected_rows
        and part["demo_filename"].n_unique() == 53,
        f"Unexpected +{horizon}s training population.",
    )

    require(
        set(part["target_class_index"].unique().to_list())
        == set(range(15)),
        f"Missing training class at +{horizon}s.",
    )

artifacts = {
    "control_plus5": "artifacts/v4_a_confirm_frozen/control_plus5.json",
    "candidate_plus5": "artifacts/v4_a_confirm_frozen/candidate_plus5.json",
    "control_plus10": "artifacts/v4_a_confirm_frozen/control_plus10.json",
    "candidate_plus10": "artifacts/v4_a_confirm_frozen/candidate_plus10.json",
}

for artifact in artifacts.values():
    require(
        not Path(artifact).exists(),
        f"Final model artifact already exists: {artifact}",
    )

contract = {
    "version": "V4_A_CONFIRM_FIT_CONTRACT_V1",
    "status": "FROZEN_BEFORE_V4_CONFIRMATION_ACCESS",

    "development_population": {
        "matches": 53,
        "target_rows": 29065,
        "rows_by_horizon": {"5": 14869, "10": 14196},
        "target_classes": experiment["scope"]["target_classes"],
        "development_only": True,
    },

    "model_variants": {
        "control": {
            "dimensions": 24,
            "feature_order": experiment["control"]["feature_order"],
        },
        "candidate": {
            "dimensions": 56,
            "feature_order": experiment["candidate"]["feature_order"],
        },
    },

    "fitting": {
        "model_family": "XGBClassifier",
        "xgboost_config": (
            experiment["model_configuration"]["xgboost_config"]
        ),
        "fit_separately_by_horizon": True,
        "fit_control_and_candidate_separately": True,
        "use_all_frozen_development_rows_per_horizon": True,
        "cross_validation_during_final_fit": False,
        "feature_scaling": False,
        "hyperparameter_search": False,
        "early_stopping": False,
        "sample_weighting": None,
        "class_weighting": None,
        "confirmation_rows_used_for_training": False,
        "expected_final_model_fits": 4,
    },

    "feature_construction": {
        "control": (
            "Reproduce the frozen V3 B3 24D feature representation."
        ),
        "candidate": (
            "Append the frozen V4-A 32D Team Context representation "
            "in its exact feature order."
        ),
        "information_cutoff": "at_or_before_current_tick",
        "confirmation_feature_semantics_must_match_development": True,
    },

    "artifact_plan": artifacts,

    "artifact_freeze_requirements": [
        "Save all four final model artifacts.",
        "Verify model loading and probability output shapes.",
        "Record SHA256 for each saved model.",
        "Record source data and feature-protocol SHA256 values.",
        "Record exact training rows, match counts and package versions.",
        "Freeze model artifacts before V4 confirmation scoring.",
        "Never refit or replace models based on confirmation results.",
    ],

    "confirmation_boundary": {
        "fresh_v4_confirmation_data_used": False,
        "v2_confirmation_reuse": False,
        "v3_confirmation_reuse": False,
        "confirmation_acquisition_protocol_frozen": False,
        "confirmation_scoring_protocol_frozen": False,
        "confirmation_performance_calculated": False,
    },

    "source_sha256": {
        "development_closure": sha256(CLOSURE),
        "experiment_protocol": sha256(EXPERIMENT),
        "feature_contract": sha256(FEATURE),
        "parent_model_protocol": sha256(PARENT_MODEL),
        "targets": sha256(TARGETS),
        "motion": sha256(MOTION),
        "team_context": sha256(TEAM),
    },

    "revision_policy": (
        "Do not modify this frozen V1 contract after fitting. "
        "Any changed training rule requires a new version."
    ),
}

with OUTPUT.open("x") as file:
    json.dump(contract, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("Development and protocol identities: PASS")
print("Training population: 53 matches / 29,065 target rows")
print("Final models: 24D and 56D, separately for +5s and +10s")
print("Expected final model fits: 4")
print("Frozen XGBoost configuration: PASS")
print("Final fit performed: NO")
print("V4 confirmation data accessed: NO")
print("Contract:", OUTPUT)
print("\nV4_A_CONFIRM_FIT_CONTRACT_FROZEN")
