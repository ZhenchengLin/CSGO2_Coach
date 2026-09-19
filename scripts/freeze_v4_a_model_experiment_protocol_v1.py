"""Freeze the first V4-A controlled model experiment.

This script does not fit or score any model.
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

B3_MODEL = Path(
    "docs/v3_b3_model_protocol_v1.json"
)

V4_FEATURE = Path(
    "docs/v4_team_context_feature_contract_v1_frozen.json"
)

GATE1E = Path(
    "docs/v4_gate1e_identity_and_features_audit_v1.json"
)

GATE3A = Path(
    "scripts/audit_v4_gate3a_model_inputs.py"
)

OUTPUT = Path(
    "docs/v4_a_model_experiment_protocol_v1_frozen.json"
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


print("\n=== V4-A MODEL EXPERIMENT PROTOCOL FREEZE ===")

free_gib = shutil.disk_usage(".").free / 1024**3

print(f"Free disk: {free_gib:.2f} GiB")

require(
    free_gib >= 12,
    "Free disk below 12 GiB.",
)

require(
    not OUTPUT.exists(),
    "Frozen model protocol already exists. Refusing to overwrite.",
)

for path in (
    TARGETS,
    MOTION,
    TEAM,
    SPLITS,
    TARGET_ID,
    SPLIT_ID,
    B3_FEATURE,
    B3_MODEL,
    V4_FEATURE,
    GATE1E,
    GATE3A,
):
    require(
        path.is_file(),
        f"Missing required input: {path}",
    )


# ------------------------------------------------------------
# 1. Verify existing frozen research inputs.
# ------------------------------------------------------------

target_id = load_json(TARGET_ID)
split_id = load_json(SPLIT_ID)

b3_feature = load_json(B3_FEATURE)
b3_model = load_json(B3_MODEL)

v4_feature = load_json(V4_FEATURE)
gate1e = load_json(GATE1E)

require(
    sha256(TARGETS) == target_id["output_sha256"],
    "Frozen target dataset changed.",
)

require(
    sha256(SPLITS) == split_id["split_sha256"],
    "Frozen CV split changed.",
)

require(
    sha256(MOTION) == b3_feature["motion_cache_sha256"],
    "Frozen V3 motion cache changed.",
)

require(
    sha256(TEAM)
    == v4_feature["frozen_development_artifact"]["sha256"],
    "Frozen V4 Team Context changed.",
)

require(
    b3_model["input"]["feature_protocol_sha256"]
    == sha256(B3_FEATURE),
    "V3 B3 feature/model protocols disagree.",
)

require(
    v4_feature["status"]
    == "FROZEN_FOR_V4_A_DEVELOPMENT",
    "V4 Team Context feature contract is not frozen.",
)

require(
    gate1e["status"]
    == "GATE_1E_IDENTITY_AND_FEATURE_RECONCILIATION_PASS",
    "Gate 1E has not passed.",
)

require(
    gate1e["issues"] == {},
    "Gate 1E contains unresolved issues.",
)


# ------------------------------------------------------------
# 2. Verify fixed dimensions and class representation.
# ------------------------------------------------------------

b3_order = b3_feature["feature_order"]

team_order = v4_feature["team_context_feature_order"]

class_order = target_id["class_order"]

require(
    len(b3_order) == 24
    and len(set(b3_order)) == 24,
    "Unexpected B3 feature representation.",
)

require(
    len(team_order) == 32
    and len(set(team_order)) == 32,
    "Unexpected Team Context representation.",
)

require(
    len(class_order) == 15
    and len(set(class_order)) == 15,
    "Unexpected target class representation.",
)

require(
    b3_order == b3_model["input"]["feature_order"],
    "B3 feature ordering disagrees with model protocol.",
)

require(
    class_order == b3_model["output"]["class_order"],
    "Target class ordering disagrees with model protocol.",
)

require(
    class_order == v4_feature["macro_zone_order"],
    "V4 macro-zone order differs from frozen targets.",
)

require(
    v4_feature["total_candidate_dimensions"] == 56,
    "V4-A must contain exactly 56 dimensions.",
)


# ------------------------------------------------------------
# 3. Verify development dataset and grouped folds.
# ------------------------------------------------------------

targets = pl.read_parquet(TARGETS)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)

team = pl.read_parquet(TEAM)

require(
    targets.height == 29065,
    "Unexpected development target row count.",
)

require(
    team.height == 14869,
    "Unexpected Team Context observation count.",
)

require(
    targets["demo_filename"].n_unique() == 53,
    "Unexpected development match count.",
)

require(
    splits.height == 53
    and splits["demo_filename"].n_unique() == 53,
    "Unexpected CV split coverage.",
)

require(
    set(splits["fold"].unique().to_list())
    == set(range(5)),
    "Expected frozen folds 0 through 4.",
)

require(
    set(splits["demo_filename"].to_list())
    == set(targets["demo_filename"].unique().to_list()),
    "CV manifest and target matches differ.",
)

for horizon, expected_rows in (
    (5, 14869),
    (10, 14196),
):
    part = targets.filter(
        pl.col("horizon_sec") == horizon
    )

    require(
        part.height == expected_rows,
        f"Unexpected +{horizon}s target coverage.",
    )

    require(
        part["demo_filename"].n_unique() == 53,
        f"Missing matches at +{horizon}s.",
    )

print("Frozen input identities: PASS")
print("Feature dimensions and class order: PASS")
print("Development rows and grouped CV: PASS")


# ------------------------------------------------------------
# 4. Freeze the experiment before first V4 model fit.
# ------------------------------------------------------------

config = dict(b3_model["xgboost_config"])

require(
    b3_model["model_family"] == "XGBClassifier",
    "Unexpected parent model family.",
)

require(
    config["objective"] == "multi:softprob"
    and config["num_class"] == 15,
    "Unexpected XGBoost objective or class count.",
)

require(
    b3_model["training"]["hyperparameter_search"] is False
    and b3_model["training"]["early_stopping"] is False,
    "Parent model uses an unexpected tuning policy.",
)

protocol = {
    "version": "V4_A_MODEL_EXPERIMENT_PROTOCOL_V1",

    "status": "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",

    "research_question": (
        "Does current-time team spatial context improve "
        "future bomb macro-zone forecasting beyond "
        "the existing V3 B3 information?"
    ),

    "scope": {
        "map": "de_mirage",
        "data_role": "HISTORICAL_DEVELOPMENT",
        "development_matches": 53,
        "target_rows": 29065,
        "horizons_sec": [5, 10],
        "target_classes": class_order,
        "class_count": 15,
    },

    "control": {
        "name": "V4_A_CONTROL_24D",
        "model_family": "XGBClassifier",
        "feature_order": b3_order,
        "dimensions": 24,
        "training_policy": (
            "Train new V4 control models from scratch. "
            "Do not alter frozen V3 model artifacts."
        ),
    },

    "candidate": {
        "name": "V4_A_TEAM_CONTEXT_56D",
        "model_family": "XGBClassifier",
        "feature_order": b3_order + team_order,
        "dimensions": 56,
        "only_intended_change": (
            "Append frozen 32-dimensional current-time "
            "Team Context features."
        ),
    },

    "model_configuration": {
        "source": str(B3_MODEL),
        "source_sha256": sha256(B3_MODEL),
        "library": "xgboost",
        "parent_recorded_version": (
            b3_model["installed_xgboost_version"]
        ),
        "xgboost_config": config,
        "same_configuration_for_both_models": True,
    },

    "training": {
        "independent_models_per_horizon": True,
        "cross_validation": "5_FOLD_GROUPED_BY_MATCH",
        "split_file": str(SPLITS),
        "split_sha256": sha256(SPLITS),
        "same_training_rows_per_fold": True,
        "same_validation_rows_per_fold": True,
        "same_target_labels": True,
        "same_model_parameters": True,
        "control_and_candidate_fitted_separately": True,
        "random_row_split": False,
        "hyperparameter_search": False,
        "early_stopping": False,
        "sample_weighting": None,
        "class_weighting": None,
        "use_confirmation_data": False,
    },

    "feature_construction": {
        "b3": (
            "Reproduce the frozen 24-dimensional V3 B3 "
            "feature builder without modifying its semantics."
        ),
        "team_context": (
            "Join frozen 32-dimensional Team Context on "
            "(demo_filename, round_num, current_tick)."
        ),
        "information_cutoff": "at_or_before_current_tick",
        "future_player_positions_as_features": False,
        "future_target_as_feature": False,
    },

    "evaluation": {
        "evaluation_type": "DEVELOPMENT_OUT_OF_FOLD",
        "prediction_type": "15_CLASS_PROBABILITY_DISTRIBUTION",
        "primary_metric": "multiclass_log_loss",
        "primary_metric_direction": "lower_is_better",
        "primary_comparison": (
            "V4 candidate log loss minus "
            "V4 control log loss on identical OOF rows."
        ),
        "report_horizons_separately": True,
        "secondary_metrics": [
            "multiclass_brier",
            "accuracy",
            "macro_f1",
            "top2_accuracy",
            "equal_match_log_loss",
            "per_zone_recall_and_support",
            "top1_confidence_ece",
        ],
        "paired_comparison_unit": (
            "Same match, observation, horizon and target."
        ),
        "bootstrap_unit": "match",
        "bootstrap_repetitions": 20000,
        "bootstrap_seed": 20260919,
        "bootstrap_interval": "95_PERCENTILE",
        "report_both_horizons_regardless_of_results": True,
        "no_posthoc_horizon_selection": True,
        "interpretation": (
            "Exploratory development comparison only. "
            "Not independent V4 confirmation."
        ),
    },

    "output_policy": {
        "new_v4_outputs_only": True,
        "do_not_overwrite_v3_outputs": True,
        "do_not_overwrite_existing_v4_outputs": True,
        "retain_per_row_oof_predictions": True,
        "retain_fold_level_metrics": True,
        "retain_model_and_data_provenance": True,
        "no_model_selection_using_confirmation": True,
    },

    "provenance": {
        "target_sha256": sha256(TARGETS),
        "motion_sha256": sha256(MOTION),
        "team_context_sha256": sha256(TEAM),
        "b3_feature_protocol_sha256": sha256(B3_FEATURE),
        "v4_feature_protocol_sha256": sha256(V4_FEATURE),
        "gate1e_report_sha256": sha256(GATE1E),
        "gate3a_script_sha256": sha256(GATE3A),
        "cv_split_sha256": sha256(SPLITS),
    },

    "scientific_boundaries": {
        "v2_confirmation_used": False,
        "v3_confirmation_used": False,
        "historical_v3_confirmation_is_not_fresh_v4_confirmation": True,
        "fresh_v4_confirmation_acquired": False,
        "v4_confirmation_protocol_frozen": False,
        "model_fitting_performed_by_this_script": False,
        "tactical_decision_layer_in_scope": False,
    },

    "revision_policy": (
        "Any change to the model configuration, features, "
        "training scheme, or evaluation protocol must be "
        "recorded as a new version. Do not silently modify "
        "this frozen experiment."
    ),
}

with OUTPUT.open("x") as file:
    json.dump(protocol, file, indent=2, ensure_ascii=False)
    file.write("\n")


print("\n=== V4 GATE 3B — FINAL SUMMARY ===")
print("Protocol:", OUTPUT)
print("Control dimensions: 24")
print("Candidate dimensions: 56")
print("Target classes: 15")
print("Development matches: 53")
print("Development target rows: 29,065")
print("Cross-validation: 5 match-grouped folds")
print("Model configuration: frozen V3 B3 XGBoost configuration")
print("Primary metric: multiclass log loss")
print("Comparison: paired development OOF")
print("\nV4_A_MODEL_EXPERIMENT_PROTOCOL_FROZEN")
print("No models were fitted or scored.")
print("No confirmation data was accessed.")
