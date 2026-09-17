from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


OUT = Path(
    "docs/v3_confirm_fit_contract_v1.json"
)

CLOSURE = Path(
    "docs/v3_development_closure_v1.json"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)

TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

TARGET_ID = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

SPATIAL = Path(
    "data/interim/"
    "v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)

SPATIAL_ID = Path(
    "docs/v3_b1_spatial_cache_identity_v1.json"
)

MACRO_MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

B3_FEATURE_PROTOCOL = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

B3_MODEL_PROTOCOL = Path(
    "docs/v3_b3_model_protocol_v1.json"
)

DEV_MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

DEV_MANIFEST_ID = Path(
    "docs/v3_dev_manifest_identity.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text()
    )


required = [
    CLOSURE,
    BASELINE_PROTOCOL,
    TARGETS,
    TARGET_ID,
    MOTION,
    SPATIAL,
    SPATIAL_ID,
    MACRO_MAPPING,
    B3_FEATURE_PROTOCOL,
    B3_MODEL_PROTOCOL,
    DEV_MANIFEST,
    DEV_MANIFEST_ID,
]

for path in required:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


closure = read_json(CLOSURE)
baseline = read_json(BASELINE_PROTOCOL)
target_id = read_json(TARGET_ID)
spatial_id = read_json(SPATIAL_ID)
feature_protocol = read_json(
    B3_FEATURE_PROTOCOL
)
model_protocol = read_json(
    B3_MODEL_PROTOCOL
)


# ============================================================
# Scientific boundary guards
# ============================================================

require(
    closure["status"]
    == "DEVELOPMENT_CLOSED",
    "V3 development is not closed.",
)

require(
    closure[
        "selected_confirmation_candidate"
    ]["name"]
    == "B3_TABULAR_MAP_AWARE_V1",
    "Unexpected confirmation candidate.",
)

require(
    closure["v2_d_confirm_used"]
    is False,
    "V2 D_CONFIRM contamination recorded.",
)

require(
    closure["v3_confirmation_used"]
    is False,
    "V3 confirmation data already recorded as used.",
)


# ============================================================
# Frozen dataset guards
# ============================================================

require(
    sha256(TARGETS)
    == target_id["output_sha256"],
    "Target dataset SHA mismatch.",
)

require(
    spatial_id["status"] == "FROZEN",
    "Spatial cache identity not frozen.",
)

require(
    sha256(SPATIAL)
    == spatial_id["output_sha256"],
    "Spatial cache SHA mismatch.",
)

require(
    spatial_id["gate1_regression_pass"]
    is True,
    "Gate-1 spatial regression did not pass.",
)

require(
    spatial_id["gate1_reserve_rows"]
    == 186_953,
    "Gate-1 reserve row regression changed.",
)


# ============================================================
# Frozen baseline semantic guards
# ============================================================

adapter = baseline[
    "probability_adapter_for_deterministic_baselines"
]

require(
    adapter["training_only"] is True,
    "Deterministic adapter must be training-only.",
)

require(
    adapter["confidence_formula"]
    == "q = (n_correct_train + 1) / (n_train + 2)",
    "B0/B1 adapter formula changed.",
)

require(
    baseline["baselines"][
        "B1_constant_motion"
    ][
        "ground_truth_uses_xyz_resolver"
    ]
    is False,
    "B1 GT resolver rule changed.",
)

require(
    baseline["baselines"][
        "B2_zone_markov"
    ]["smoothing"]
    == (
        "Add-one Dirichlet/Laplace smoothing "
        "over all 15 target classes."
    ),
    "B2 smoothing rule changed.",
)


# ============================================================
# Frozen B3 guards
# ============================================================

require(
    feature_protocol["status"]
    == "FROZEN_BEFORE_FIRST_B3_FIT",
    "B3 feature protocol status changed.",
)

require(
    feature_protocol["total_dimensions"]
    == 24,
    "Expected 24 B3 features.",
)

require(
    feature_protocol[
        "feature_selection_using_confirmation_data"
    ]
    is False,
    "Confirmation-driven B3 feature selection forbidden.",
)

require(
    model_protocol["status"]
    == "FROZEN_BEFORE_FIRST_B3_FIT",
    "B3 model protocol status changed.",
)

require(
    model_protocol["input"]["dimensions"]
    == 24,
    "B3 model input dimension changed.",
)

require(
    model_protocol[
        "training"
    ][
        "hyperparameter_search"
    ]
    is False,
    "B3 hyperparameter search must remain off.",
)

require(
    model_protocol[
        "training"
    ][
        "early_stopping"
    ]
    is False,
    "B3 early stopping must remain off.",
)


git_head = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()


record = {
    "version":
        "V3_CONFIRM_FIT_CONTRACT_V1",

    "status":
        "FROZEN_BEFORE_CONFIRMATION_ACCESS",

    "frozen_date":
        "2026-09-17",

    "parent_development_closure_commit":
        git_head,

    "development_boundary": {
        "matches":
            53,

        "horizon_rows": {
            "5":
                14869,

            "10":
                14196,
        },

        "development_closed":
            True,

        "selected_candidate":
            "B3_TABULAR_MAP_AWARE_V1",

        "V2_D_CONFIRM":
            "SEALED_AND_FORBIDDEN",

        "D_V3_CONFIRM_accessed":
            False,
    },

    "class_space": {
        "n_classes":
            15,

        "class_order_source":
            "docs/v3_b3_feature_protocol_v1.json",

        "macro_mapping_source":
            str(MACRO_MAPPING),
    },

    "B0_persistence_final_fit": {
        "fit_population":
            (
                "All frozen D_V3_DEV target rows "
                "for the corresponding horizon."
            ),

        "hard_prediction":
            (
                "predicted_macro_zone = "
                "current_macro_zone"
            ),

        "n_correct_definition":
            "stay_same_zone == True",

        "probability_adapter":
            (
                "q=(n_correct_train+1)/(n_train+2)"
            ),

        "predicted_class_probability":
            "q",

        "each_other_class_probability":
            "(1-q)/14",

        "fit_separately_by_horizon":
            True,

        "confirmation_rows_used_for_q":
            False,
    },

    "B1_constant_motion_final_fit": {
        "motion_source":
            str(MOTION),

        "spatial_training_source":
            str(SPATIAL),

        "spatial_training_rows":
            774111,

        "spatial_training_matches":
            53,

        "resolver":
            "scipy.spatial.cKDTree",

        "resolver_build":
            (
                "cKDTree(train_xyz), where train_xyz "
                "is all frozen Gate-1 exact spatial "
                "samples from the 53 development matches."
            ),

        "projection":
            (
                "projected_xyz = current_xyz "
                "+ float(horizon_sec) * velocity_xyz"
            ),

        "query":
            {
                "k":
                    1,

                "workers":
                    -1,
            },

        "fine_place":
            (
                "train_places[nn_index]"
            ),

        "macro_zone":
            (
                "frozen place_to_zone lookup from "
                "v3_macro_zone_mapping_v1_frozen.json"
            ),

        "class_index":
            (
                "frozen zone_to_index lookup from "
                "the B3/V3 class order"
            ),

        "custom_tie_breaking":
            False,

        "distance_metric":
            (
                "cKDTree default Euclidean distance "
                "in raw XYZ coordinates"
            ),

        "n_correct_definition":
            (
                "full-development hard predicted "
                "class index == target_class_index"
            ),

        "probability_adapter":
            (
                "q=(n_correct_train+1)/(n_train+2)"
            ),

        "predicted_class_probability":
            "q",

        "each_other_class_probability":
            "(1-q)/14",

        "q_fit_separately_by_horizon":
            True,

        "ground_truth_uses_xyz_resolver":
            False,

        "confirmation_semantic_samples_used":
            False,

        "confirmation_rows_used_for_q":
            False,
    },

    "B2_zone_markov_final_fit": {
        "fit_population":
            (
                "All frozen D_V3_DEV target rows "
                "for the corresponding horizon."
            ),

        "matrix":
            "15x15 P(target_zone | current_zone)",

        "counts":
            (
                "Count current_class_index -> "
                "target_class_index transitions."
            ),

        "smoothing":
            (
                "(count_ij + 1) / "
                "(row_count_i + 15)"
            ),

        "fit_separately_by_horizon":
            True,

        "confirmation_rows_used_for_fit":
            False,
    },

    "B3_tabular_map_aware_final_fit": {
        "candidate":
            "B3_TABULAR_MAP_AWARE_V1",

        "feature_protocol":
            str(B3_FEATURE_PROTOCOL),

        "model_protocol":
            str(B3_MODEL_PROTOCOL),

        "feature_dimensions":
            24,

        "fit_population":
            (
                "All frozen D_V3_DEV rows for "
                "the corresponding horizon."
            ),

        "models":
            {
                "5":
                    (
                        "One 15-class XGBoost model fit "
                        "on all 14,869 +5s development rows."
                    ),

                "10":
                    (
                        "One 15-class XGBoost model fit "
                        "on all 14,196 +10s development rows."
                    ),
            },

        "xgboost_config":
            model_protocol["xgboost_config"],

        "hyperparameter_search":
            False,

        "early_stopping":
            False,

        "sample_weighting":
            None,

        "class_weighting":
            None,

        "feature_scaling":
            False,

        "confirmation_rows_used_for_fit":
            False,
    },

    "artifact_plan": {
        "directory":
            "artifacts/v3_confirm_frozen",

        "B0":
            "b0_adapter.json",

        "B1":
            "b1_constant_motion.json",

        "B2":
            "b2_zone_markov.json",

        "B3_plus5":
            "b3_plus5.json",

        "B3_plus10":
            "b3_plus10.json",

        "freeze_record":
            "docs/v3_confirm_model_freeze.json",

        "requirements": [
            "SHA256 for every artifact",
            "training dataset SHA256",
            "spatial cache SHA256",
            "motion cache SHA256",
            "feature/model protocol SHA256",
            "package versions",
            "training row counts",
            "training match count",
            "git commit identity",
            "confirmation_data_accessed=false",
        ],
    },

    "confirmation_prediction_rule": {
        "model_regeneration_after_confirmation_score":
            False,

        "feature_changes_after_confirmation_score":
            False,

        "adapter_recalibration_on_confirmation":
            False,

        "B2_refit_on_confirmation":
            False,

        "confirmation_outcome_based_row_drops":
            False,
    },

    "source_sha256": {
        "development_closure":
            sha256(CLOSURE),

        "baseline_protocol":
            sha256(BASELINE_PROTOCOL),

        "targets":
            sha256(TARGETS),

        "target_identity":
            sha256(TARGET_ID),

        "motion":
            sha256(MOTION),

        "spatial":
            sha256(SPATIAL),

        "spatial_identity":
            sha256(SPATIAL_ID),

        "macro_mapping":
            sha256(MACRO_MAPPING),

        "b3_feature_protocol":
            sha256(B3_FEATURE_PROTOCOL),

        "b3_model_protocol":
            sha256(B3_MODEL_PROTOCOL),

        "development_manifest":
            sha256(DEV_MANIFEST),

        "development_manifest_identity":
            sha256(DEV_MANIFEST_ID),
    },

    "package_versions": {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "polars",
            "scipy",
            "scikit-learn",
            "xgboost",
        ]
    },

    "scientific_note":
        (
            "This contract is frozen after V3 development "
            "closure and before access to D_V3_CONFIRM. "
            "It defines final full-development fitting only "
            "and does not calculate confirmation performance."
        ),
}


if OUT.exists():
    existing = read_json(OUT)

    require(
        existing == record,
        "Existing confirmation fit contract differs.",
    )

else:
    OUT.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n"
    )


print("V3 CONFIRMATION FIT CONTRACT")
print()
print("development matches: 53")
print("B0 final adapter: full-development training q")
print(
    "B1 resolver: "
    "cKDTree k=1 workers=-1 on 774,111 dev samples"
)
print("B1 projection: xyz + horizon * velocity")
print("B1 final adapter: full-development training q")
print("B2 final fit: full-development Laplace Markov")
print("B3 +5 rows: 14,869")
print("B3 +10 rows: 14,196")
print("B3 hyperparameter search: NO")
print("confirmation accessed: NO")
print()
print("V3_CONFIRM_FIT_CONTRACT_FROZEN")
