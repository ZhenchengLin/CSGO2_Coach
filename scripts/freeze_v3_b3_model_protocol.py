from __future__ import annotations

import hashlib
import json
from pathlib import Path

import xgboost


FEATURE_PROTOCOL = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)

OUTPUT = Path(
    "docs/v3_b3_model_protocol_v1.json"
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


for path in [
    FEATURE_PROTOCOL,
    BASELINE_PROTOCOL,
]:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


feature_protocol = json.loads(
    FEATURE_PROTOCOL.read_text()
)

require(
    feature_protocol["status"]
    == "FROZEN_BEFORE_FIRST_B3_FIT",
    "B3 feature protocol is not frozen.",
)

require(
    feature_protocol["total_dimensions"]
    == 24,
    "Expected 24 frozen B3 features.",
)

require(
    feature_protocol[
        "feature_selection_using_confirmation_data"
    ]
    is False,
    "Confirmation-driven feature selection is forbidden.",
)


baseline_protocol = json.loads(
    BASELINE_PROTOCOL.read_text()
)

b3_parent = baseline_protocol[
    "baselines"
][
    "B3_tabular_map_aware"
]

require(
    b3_parent["status"]
    == "FUTURE_STAGE",
    "Unexpected parent B3 status.",
)

require(
    b3_parent[
        "feature_selection_using_confirmation_data"
    ]
    is False,
    "Parent protocol forbids confirmation-driven feature selection.",
)


xgb_config = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.5,
    "reg_lambda": 5.0,

    "objective": "multi:softprob",
    "num_class": 15,

    "eval_metric": "mlogloss",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
}


record = {
    "version":
        "V3_B3_MODEL_PROTOCOL_V1",

    "status":
        "FROZEN_BEFORE_FIRST_B3_FIT",

    "model_family":
        "XGBClassifier",

    "library":
        "xgboost",

    "installed_xgboost_version":
        xgboost.__version__,

    "configuration_source":
        (
            "Project frozen V0 XGB-A5 configuration, "
            "reused without hyperparameter tuning."
        ),

    "intentional_task_changes_from_v0": {
        "num_class":
            {
                "from": 3,
                "to": 15,
                "reason":
                    "V3 target has 15 frozen macro-zone classes.",
            },

        "models":
            {
                "5_seconds":
                    "independent model",

                "10_seconds":
                    "independent model",
            },
    },

    "xgboost_config":
        xgb_config,

    "training": {
        "cv":
            "existing frozen 5-fold grouped-by-match split",

        "fit_separately_by_horizon":
            True,

        "hyperparameter_search":
            False,

        "early_stopping":
            False,

        "confirmation_data_for_training":
            False,

        "confirmation_data_for_model_selection":
            False,

        "sample_weighting":
            None,

        "class_weighting":
            None,
    },

    "input": {
        "feature_protocol":
            str(FEATURE_PROTOCOL),

        "feature_protocol_sha256":
            sha256(FEATURE_PROTOCOL),

        "dimensions":
            24,

        "feature_order":
            feature_protocol[
                "feature_order"
            ],
    },

    "output": {
        "probability_type":
            "native 15-class softprob",

        "hard_prediction":
            "argmax probability",

        "class_order":
            feature_protocol[
                "features"
            ][
                "current_macro_zone"
            ][
                "categories"
            ],
    },

    "evaluation": {
        "primary_metric":
            "multiclass log loss",

        "secondary_metrics": [
            "multiclass Brier",
            "accuracy",
            "macro F1",
            "top-2 accuracy",
            "equal-match log loss",
            "per-zone recall/support",
            "top-1 confidence ECE",
        ],

        "paired_comparators": [
            "B0 persistence",
            "B1 constant motion",
            "B2 zone Markov",
        ],

        "bootstrap_unit":
            "match",

        "bootstrap_repetitions":
            20000,

        "bootstrap_seed":
            20260917,
    },

    "anti_leakage": {
        "feature_information_cutoff":
            "at or before current time t",

        "split_unit":
            "match",

        "random_row_split":
            False,

        "v2_d_confirm_used":
            False,

        "future_d_v3_confirm_used":
            False,
    },

    "post_score_rule":
        (
            "Do not silently tune this B3 V1 configuration "
            "after inspecting its development OOF result. "
            "Any changed feature set or hyperparameter set "
            "must be declared as a new experiment/version."
        ),

    "next_action":
        "Run the first B3 V1 development OOF fit.",
}


if OUTPUT.exists():

    existing = json.loads(
        OUTPUT.read_text()
    )

    require(
        existing == record,
        "Existing B3 model protocol differs.",
    )

else:

    OUTPUT.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n"
    )


print("B3 MODEL PROTOCOL")
print()
print(
    "xgboost version:",
    xgboost.__version__,
)
print(
    "features:",
    record["input"]["dimensions"],
)
print(
    "classes:",
    xgb_config["num_class"],
)
print(
    "n_estimators:",
    xgb_config["n_estimators"],
)
print(
    "max_depth:",
    xgb_config["max_depth"],
)
print(
    "learning_rate:",
    xgb_config["learning_rate"],
)
print(
    "reg_alpha:",
    xgb_config["reg_alpha"],
)
print(
    "reg_lambda:",
    xgb_config["reg_lambda"],
)
print(
    "hyperparameter search: NO"
)
print(
    "early stopping: NO"
)
print(
    "confirmation data: NO"
)
print()
print("B3_MODEL_PROTOCOL_FROZEN")
