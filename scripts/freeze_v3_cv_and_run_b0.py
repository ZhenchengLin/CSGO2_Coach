from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# Inputs
# ============================================================

DATASET = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

DATASET_IDENTITY = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)

EXCLUSIONS = Path(
    "data/interim/v3_dev_target_exclusions.csv"
)

ROUND_EXCLUSIONS = Path(
    "data/interim/v3_dev_round_exclusions.csv"
)


# ============================================================
# Outputs
# ============================================================

ANOMALY_RECORD = Path(
    "docs/v3_gate3c_anomaly_classification.json"
)

CV_SPLITS = Path(
    "docs/v3_dev_cv_splits_v1.csv"
)

CV_IDENTITY = Path(
    "docs/v3_dev_cv_splits_v1_identity.json"
)

B0_PREDICTIONS = Path(
    "data/interim/v3_b0_persistence_oof_predictions.parquet"
)

B0_RESULTS = Path(
    "docs/v3_b0_persistence_results.json"
)


N_FOLDS = 5
N_CLASSES = 15

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


def stable_key(text: str) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Required inputs
# ============================================================

for path in [
    DATASET,
    DATASET_IDENTITY,
    MANIFEST,
    BASELINE_PROTOCOL,
    EXCLUSIONS,
    ROUND_EXCLUSIONS,
]:
    require(
        path.exists(),
        f"Missing required input: {path}",
    )


# ============================================================
# Verify frozen dataset
# ============================================================

dataset_identity = json.loads(
    DATASET_IDENTITY.read_text()
)

require(
    dataset_identity[
        "status"
    ]
    == "FROZEN",
    "V3 target dataset is not frozen.",
)


actual_dataset_sha = sha256_file(
    DATASET
)

require(
    actual_dataset_sha
    == dataset_identity[
        "output_sha256"
    ],
    (
        "V3 target dataset SHA changed.\n"
        f"Expected: {dataset_identity['output_sha256']}\n"
        f"Actual:   {actual_dataset_sha}"
    ),
)


class_order = list(
    dataset_identity[
        "class_order"
    ]
)

require(
    len(
        class_order
    )
    == N_CLASSES,
    "Expected 15 V3 classes.",
)


dataset = pl.read_parquet(
    DATASET
)

manifest = pl.read_csv(
    MANIFEST,
    infer_schema_length=None,
)

exclusions = pl.read_csv(
    EXCLUSIONS,
    infer_schema_length=None,
)

round_exclusions = pl.read_csv(
    ROUND_EXCLUSIONS,
    infer_schema_length=None,
)


require(
    dataset[
        "demo_filename"
    ].n_unique()
    == 53,
    "Expected 53 development matches.",
)


require(
    set(
        dataset[
            "horizon_sec"
        ].unique().to_list()
    )
    == {
        5,
        10,
    },
    "Unexpected horizons.",
)


# ============================================================
# Gate 3C anomaly classification
# ============================================================

nonboundary = (
    exclusions
    .filter(
        pl.col("reason")
        != "TARGET_AFTER_ROUND_END"
    )
)


reason_counts = {
    row[
        "reason"
    ]:
        int(
            row[
                "len"
            ]
        )
    for row in (
        nonboundary
        .group_by(
            "reason"
        )
        .len()
        .iter_rows(
            named=True
        )
    )
}


require(
    reason_counts
    == {
        "CURRENT_SNAPSHOT_UNAVAILABLE":
            10,

        "TARGET_SNAPSHOT_UNAVAILABLE":
            5,

        "CURRENT_BOMB_STATE_UNRESOLVED":
            1,
    },
    (
        "Non-boundary exclusion counts changed: "
        f"{reason_counts}"
    ),
)


round_reason_counts = {
    row[
        "reason"
    ]:
        int(
            row[
                "len"
            ]
        )
    for row in (
        round_exclusions
        .group_by(
            "reason"
        )
        .len()
        .iter_rows(
            named=True
        )
    )
}


require(
    round_reason_counts
    == {
        "MISSING_FREEZE_END":
            10,
    },
    (
        "Unexpected round-level exclusions: "
        f"{round_reason_counts}"
    ),
)


anomaly_record = {
    "version":
        "V3",

    "gate":
        "3C",

    "status":
        "CLOSED",

    "classification": {
        "CURRENT_SNAPSHOT_UNAVAILABLE": {
            "horizon_rows":
                10,

            "class":
                "SOURCE_SNAPSHOT_STREAM_TERMINATION",

            "interpretation":
                (
                    "The round metadata continues but no "
                    "player snapshot exists at or after the "
                    "nominal current tick within the frozen "
                    "+1 raw-tick tolerance."
                ),

            "policy":
                "EXCLUDE",

            "interpolation_allowed":
                False,

            "carry_forward_allowed":
                False,
        },

        "TARGET_SNAPSHOT_UNAVAILABLE": {
            "horizon_rows":
                5,

            "class":
                "SOURCE_SNAPSHOT_STREAM_TERMINATION",

            "interpretation":
                (
                    "The current observation is valid but "
                    "the source player-snapshot stream ends "
                    "before the future target tick."
                ),

            "policy":
                "EXCLUDE",

            "interpolation_allowed":
                False,

            "carry_forward_allowed":
                False,
        },

        "CURRENT_BOMB_STATE_UNRESOLVED": {
            "horizon_rows":
                1,

            "class":
                "CAUSAL_BOMB_EVIDENCE_UNAVAILABLE",

            "interpretation":
                (
                    "At the exact current snapshot there is "
                    "no inventory C4 carrier and no bomb "
                    "event at or before t that can causally "
                    "resolve the bomb state."
                ),

            "policy":
                "EXCLUDE",

            "future_evidence_allowed":
                False,
        },

        "MISSING_FREEZE_END": {
            "rounds":
                10,

            "class":
                "ROUND_TIMING_METADATA_UNAVAILABLE",

            "policy":
                "EXCLUDE_ROUND",
        },
    },

    "candidate_horizon_rows":
        int(
            dataset.height
            + exclusions.height
        ),

    "valid_horizon_rows":
        int(
            dataset.height
        ),

    "all_excluded_horizon_rows":
        int(
            exclusions.height
        ),

    "nonboundary_horizon_rows":
        int(
            nonboundary.height
        ),

    "nonboundary_fraction_of_candidates":
        float(
            nonboundary.height
            /
            (
                dataset.height
                + exclusions.height
            )
        ),

    "target_contract_change_required":
        False,

    "macro_zone_change_required":
        False,

    "match_removal_required":
        False,

    "conclusion":
        (
            "All observed non-boundary exclusions have "
            "deterministic source-availability or causal-"
            "evidence explanations already covered by the "
            "frozen V3 contract. Gate 3C is closed without "
            "changing the target definition."
        ),
}


if ANOMALY_RECORD.exists():

    existing = json.loads(
        ANOMALY_RECORD.read_text()
    )

    require(
        existing
        == anomaly_record,
        (
            "Existing Gate 3C anomaly record differs "
            "from proposed frozen classification."
        ),
    )

else:

    ANOMALY_RECORD.write_text(
        json.dumps(
            anomaly_record,
            indent=2,
        )
        + "\n"
    )


# ============================================================
# Freeze match-level CV assignment
#
# No targets used.
#
# Within each historical source role:
# deterministic SHA256(filename) order
# then round-robin across five folds.
# ============================================================

role_offsets = {
    "V0_DEVELOPMENT":
        0,

    "V2_RESERVE":
        1,

    "V3_FRESH":
        2,
}


manifest_roles = set(
    manifest[
        "dev_source_role"
    ].unique().to_list()
)


require(
    manifest_roles
    == set(
        role_offsets
    ),
    (
        "Unexpected manifest source roles: "
        f"{manifest_roles}"
    ),
)


split_rows = []


for role in [
    "V0_DEVELOPMENT",
    "V2_RESERVE",
    "V3_FRESH",
]:

    role_rows = (
        manifest
        .filter(
            pl.col(
                "dev_source_role"
            )
            == role
        )
        .to_dicts()
    )


    role_rows.sort(
        key=lambda row: (
            stable_key(
                str(
                    row[
                        "demo_filename"
                    ]
                )
            ),
            str(
                row[
                    "demo_filename"
                ]
            ),
        )
    )


    offset = role_offsets[
        role
    ]


    for index, row in enumerate(
        role_rows
    ):

        fold = (
            index
            + offset
        ) % N_FOLDS


        split_rows.append({
            "demo_filename":
                str(
                    row[
                        "demo_filename"
                    ]
                ),

            "match_date":
                str(
                    row[
                        "match_date"
                    ]
                ),

            "dev_source_role":
                role,

            "fold":
                int(
                    fold
                ),
        })


splits = pl.DataFrame(
    split_rows,
    infer_schema_length=None,
).sort([
    "fold",
    "match_date",
    "demo_filename",
])


require(
    splits.height
    == 53,
    "Expected 53 CV assignments.",
)


require(
    splits[
        "demo_filename"
    ].n_unique()
    == 53,
    "A match appears in more than one CV fold.",
)


require(
    set(
        splits[
            "fold"
        ].unique().to_list()
    )
    == {
        0,
        1,
        2,
        3,
        4,
    },
    "Expected exactly five folds.",
)


# ------------------------------------------------------------
# Every dataset match has exactly one frozen fold
# ------------------------------------------------------------

dataset_matches = set(
    dataset[
        "demo_filename"
    ].unique().to_list()
)

split_matches = set(
    splits[
        "demo_filename"
    ].to_list()
)


require(
    dataset_matches
    == split_matches,
    "CV split and target dataset match sets differ.",
)


# ------------------------------------------------------------
# Freeze CSV
# ------------------------------------------------------------

if CV_SPLITS.exists():

    existing = pl.read_csv(
        CV_SPLITS,
        infer_schema_length=None,
    )

    require(
        existing.equals(
            splits
        ),
        (
            "Existing CV split differs from proposed "
            "deterministic split."
        ),
    )

else:

    splits.write_csv(
        CV_SPLITS
    )


cv_sha = sha256_file(
    CV_SPLITS
)


fold_summary = (
    splits
    .group_by([
        "fold",
        "dev_source_role",
    ])
    .len()
    .sort([
        "fold",
        "dev_source_role",
    ])
)


cv_identity = {
    "version":
        "V3",

    "artifact":
        "D_V3_DEV grouped cross-validation split",

    "protocol_version":
        "v1",

    "status":
        "FROZEN",

    "n_matches":
        53,

    "n_folds":
        N_FOLDS,

    "group_unit":
        "match",

    "target_labels_used_to_assign_folds":
        False,

    "assignment_method":
        (
            "Within each dev_source_role, sort matches by "
            "SHA256(demo_filename), then round-robin across "
            "five folds using a frozen role-specific offset."
        ),

    "role_offsets":
        role_offsets,

    "random_row_split_allowed":
        False,

    "same_match_train_validation_overlap_allowed":
        False,

    "split_file":
        str(
            CV_SPLITS
        ),

    "split_sha256":
        cv_sha,
}


if CV_IDENTITY.exists():

    existing = json.loads(
        CV_IDENTITY.read_text()
    )

    require(
        existing
        == cv_identity,
        "Existing CV identity differs.",
    )

else:

    CV_IDENTITY.write_text(
        json.dumps(
            cv_identity,
            indent=2,
        )
        + "\n"
    )


# ============================================================
# Attach fold to dataset
# ============================================================

data = dataset.join(
    splits.select([
        "demo_filename",
        "fold",
    ]),
    on="demo_filename",
    how="left",
)


require(
    data[
        "fold"
    ].null_count()
    == 0,
    "Some target rows have no CV fold.",
)


# ============================================================
# B0 Persistence
#
# Hard:
#   prediction = current_macro_zone
#
# Probability adapter:
#   q = (training correct + 1) / (n_train + 2)
#
#   predicted class gets q
#   other 14 classes share 1-q equally
# ============================================================

prediction_rows = []

fold_records = []


for horizon in HORIZONS:

    horizon_data = (
        data
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )
    )


    for fold in range(
        N_FOLDS
    ):

        train = (
            horizon_data
            .filter(
                pl.col("fold")
                != fold
            )
        )

        test = (
            horizon_data
            .filter(
                pl.col("fold")
                == fold
            )
        )


        require(
            train.height > 0,
            "Empty B0 training fold.",
        )

        require(
            test.height > 0,
            "Empty B0 validation fold.",
        )


        n_correct_train = (
            train
            .filter(
                pl.col(
                    "stay_same_zone"
                )
                == True
            )
            .height
        )


        q = (
            n_correct_train
            + 1
        ) / (
            train.height
            + 2
        )


        other_p = (
            1.0
            - q
        ) / (
            N_CLASSES
            - 1
        )


        require(
            0.0 < other_p < 1.0,
            "Invalid B0 other-class probability.",
        )

        require(
            0.0 < q < 1.0,
            "Invalid B0 persistence probability.",
        )


        fold_loss = []


        for row in test.iter_rows(
            named=True
        ):

            current_index = int(
                row[
                    "current_class_index"
                ]
            )

            target_index = int(
                row[
                    "target_class_index"
                ]
            )


            probabilities = np.full(
                N_CLASSES,
                other_p,
                dtype=np.float64,
            )

            probabilities[
                current_index
            ] = q


            require(
                math.isclose(
                    float(
                        probabilities.sum()
                    ),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "B0 probabilities do not sum to 1.",
            )


            true_probability = float(
                probabilities[
                    target_index
                ]
            )

            row_log_loss = (
                -math.log(
                    true_probability
                )
            )


            one_hot = np.zeros(
                N_CLASSES,
                dtype=np.float64,
            )

            one_hot[
                target_index
            ] = 1.0


            row_brier = float(
                np.sum(
                    (
                        probabilities
                        - one_hot
                    )
                    ** 2
                )
            )


            prediction_row = {
                "demo_filename":
                    row[
                        "demo_filename"
                    ],

                "round_num":
                    int(
                        row[
                            "round_num"
                        ]
                    ),

                "current_nominal_tick":
                    int(
                        row[
                            "current_nominal_tick"
                        ]
                    ),

                "horizon_sec":
                    horizon,

                "fold":
                    fold,

                "current_macro_zone":
                    row[
                        "current_macro_zone"
                    ],

                "target_macro_zone":
                    row[
                        "target_macro_zone"
                    ],

                "predicted_macro_zone":
                    row[
                        "current_macro_zone"
                    ],

                "target_class_index":
                    target_index,

                "predicted_class_index":
                    current_index,

                "stay_same_zone":
                    bool(
                        row[
                            "stay_same_zone"
                        ]
                    ),

                "train_persistence_q":
                    float(
                        q
                    ),

                "log_loss":
                    float(
                        row_log_loss
                    ),

                "multiclass_brier":
                    float(
                        row_brier
                    ),
            }


            for class_index, zone in enumerate(
                class_order
            ):

                prediction_row[
                    f"p_{zone}"
                ] = float(
                    probabilities[
                        class_index
                    ]
                )


            prediction_rows.append(
                prediction_row
            )

            fold_loss.append(
                row_log_loss
            )


        fold_records.append({
            "horizon_sec":
                horizon,

            "fold":
                fold,

            "n_train":
                train.height,

            "n_validation":
                test.height,

            "n_train_persistence":
                n_correct_train,

            "train_persistence_q":
                float(
                    q
                ),

            "validation_log_loss":
                float(
                    np.mean(
                        fold_loss
                    )
                ),
        })


predictions = pl.DataFrame(
    prediction_rows,
    infer_schema_length=None,
)


require(
    predictions.height
    == dataset.height,
    (
        "OOF prediction row count differs "
        "from target dataset."
    ),
)


duplicate_oof = (
    predictions
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
    duplicate_oof.height
    == 0,
    "Duplicate/missing B0 OOF prediction keys.",
)


# ============================================================
# Aggregate B0 metrics
# ============================================================

results_by_horizon = {}


for horizon in HORIZONS:

    pred = (
        predictions
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )
    )


    y_true = np.array(
        pred[
            "target_class_index"
        ].to_list(),
        dtype=np.int64,
    )

    y_pred = np.array(
        pred[
            "predicted_class_index"
        ].to_list(),
        dtype=np.int64,
    )


    row_weighted_ll = float(
        pred[
            "log_loss"
        ].mean()
    )

    row_weighted_brier = float(
        pred[
            "multiclass_brier"
        ].mean()
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(
                range(
                    N_CLASSES
                )
            ),
            average="macro",
            zero_division=0,
        )
    )


    per_match = (
        pred
        .group_by(
            "demo_filename"
        )
        .agg([
            pl.col(
                "log_loss"
            )
            .mean()
            .alias(
                "match_log_loss"
            ),

            pl.len().alias(
                "n_rows"
            ),
        ])
    )


    equal_match_ll = float(
        per_match[
            "match_log_loss"
        ].mean()
    )


    results_by_horizon[
        str(
            horizon
        )
    ] = {
        "n_rows":
            int(
                pred.height
            ),

        "n_matches":
            int(
                pred[
                    "demo_filename"
                ].n_unique()
            ),

        "row_weighted_log_loss":
            row_weighted_ll,

        "equal_match_log_loss":
            equal_match_ll,

        "multiclass_brier":
            row_weighted_brier,

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "move_rate":
            float(
                1.0
                - accuracy
            ),
    }


# ============================================================
# Write predictions first
# ============================================================

B0_PREDICTIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


predictions.write_parquet(
    B0_PREDICTIONS,
    compression="zstd",
)


prediction_sha = sha256_file(
    B0_PREDICTIONS
)


# ============================================================
# Final B0 result record
# ============================================================

b0_record = {
    "version":
        "V3",

    "baseline":
        "B0_PERSISTENCE",

    "status":
        "DEVELOPMENT_OOF_COMPLETE",

    "evaluation_type":
        "5-fold match-grouped out-of-fold development evaluation",

    "development_dataset_sha256":
        actual_dataset_sha,

    "cv_split_sha256":
        cv_sha,

    "hard_prediction":
        (
            "predicted future macro-zone = "
            "current macro-zone"
        ),

    "probability_adapter":
        {
            "formula":
                (
                    "q=(n_correct_train+1)/(n_train+2)"
                ),

            "predicted_class_probability":
                "q",

            "other_class_probability":
                "(1-q)/14",

            "fit_with_validation_data":
                False,

            "fit_separately_by_horizon_and_fold":
                True,
        },

    "primary_metric":
        "multiclass_log_loss",

    "multiclass_brier_definition":
        (
            "mean over rows of sum_j "
            "(p_j - y_j)^2"
        ),

    "results_by_horizon":
        results_by_horizon,

    "fold_records":
        fold_records,

    "oof_predictions":
        str(
            B0_PREDICTIONS
        ),

    "oof_predictions_sha256":
        prediction_sha,

    "v2_d_confirm_used":
        False,

    "v3_confirmation_data_used":
        False,
}


if B0_RESULTS.exists():

    existing = json.loads(
        B0_RESULTS.read_text()
    )

    require(
        existing
        == b0_record,
        "Existing B0 result differs.",
    )

else:

    B0_RESULTS.write_text(
        json.dumps(
            b0_record,
            indent=2,
        )
        + "\n"
    )


# ============================================================
# Report
# ============================================================

print("=" * 108)
print("V3 GATE 3C ANOMALY CLASSIFICATION")
print("=" * 108)

print(
    "Candidate horizon rows:",
    anomaly_record[
        "candidate_horizon_rows"
    ],
)

print(
    "Non-boundary exclusions:",
    anomaly_record[
        "nonboundary_horizon_rows"
    ],
)

print(
    "Fraction:",
    f"{anomaly_record['nonboundary_fraction_of_candidates']:.6%}",
)

print(
    "Target-contract change required:",
    False,
)

print(
    "Gate 3C:",
    "CLOSED",
)


print()
print("=" * 108)
print("V3 DEVELOPMENT CV SPLIT")
print("=" * 108)

print(
    fold_summary
)


print()
print(
    "CV split SHA256:",
    cv_sha,
)


print()
print("=" * 108)
print("B0 PERSISTENCE — OOF DEVELOPMENT RESULTS")
print("=" * 108)


for horizon in HORIZONS:

    result = results_by_horizon[
        str(
            horizon
        )
    ]

    print()
    print(
        f"+{horizon}s"
    )

    print(
        "  rows:",
        f"{result['n_rows']:,}",
    )

    print(
        "  matches:",
        result[
            "n_matches"
        ],
    )

    print(
        "  log loss:",
        f"{result['row_weighted_log_loss']:.6f}",
    )

    print(
        "  equal-match LL:",
        f"{result['equal_match_log_loss']:.6f}",
    )

    print(
        "  Brier:",
        f"{result['multiclass_brier']:.6f}",
    )

    print(
        "  accuracy:",
        f"{result['accuracy']:.6%}",
    )

    print(
        "  macro F1:",
        f"{result['macro_f1']:.6f}",
    )

    print(
        "  move rate:",
        f"{result['move_rate']:.6%}",
    )


print()
print("FOLD CALIBRATION PARAMETERS")
print("-" * 108)

for record in fold_records:

    print(
        f"+{record['horizon_sec']}s "
        f"fold={record['fold']} "
        f"train={record['n_train']} "
        f"valid={record['n_validation']} "
        f"q={record['train_persistence_q']:.6f} "
        f"LL={record['validation_log_loss']:.6f}"
    )


print()
print("=" * 108)

print(
    "✅ V3 GATE 3C CLOSED"
)

print(
    "✅ V3 DEVELOPMENT CV SPLITS FROZEN"
)

print(
    "✅ B0 PERSISTENCE OOF COMPLETE"
)

print(
    "NEXT_ACTION=BUILD_AND_RUN_B1_CONSTANT_MOTION"
)

print("=" * 108)

print()
print("Artifacts:")
print(" ", ANOMALY_RECORD)
print(" ", CV_SPLITS)
print(" ", CV_IDENTITY)
print(" ", B0_PREDICTIONS)
print(" ", B0_RESULTS)
