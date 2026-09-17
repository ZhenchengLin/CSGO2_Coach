from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

TARGET_ID = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

SPLITS = Path(
    "docs/v3_dev_cv_splits_v1.csv"
)

SPLIT_ID = Path(
    "docs/v3_dev_cv_splits_v1_identity.json"
)

FEATURE_PROTOCOL = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

MODEL_PROTOCOL = Path(
    "docs/v3_b3_model_protocol_v1.json"
)

B0 = Path(
    "data/interim/v3_b0_persistence_oof_predictions.parquet"
)

B1 = Path(
    "data/interim/v3_b1_constant_motion_oof_predictions.parquet"
)

B2 = Path(
    "data/interim/v3_b2_markov_oof_predictions.parquet"
)

OOF = Path(
    "data/interim/v3_b3_tabular_map_aware_oof_predictions.parquet"
)

RESULTS = Path(
    "docs/v3_b3_tabular_map_aware_results.json"
)


HORIZONS = [5, 10]
N_CLASSES = 15
N_FOLDS = 5

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260917

CALIBRATION_EDGES = np.linspace(
    0.0,
    1.0,
    11,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text()
    )


for path in [
    TARGETS,
    TARGET_ID,
    MOTION,
    SPLITS,
    SPLIT_ID,
    FEATURE_PROTOCOL,
    MODEL_PROTOCOL,
    B0,
    B1,
    B2,
]:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


# ------------------------------------------------------------
# Verify frozen identities / protocols
# ------------------------------------------------------------

target_id = read_json(
    TARGET_ID
)

require(
    sha256(TARGETS)
    == target_id["output_sha256"],
    "Frozen target SHA mismatch.",
)


split_id = read_json(
    SPLIT_ID
)

require(
    sha256(SPLITS)
    == split_id["split_sha256"],
    "Frozen CV split SHA mismatch.",
)


feature_protocol = read_json(
    FEATURE_PROTOCOL
)

require(
    feature_protocol["status"]
    == "FROZEN_BEFORE_FIRST_B3_FIT",
    "Feature protocol is not frozen.",
)

require(
    feature_protocol["total_dimensions"] == 24,
    "Expected 24 frozen B3 features.",
)

require(
    sha256(TARGETS)
    == feature_protocol["target_dataset_sha256"],
    "Feature protocol target SHA mismatch.",
)

require(
    sha256(MOTION)
    == feature_protocol["motion_cache_sha256"],
    "Feature protocol motion SHA mismatch.",
)


model_protocol = read_json(
    MODEL_PROTOCOL
)

require(
    model_protocol["status"]
    == "FROZEN_BEFORE_FIRST_B3_FIT",
    "Model protocol is not frozen.",
)

require(
    model_protocol["input"]["dimensions"] == 24,
    "Model protocol feature dimension mismatch.",
)

require(
    model_protocol["input"]["feature_protocol_sha256"]
    == sha256(FEATURE_PROTOCOL),
    "Model protocol feature protocol SHA mismatch.",
)


XGB_CONFIG = dict(
    model_protocol["xgboost_config"]
)

require(
    XGB_CONFIG["num_class"] == 15,
    "Expected 15-class XGBoost.",
)

require(
    model_protocol["training"]["hyperparameter_search"]
    is False,
    "Hyperparameter search must remain disabled.",
)

require(
    model_protocol["training"]["early_stopping"]
    is False,
    "Early stopping must remain disabled.",
)


class_order = model_protocol[
    "output"
][
    "class_order"
]

feature_order = model_protocol[
    "input"
][
    "feature_order"
]

require(
    len(class_order) == 15,
    "Expected 15 classes.",
)

require(
    len(feature_order) == 24,
    "Expected 24 features.",
)


# ------------------------------------------------------------
# Load / join causal features
# ------------------------------------------------------------

targets = pl.read_parquet(
    TARGETS
)

motion = pl.read_parquet(
    MOTION
)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)


MOTION_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]

KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


require(
    motion.unique(
        subset=MOTION_KEY
    ).height == motion.height,
    "Motion key is not unique.",
)


motion_small = motion.select([
    *MOTION_KEY,
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "status",
])


data = (
    targets
    .join(
        motion_small,
        on=MOTION_KEY,
        how="left",
    )
    .join(
        splits.select([
            "demo_filename",
            "fold",
        ]),
        on="demo_filename",
        how="left",
    )
)


require(
    data.height == 29_065,
    f"Expected 29,065 rows, got {data.height}",
)

require(
    data["fold"].null_count() == 0,
    "Missing fold assignment.",
)

require(
    data.filter(
        pl.col("status") != "RESOLVED"
    ).height == 0,
    "Unresolved motion row.",
)


# ------------------------------------------------------------
# Frozen 24-D feature builder
# ------------------------------------------------------------

zone_to_index = {
    zone: i
    for i, zone in enumerate(
        class_order
    )
}


def build_features(frame):

    n = frame.height

    X = np.zeros(
        (
            n,
            24,
        ),
        dtype=np.float32,
    )

    current_zone = (
        frame["current_macro_zone"]
        .to_list()
    )

    for row_i, zone in enumerate(
        current_zone
    ):
        X[
            row_i,
            zone_to_index[zone],
        ] = 1.0


    source = frame[
        "current_source"
    ].to_list()

    for row_i, value in enumerate(
        source
    ):

        if value in {
            "CARRIED_INVENTORY",
            "CARRIED_PICKUP_FALLBACK",
        }:
            X[row_i, 15] = 1.0

        elif value == "DROPPED":
            X[row_i, 16] = 1.0

        else:
            raise RuntimeError(
                f"Unexpected current_source: {value}"
            )


    xyz = np.column_stack([
        frame["current_bomb_X"].to_numpy(),
        frame["current_bomb_Y"].to_numpy(),
        frame["current_bomb_Z"].to_numpy(),
    ]).astype(
        np.float32,
        copy=False,
    )

    velocity = np.column_stack([
        frame["velocity_X"].to_numpy(),
        frame["velocity_Y"].to_numpy(),
        frame["velocity_Z"].to_numpy(),
    ]).astype(
        np.float32,
        copy=False,
    )

    speed = np.linalg.norm(
        velocity.astype(
            np.float64
        ),
        axis=1,
    ).astype(
        np.float32
    )


    X[:, 17:20] = xyz
    X[:, 20:23] = velocity
    X[:, 23] = speed


    require(
        np.isfinite(X).all(),
        "Non-finite B3 feature.",
    )

    require(
        X.shape[1] == 24,
        "B3 feature dimension changed.",
    )

    return X


# Smoke-test actual order against protocol.
expected_order = (
    [
        f"zone__{zone}"
        for zone in class_order
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
    feature_order == expected_order,
    "Implemented feature order differs from frozen protocol.",
)


# ------------------------------------------------------------
# Calibration helper
# ------------------------------------------------------------

def calibration_ece(
    confidence,
    correct,
):

    total = len(
        confidence
    )

    ece = 0.0
    records = []

    for i in range(10):

        lo = CALIBRATION_EDGES[i]
        hi = CALIBRATION_EDGES[i + 1]

        if i == 9:
            mask = (
                (confidence >= lo)
                & (confidence <= hi)
            )
        else:
            mask = (
                (confidence >= lo)
                & (confidence < hi)
            )

        n = int(
            mask.sum()
        )

        if n == 0:
            records.append({
                "bin": i,
                "lower": float(lo),
                "upper": float(hi),
                "n": 0,
                "mean_confidence": None,
                "accuracy": None,
            })
            continue

        mean_conf = float(
            confidence[mask].mean()
        )

        accuracy = float(
            correct[mask].mean()
        )

        ece += (
            n / total
        ) * abs(
            mean_conf
            - accuracy
        )

        records.append({
            "bin": i,
            "lower": float(lo),
            "upper": float(hi),
            "n": n,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })

    return (
        float(ece),
        records,
    )


# ------------------------------------------------------------
# First B3 V1 OOF fit
# ------------------------------------------------------------

print("B3 V1 TABULAR MAP-AWARE — FIRST OOF FIT")
print()
print("features:", len(feature_order))
print("classes:", len(class_order))
print("hyperparameter search: NO")
print("confirmation used: NO")


prediction_frames = []
fold_records = []


for horizon in HORIZONS:

    print()
    print(f"+{horizon}s")

    horizon_data = data.filter(
        pl.col("horizon_sec")
        == horizon
    )

    for fold in range(
        N_FOLDS
    ):

        train = horizon_data.filter(
            pl.col("fold") != fold
        )

        valid = horizon_data.filter(
            pl.col("fold") == fold
        )


        train_classes = set(
            train[
                "target_class_index"
            ]
            .unique()
            .to_list()
        )

        require(
            train_classes
            == set(range(15)),
            (
                f"Training fold {fold}, +{horizon}s "
                "does not contain all 15 classes."
            ),
        )


        X_train = build_features(
            train
        )

        X_valid = build_features(
            valid
        )

        y_train = (
            train[
                "target_class_index"
            ]
            .to_numpy()
            .astype(np.int64)
        )

        y_valid = (
            valid[
                "target_class_index"
            ]
            .to_numpy()
            .astype(np.int64)
        )


        model = XGBClassifier(
            **XGB_CONFIG
        )

        model.fit(
            X_train,
            y_train,
        )


        probabilities = model.predict_proba(
            X_valid
        )


        require(
            probabilities.shape
            == (
                valid.height,
                N_CLASSES,
            ),
            (
                "Unexpected predict_proba shape: "
                f"{probabilities.shape}"
            ),
        )

        require(
            np.allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-6,
                rtol=0.0,
            ),
            "B3 probabilities do not sum to one.",
        )


        predicted_index = np.argmax(
            probabilities,
            axis=1,
        )

        true_probability = probabilities[
            np.arange(
                valid.height
            ),
            y_valid,
        ]

        require(
            np.all(
                true_probability > 0
            ),
            "Zero true-class probability encountered.",
        )


        row_ll = -np.log(
            true_probability
        )

        row_brier = (
            np.sum(
                probabilities ** 2,
                axis=1,
            )
            - 2.0 * true_probability
            + 1.0
        )


        # Stable top-2:
        # probability descending, then frozen class index.
        top2_correct = np.zeros(
            valid.height,
            dtype=bool,
        )

        class_indices = np.arange(
            N_CLASSES
        )

        for i in range(
            valid.height
        ):

            order = np.lexsort(
                (
                    class_indices,
                    -probabilities[i],
                )
            )

            top2_correct[i] = (
                y_valid[i]
                in order[:2]
            )


        predicted_zone = [
            class_order[i]
            for i in predicted_index
        ]


        base = valid.select([
            "demo_filename",
            "round_num",
            "current_nominal_tick",
            "current_tick",
            "horizon_sec",
            "fold",
            "current_macro_zone",
            "current_class_index",
            "target_macro_zone",
            "target_class_index",
            "stay_same_zone",
        ]).to_dict(
            as_series=False
        )


        base.update({
            "predicted_macro_zone":
                predicted_zone,

            "predicted_class_index":
                predicted_index.tolist(),

            "top2_correct":
                top2_correct.tolist(),

            "top1_confidence":
                probabilities.max(
                    axis=1
                ).tolist(),

            "log_loss":
                row_ll.tolist(),

            "multiclass_brier":
                row_brier.tolist(),
        })


        for class_idx, zone in enumerate(
            class_order
        ):

            base[
                f"p_{zone}"
            ] = probabilities[
                :, class_idx
            ].tolist()


        prediction_frames.append(
            pl.DataFrame(
                base
            )
        )


        acc = float(
            np.mean(
                predicted_index
                == y_valid
            )
        )

        ll = float(
            row_ll.mean()
        )


        fold_records.append({
            "horizon_sec":
                horizon,

            "fold":
                fold,

            "n_train":
                train.height,

            "n_validation":
                valid.height,

            "accuracy":
                acc,

            "top2_accuracy":
                float(
                    top2_correct.mean()
                ),

            "log_loss":
                ll,
        })


        print(
            f"  fold {fold}: "
            f"train={train.height:,} "
            f"valid={valid.height:,} "
            f"acc={acc:.4%} "
            f"top2={top2_correct.mean():.4%} "
            f"LL={ll:.6f}"
        )


pred = pl.concat(
    prediction_frames,
    how="vertical",
)


require(
    pred.height == targets.height,
    (
        f"B3 OOF rows {pred.height} "
        f"!= {targets.height}"
    ),
)

require(
    pred.unique(
        subset=KEY
    ).height == pred.height,
    "B3 OOF evaluation key is not unique.",
)


# ------------------------------------------------------------
# Pair B0 / B1 / B2
# ------------------------------------------------------------

def comparator(
    path,
    prefix,
):
    return (
        pl.read_parquet(
            path
        )
        .select([
            *KEY,

            pl.col(
                "log_loss"
            ).alias(
                f"{prefix}_log_loss"
            ),

            pl.col(
                "multiclass_brier"
            ).alias(
                f"{prefix}_brier"
            ),

            pl.col(
                "predicted_class_index"
            ).alias(
                f"{prefix}_predicted_class_index"
            ),
        ])
    )


pred = (
    pred
    .join(
        comparator(
            B0,
            "b0",
        ),
        on=KEY,
        how="left",
    )
    .join(
        comparator(
            B1,
            "b1",
        ),
        on=KEY,
        how="left",
    )
    .join(
        comparator(
            B2,
            "b2",
        ),
        on=KEY,
        how="left",
    )
)


for prefix in [
    "b0",
    "b1",
    "b2",
]:
    require(
        pred[
            f"{prefix}_log_loss"
        ].null_count()
        == 0,
        f"{prefix.upper()} pairing failed.",
    )


pred = pred.with_columns([
    (
        pl.col("log_loss")
        - pl.col("b0_log_loss")
    ).alias(
        "delta_ll_b3_minus_b0"
    ),

    (
        pl.col("log_loss")
        - pl.col("b1_log_loss")
    ).alias(
        "delta_ll_b3_minus_b1"
    ),

    (
        pl.col("log_loss")
        - pl.col("b2_log_loss")
    ).alias(
        "delta_ll_b3_minus_b2"
    ),

    (
        pl.col("multiclass_brier")
        - pl.col("b2_brier")
    ).alias(
        "delta_brier_b3_minus_b2"
    ),
])


# ------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------

results_by_horizon = {}


for horizon in HORIZONS:

    p = pred.filter(
        pl.col("horizon_sec")
        == horizon
    )

    y_true = (
        p[
            "target_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    y_pred = (
        p[
            "predicted_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    confidence = (
        p[
            "top1_confidence"
        ]
        .to_numpy()
        .astype(np.float64)
    )

    correct = (
        y_true == y_pred
    )


    per_match = (
        p
        .group_by(
            "demo_filename"
        )
        .agg([
            pl.col(
                "log_loss"
            ).mean().alias(
                "b3_ll"
            ),

            pl.col(
                "b0_log_loss"
            ).mean().alias(
                "b0_ll"
            ),

            pl.col(
                "b1_log_loss"
            ).mean().alias(
                "b1_ll"
            ),

            pl.col(
                "b2_log_loss"
            ).mean().alias(
                "b2_ll"
            ),

            pl.col(
                "delta_ll_b3_minus_b2"
            ).mean().alias(
                "delta_b2"
            ),
        ])
    )


    delta_b2 = (
        per_match[
            "delta_b2"
        ]
        .to_numpy()
        .astype(np.float64)
    )

    require(
        len(delta_b2) == 53,
        "Expected 53 match-level B3-B2 deltas.",
    )


    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + horizon
    )

    indices = rng.integers(
        0,
        53,
        size=(
            BOOTSTRAP_REPS,
            53,
        ),
    )

    boot = delta_b2[
        indices
    ].mean(
        axis=1
    )

    ci = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )


    ece, calibration_bins = (
        calibration_ece(
            confidence,
            correct,
        )
    )


    per_zone = []

    for class_idx, zone in enumerate(
        class_order
    ):

        mask = (
            y_true
            == class_idx
        )

        support = int(
            mask.sum()
        )

        recall = (
            float(
                np.mean(
                    y_pred[mask]
                    == class_idx
                )
            )
            if support
            else None
        )

        per_zone.append({
            "zone":
                zone,

            "support":
                support,

            "recall":
                recall,
        })


    moving = p.filter(
        pl.col("stay_same_zone")
        == False
    )

    staying = p.filter(
        pl.col("stay_same_zone")
        == True
    )


    results_by_horizon[
        str(horizon)
    ] = {
        "rows":
            p.height,

        "matches":
            53,

        "log_loss":
            float(
                p["log_loss"].mean()
            ),

        "equal_match_log_loss":
            float(
                per_match["b3_ll"].mean()
            ),

        "multiclass_brier":
            float(
                p[
                    "multiclass_brier"
                ].mean()
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(
                        range(15)
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top2_accuracy":
            float(
                p[
                    "top2_correct"
                ].mean()
            ),

        "calibration": {
            "top1_ece":
                ece,

            "bins":
                calibration_bins,
        },

        "moving_accuracy":
            float(
                (
                    moving[
                        "predicted_class_index"
                    ]
                    ==
                    moving[
                        "target_class_index"
                    ]
                ).mean()
            ),

        "staying_accuracy":
            float(
                (
                    staying[
                        "predicted_class_index"
                    ]
                    ==
                    staying[
                        "target_class_index"
                    ]
                ).mean()
            ),

        "per_zone":
            per_zone,

        "b0_log_loss_same_rows":
            float(
                p[
                    "b0_log_loss"
                ].mean()
            ),

        "b1_log_loss_same_rows":
            float(
                p[
                    "b1_log_loss"
                ].mean()
            ),

        "b2_log_loss_same_rows":
            float(
                p[
                    "b2_log_loss"
                ].mean()
            ),

        "delta_ll_b3_minus_b0":
            float(
                (
                    p["log_loss"]
                    - p["b0_log_loss"]
                ).mean()
            ),

        "delta_ll_b3_minus_b1":
            float(
                (
                    p["log_loss"]
                    - p["b1_log_loss"]
                ).mean()
            ),

        "delta_ll_b3_minus_b2":
            float(
                p[
                    "delta_ll_b3_minus_b2"
                ].mean()
            ),

        "equal_match_delta_ll_b3_minus_b2":
            float(
                per_match[
                    "delta_b2"
                ].mean()
            ),

        "b3_minus_b2_match_bootstrap_95ci":
            [
                float(
                    ci[0]
                ),
                float(
                    ci[1]
                ),
            ],

        "delta_brier_b3_minus_b2":
            float(
                p[
                    "delta_brier_b3_minus_b2"
                ].mean()
            ),
    }


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OOF.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pred.write_parquet(
    OOF,
    compression="zstd",
)


results = {
    "version":
        "V3",

    "baseline":
        "B3_TABULAR_MAP_AWARE_V1",

    "status":
        "DEVELOPMENT_OOF_COMPLETE",

    "fit":
        "FIRST_PREDECLARED_B3_V1_FIT",

    "feature_protocol_sha256":
        sha256(
            FEATURE_PROTOCOL
        ),

    "model_protocol_sha256":
        sha256(
            MODEL_PROTOCOL
        ),

    "target_dataset_sha256":
        sha256(
            TARGETS
        ),

    "cv_split_sha256":
        sha256(
            SPLITS
        ),

    "motion_cache_sha256":
        sha256(
            MOTION
        ),

    "results_by_horizon":
        results_by_horizon,

    "fold_records":
        fold_records,

    "oof_predictions":
        str(
            OOF
        ),

    "oof_predictions_sha256":
        sha256(
            OOF
        ),

    "v2_d_confirm_used":
        False,

    "v3_confirmation_used":
        False,

    "post_score_rule":
        (
            "This B3 V1 feature/model configuration "
            "must not be silently tuned after score inspection."
        ),
}


RESULTS.write_text(
    json.dumps(
        results,
        indent=2,
    )
    + "\n"
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print(
    "B3 V1 — OOF DEVELOPMENT RESULTS"
)
print()


for horizon in HORIZONS:

    r = results_by_horizon[
        str(horizon)
    ]

    print(
        f"+{horizon}s"
    )

    print(
        f"  rows: {r['rows']:,}"
    )

    print(
        f"  log loss: "
        f"{r['log_loss']:.6f}"
    )

    print(
        f"  equal-match LL: "
        f"{r['equal_match_log_loss']:.6f}"
    )

    print(
        f"  Brier: "
        f"{r['multiclass_brier']:.6f}"
    )

    print(
        f"  accuracy: "
        f"{r['accuracy']:.6%}"
    )

    print(
        f"  macro F1: "
        f"{r['macro_f1']:.6f}"
    )

    print(
        f"  top-2 accuracy: "
        f"{r['top2_accuracy']:.6%}"
    )

    print(
        f"  top-1 ECE: "
        f"{r['calibration']['top1_ece']:.6f}"
    )

    print(
        f"  moving accuracy: "
        f"{r['moving_accuracy']:.6%}"
    )

    print(
        f"  staying accuracy: "
        f"{r['staying_accuracy']:.6%}"
    )

    print()

    print(
        f"  B0 LL: "
        f"{r['b0_log_loss_same_rows']:.6f}"
    )

    print(
        f"  B1 LL: "
        f"{r['b1_log_loss_same_rows']:.6f}"
    )

    print(
        f"  B2 LL: "
        f"{r['b2_log_loss_same_rows']:.6f}"
    )

    print()

    print(
        f"  ΔLL B3-B0: "
        f"{r['delta_ll_b3_minus_b0']:+.6f}"
    )

    print(
        f"  ΔLL B3-B1: "
        f"{r['delta_ll_b3_minus_b1']:+.6f}"
    )

    print(
        f"  ΔLL B3-B2: "
        f"{r['delta_ll_b3_minus_b2']:+.6f}"
    )

    print(
        f"  equal-match ΔLL B3-B2: "
        f"{r['equal_match_delta_ll_b3_minus_b2']:+.6f}"
    )

    ci = (
        r[
            "b3_minus_b2_match_bootstrap_95ci"
        ]
    )

    print(
        "  B3-B2 bootstrap 95% CI: "
        f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
    )

    print(
        f"  ΔBrier B3-B2: "
        f"{r['delta_brier_b3_minus_b2']:+.6f}"
    )

    print()


print("B3_V1_OOF_COMPLETE")
print("FIRST_PREDECLARED_B3_V1_FIT")
print("NO_CONFIRMATION_DATA_USED")
