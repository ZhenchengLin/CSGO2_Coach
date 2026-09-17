from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score


TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

TARGET_ID = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

SPLITS = Path(
    "docs/v3_dev_cv_splits_v1.csv"
)

SPLIT_ID = Path(
    "docs/v3_dev_cv_splits_v1_identity.json"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)

B0 = Path(
    "data/interim/v3_b0_persistence_oof_predictions.parquet"
)

B1 = Path(
    "data/interim/v3_b1_constant_motion_oof_predictions.parquet"
)

B1_RESULTS = Path(
    "docs/v3_b1_constant_motion_results.json"
)

AMENDMENT = Path(
    "docs/v3_dev_data_role_amendment_v2.json"
)

REGRESSION = Path(
    "docs/v3_gate2_reserve_regression.json"
)

CONTRACT = Path(
    "docs/v3_b2_execution_contract_v1.json"
)

OOF = Path(
    "data/interim/v3_b2_markov_oof_predictions.parquet"
)

RESULTS = Path(
    "docs/v3_b2_markov_results.json"
)


N_CLASSES = 15
N_FOLDS = 5
HORIZONS = [5, 10]

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260917

CALIBRATION_BINS = np.linspace(
    0.0,
    1.0,
    11,
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


def read_json(path):
    return json.loads(
        path.read_text()
    )


# ============================================================
# Verify required frozen inputs
# ============================================================

for path in [
    TARGETS,
    TARGET_ID,
    SPLITS,
    SPLIT_ID,
    MAPPING,
    BASELINE_PROTOCOL,
    B0,
    B1,
    B1_RESULTS,
    AMENDMENT,
    REGRESSION,
]:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


target_id = read_json(
    TARGET_ID
)

require(
    sha256(TARGETS)
    == target_id["output_sha256"],
    "Frozen target dataset SHA mismatch.",
)


split_id = read_json(
    SPLIT_ID
)

require(
    sha256(SPLITS)
    == split_id["split_sha256"],
    "Frozen CV split SHA mismatch.",
)


mapping = read_json(
    MAPPING
)

require(
    mapping["status"] == "FROZEN",
    "Macro mapping is not frozen.",
)


amendment = read_json(
    AMENDMENT
)

require(
    amendment["status"] == "FROZEN",
    "Data-role amendment is not frozen.",
)


regression = read_json(
    REGRESSION
)

require(
    regression["status"] == "PASS",
    "Gate-2 reserve regression is not PASS.",
)


# ============================================================
# Verify B2 frozen protocol
# ============================================================

protocol = read_json(
    BASELINE_PROTOCOL
)

b2_protocol = protocol[
    "baselines"
][
    "B2_zone_markov"
]

print("Frozen B2 protocol:")
print(
    json.dumps(
        b2_protocol,
        indent=2,
    )
)


# ============================================================
# Class order
# ============================================================

class_order = list(
    mapping["zones"].keys()
)

require(
    len(class_order) == N_CLASSES,
    "Expected 15 frozen macro-zones.",
)


zone_to_index = {
    zone: i
    for i, zone
    in enumerate(class_order)
}


# ============================================================
# Freeze execution details BEFORE scoring
# ============================================================

contract = {
    "version":
        "V3_B2_MARKOV_EXECUTION_CONTRACT_V1",

    "status":
        "FROZEN_BEFORE_SCORING",

    "model":
        (
            "First-order conditional macro-zone transition "
            "distribution P(target_zone | current_zone, horizon)."
        ),

    "fit_scope":
        (
            "Training matches only within each frozen CV fold; "
            "fit separately for +5s and +10s."
        ),

    "transition_matrix": {
        "shape":
            [15, 15],

        "rows":
            "current macro-zone",

        "columns":
            "future target macro-zone",
    },

    "smoothing": {
        "method":
            "add-one / Laplace",

        "formula":
            "(count_ij + 1) / (sum_j count_ij + 15)",
    },

    "prediction": {
        "probabilities":
            "Smoothed transition row for current macro-zone",

        "hard_prediction":
            "argmax probability",

        "argmax_tie_break":
            "lowest frozen class index",

        "top2_tie_break":
            "descending probability then lowest frozen class index",
    },

    "evaluation": {
        "primary":
            "multiclass log loss",

        "secondary": [
            "accuracy",
            "macro F1",
            "top-2 accuracy",
            "multiclass Brier sum_j",
            "equal-match log loss",
            "per-zone recall/support",
            "top-1 confidence ECE",
        ],

        "calibration": {
            "type":
                "top-1 confidence ECE",

            "bins":
                10,

            "edges":
                [
                    float(x)
                    for x in CALIBRATION_BINS
                ],
        },

        "paired_comparators": [
            "B0 persistence OOF",
            "B1 constant-motion OOF",
        ],

        "bootstrap_unit":
            "match",

        "bootstrap_repetitions":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,
    },

    "v2_d_confirm_used":
        False,
}


if CONTRACT.exists():
    require(
        read_json(CONTRACT) == contract,
        "Existing B2 execution contract differs.",
    )
else:
    CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
        )
        + "\n"
    )


# ============================================================
# Load data
# ============================================================

targets = pl.read_parquet(
    TARGETS
)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)

b0 = pl.read_parquet(
    B0
)

b1 = pl.read_parquet(
    B1
)


require(
    targets.height == 29_065,
    f"Expected 29,065 targets, got {targets.height}",
)

require(
    targets["demo_filename"].n_unique() == 53,
    "Expected 53 target matches.",
)

require(
    b0.height == targets.height,
    "B0 row count mismatch.",
)

require(
    b1.height == targets.height,
    "B1 row count mismatch.",
)


# Verify class encoding.
encoded = targets.with_columns(
    pl.col("current_macro_zone")
    .replace_strict(
        zone_to_index
    )
    .alias("_current_expected"),

    pl.col("target_macro_zone")
    .replace_strict(
        zone_to_index
    )
    .alias("_target_expected"),
)

require(
    encoded.filter(
        pl.col("current_class_index")
        != pl.col("_current_expected")
    ).height == 0,
    "Current class encoding mismatch.",
)

require(
    encoded.filter(
        pl.col("target_class_index")
        != pl.col("_target_expected")
    ).height == 0,
    "Target class encoding mismatch.",
)


data = targets.join(
    splits.select([
        "demo_filename",
        "fold",
    ]),
    on="demo_filename",
    how="left",
)

require(
    data["fold"].null_count() == 0,
    "Missing fold assignments.",
)


KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


require(
    targets.unique(
        subset=KEY
    ).height == targets.height,
    "Target key not unique.",
)


# ============================================================
# Helpers
# ============================================================

def build_markov_matrix(frame):
    counts = np.zeros(
        (
            N_CLASSES,
            N_CLASSES,
        ),
        dtype=np.int64,
    )

    current = (
        frame["current_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    target = (
        frame["target_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    np.add.at(
        counts,
        (
            current,
            target,
        ),
        1,
    )

    smoothed = (
        counts.astype(np.float64)
        + 1.0
    )

    probabilities = (
        smoothed
        / smoothed.sum(
            axis=1,
            keepdims=True,
        )
    )

    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ),
        "Markov probability rows do not sum to 1.",
    )

    return counts, probabilities


def deterministic_order(probabilities):
    class_index = np.arange(
        N_CLASSES
    )

    # Primary key: -probability.
    # Secondary key: frozen class index.
    return np.lexsort(
        (
            class_index,
            -probabilities,
        )
    )


def calibration_ece(
    confidence,
    correct,
):
    total = len(confidence)

    ece = 0.0
    records = []

    for i in range(10):

        lo = CALIBRATION_BINS[i]
        hi = CALIBRATION_BINS[i + 1]

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

        n = int(mask.sum())

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
            mean_conf - accuracy
        )

        records.append({
            "bin": i,
            "lower": float(lo),
            "upper": float(hi),
            "n": n,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })

    return float(ece), records


# ============================================================
# OOF scoring
# ============================================================

prediction_frames = []
fold_records = []


print()
print("B2 MARKOV — OOF SCORING")


for fold in range(
    N_FOLDS
):

    print()
    print(
        f"FOLD {fold}"
    )

    for horizon in HORIZONS:

        horizon_data = data.filter(
            pl.col("horizon_sec")
            == horizon
        )

        train = horizon_data.filter(
            pl.col("fold")
            != fold
        )

        valid = horizon_data.filter(
            pl.col("fold")
            == fold
        )

        require(
            train.height > 0
            and valid.height > 0,
            "Empty train or validation fold.",
        )


        counts, transition_p = (
            build_markov_matrix(
                train
            )
        )


        current_index = (
            valid["current_class_index"]
            .to_numpy()
            .astype(np.int64)
        )

        true_index = (
            valid["target_class_index"]
            .to_numpy()
            .astype(np.int64)
        )


        probabilities = transition_p[
            current_index
        ]


        predicted_index = np.argmax(
            probabilities,
            axis=1,
        )


        true_probability = probabilities[
            np.arange(valid.height),
            true_index,
        ]


        log_loss = -np.log(
            true_probability
        )


        brier = (
            np.sum(
                probabilities ** 2,
                axis=1,
            )
            - 2.0 * true_probability
            + 1.0
        )


        top2_correct = np.zeros(
            valid.height,
            dtype=bool,
        )


        for i in range(
            valid.height
        ):

            order = deterministic_order(
                probabilities[i]
            )

            top2_correct[i] = (
                true_index[i]
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
                log_loss.tolist(),

            "multiclass_brier":
                brier.tolist(),
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
            pl.DataFrame(base)
        )


        hard_accuracy = float(
            np.mean(
                predicted_index
                == true_index
            )
        )


        fold_records.append({
            "fold":
                fold,

            "horizon_sec":
                horizon,

            "n_train":
                train.height,

            "n_validation":
                valid.height,

            "hard_accuracy":
                hard_accuracy,

            "top2_accuracy":
                float(
                    top2_correct.mean()
                ),

            "log_loss":
                float(
                    log_loss.mean()
                ),

            "transition_row_support": [
                int(x)
                for x in counts.sum(
                    axis=1
                )
            ],
        })


        print(
            f"  +{horizon}s "
            f"train={train.height:,} "
            f"valid={valid.height:,} "
            f"acc={hard_accuracy:.4%} "
            f"top2={top2_correct.mean():.4%} "
            f"LL={log_loss.mean():.6f}"
        )


pred = pl.concat(
    prediction_frames,
    how="vertical",
)


require(
    pred.height == targets.height,
    "B2 OOF row count mismatch.",
)

require(
    pred.unique(
        subset=KEY
    ).height == pred.height,
    "B2 OOF key is not unique.",
)


# ============================================================
# Join B0 and B1 paired rows
# ============================================================

b0_small = b0.select([
    *KEY,

    pl.col(
        "log_loss"
    ).alias(
        "b0_log_loss"
    ),

    pl.col(
        "multiclass_brier"
    ).alias(
        "b0_brier"
    ),

    pl.col(
        "predicted_class_index"
    ).alias(
        "b0_predicted_class_index"
    ),
])


b1_small = b1.select([
    *KEY,

    pl.col(
        "log_loss"
    ).alias(
        "b1_log_loss"
    ),

    pl.col(
        "multiclass_brier"
    ).alias(
        "b1_brier"
    ),

    pl.col(
        "predicted_class_index"
    ).alias(
        "b1_predicted_class_index"
    ),
])


pred = (
    pred
    .join(
        b0_small,
        on=KEY,
        how="left",
    )
    .join(
        b1_small,
        on=KEY,
        how="left",
    )
    .with_columns([
        (
            pl.col("log_loss")
            - pl.col("b0_log_loss")
        ).alias(
            "delta_ll_b2_minus_b0"
        ),

        (
            pl.col("log_loss")
            - pl.col("b1_log_loss")
        ).alias(
            "delta_ll_b2_minus_b1"
        ),

        (
            pl.col("multiclass_brier")
            - pl.col("b0_brier")
        ).alias(
            "delta_brier_b2_minus_b0"
        ),

        (
            pl.col("multiclass_brier")
            - pl.col("b1_brier")
        ).alias(
            "delta_brier_b2_minus_b1"
        ),
    ])
)


require(
    pred["b0_log_loss"].null_count()
    == 0,
    "B0 pairing failed.",
)

require(
    pred["b1_log_loss"].null_count()
    == 0,
    "B1 pairing failed.",
)


# ============================================================
# Aggregate results
# ============================================================

results_by_horizon = {}


for horizon in HORIZONS:

    p = pred.filter(
        pl.col("horizon_sec")
        == horizon
    )

    y_true = (
        p["target_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    y_pred = (
        p["predicted_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    confidence = (
        p["top1_confidence"]
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
            pl.col("log_loss")
            .mean()
            .alias("b2_ll"),

            pl.col("b0_log_loss")
            .mean()
            .alias("b0_ll"),

            pl.col("b1_log_loss")
            .mean()
            .alias("b1_ll"),

            pl.col(
                "delta_ll_b2_minus_b0"
            )
            .mean()
            .alias("delta_b0"),

            pl.col(
                "delta_ll_b2_minus_b1"
            )
            .mean()
            .alias("delta_b1"),
        ])
    )


    delta_b0 = (
        per_match["delta_b0"]
        .to_numpy()
        .astype(np.float64)
    )

    delta_b1 = (
        per_match["delta_b1"]
        .to_numpy()
        .astype(np.float64)
    )


    require(
        len(delta_b0) == 53,
        "Expected 53 match-level deltas.",
    )


    rng = np.random.default_rng(
        BOOTSTRAP_SEED + horizon
    )

    indices = rng.integers(
        0,
        53,
        size=(
            BOOTSTRAP_REPS,
            53,
        ),
    )


    boot_b0 = delta_b0[
        indices
    ].mean(
        axis=1
    )

    boot_b1 = delta_b1[
        indices
    ].mean(
        axis=1
    )


    ci_b0 = np.quantile(
        boot_b0,
        [
            0.025,
            0.975,
        ],
    )

    ci_b1 = np.quantile(
        boot_b1,
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


    result = {
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
                per_match["b2_ll"].mean()
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
                        range(
                            N_CLASSES
                        )
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

        "per_zone":
            per_zone,

        "b0_log_loss_same_rows":
            float(
                p["b0_log_loss"].mean()
            ),

        "b1_log_loss_same_rows":
            float(
                p["b1_log_loss"].mean()
            ),

        "delta_ll_b2_minus_b0":
            float(
                p[
                    "delta_ll_b2_minus_b0"
                ].mean()
            ),

        "equal_match_delta_ll_b2_minus_b0":
            float(
                per_match[
                    "delta_b0"
                ].mean()
            ),

        "b2_minus_b0_match_bootstrap_95ci":
            [
                float(
                    ci_b0[0]
                ),
                float(
                    ci_b0[1]
                ),
            ],

        "delta_ll_b2_minus_b1":
            float(
                p[
                    "delta_ll_b2_minus_b1"
                ].mean()
            ),

        "equal_match_delta_ll_b2_minus_b1":
            float(
                per_match[
                    "delta_b1"
                ].mean()
            ),

        "b2_minus_b1_match_bootstrap_95ci":
            [
                float(
                    ci_b1[0]
                ),
                float(
                    ci_b1[1]
                ),
            ],

        "delta_brier_b2_minus_b0":
            float(
                p[
                    "delta_brier_b2_minus_b0"
                ].mean()
            ),

        "delta_brier_b2_minus_b1":
            float(
                p[
                    "delta_brier_b2_minus_b1"
                ].mean()
            ),
    }


    results_by_horizon[
        str(
            horizon
        )
    ] = result


# ============================================================
# Save
# ============================================================

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
        "B2_MARKOV",

    "status":
        "DEVELOPMENT_OOF_COMPLETE",

    "evaluation":
        "5-fold grouped-by-match OOF development",

    "target_dataset_sha256":
        sha256(TARGETS),

    "cv_split_sha256":
        sha256(SPLITS),

    "execution_contract_sha256":
        sha256(CONTRACT),

    "bootstrap": {
        "unit":
            "match",

        "repetitions":
            BOOTSTRAP_REPS,

        "seed":
            BOOTSTRAP_SEED,
    },

    "results_by_horizon":
        results_by_horizon,

    "fold_records":
        fold_records,

    "oof_predictions":
        str(OOF),

    "oof_predictions_sha256":
        sha256(OOF),

    "v2_d_confirm_used":
        False,

    "v3_confirmation_used":
        False,
}


RESULTS.write_text(
    json.dumps(
        results,
        indent=2,
    )
    + "\n"
)


# ============================================================
# Console report
# ============================================================

print()
print(
    "B2 MARKOV — OOF DEVELOPMENT RESULTS"
)
print()


for horizon in HORIZONS:

    r = results_by_horizon[
        str(
            horizon
        )
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

    print()

    print(
        f"  B0 LL: "
        f"{r['b0_log_loss_same_rows']:.6f}"
    )

    print(
        f"  ΔLL B2-B0: "
        f"{r['delta_ll_b2_minus_b0']:+.6f}"
    )

    print(
        f"  equal-match ΔLL B2-B0: "
        f"{r['equal_match_delta_ll_b2_minus_b0']:+.6f}"
    )

    ci0 = (
        r[
            "b2_minus_b0_match_bootstrap_95ci"
        ]
    )

    print(
        "  B2-B0 bootstrap 95% CI: "
        f"[{ci0[0]:+.6f}, {ci0[1]:+.6f}]"
    )

    print()

    print(
        f"  B1 LL: "
        f"{r['b1_log_loss_same_rows']:.6f}"
    )

    print(
        f"  ΔLL B2-B1: "
        f"{r['delta_ll_b2_minus_b1']:+.6f}"
    )

    ci1 = (
        r[
            "b2_minus_b1_match_bootstrap_95ci"
        ]
    )

    print(
        "  B2-B1 bootstrap 95% CI: "
        f"[{ci1[0]:+.6f}, {ci1[1]:+.6f}]"
    )

    print()


print("B2_OOF_COMPLETE")
print("NO_V2_D_CONFIRM_USED")
