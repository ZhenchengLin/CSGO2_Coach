from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score


TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
TARGET_ID = Path("docs/v3_dev_target_dataset_identity.json")

SPLITS = Path("docs/v3_dev_cv_splits_v1.csv")
SPLIT_ID = Path("docs/v3_dev_cv_splits_v1_identity.json")

SPATIAL = Path(
    "data/interim/"
    "v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)
SPATIAL_ID = Path(
    "docs/v3_b1_spatial_cache_identity_v1.json"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

B0 = Path(
    "data/interim/"
    "v3_b0_persistence_oof_predictions.parquet"
)

AMENDMENT = Path(
    "docs/v3_dev_data_role_amendment_v2.json"
)

RESERVE_REGRESSION = Path(
    "docs/v3_gate2_reserve_regression.json"
)

CONTRACT = Path(
    "docs/v3_b1_execution_contract_v2.json"
)

OOF = Path(
    "data/interim/"
    "v3_b1_constant_motion_oof_predictions.parquet"
)

RESULTS = Path(
    "docs/v3_b1_constant_motion_results.json"
)


N_CLASSES = 15
N_FOLDS = 5
HORIZONS = [5, 10]

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260917


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


def load_json(path):
    return json.loads(path.read_text())


# ============================================================
# Verify frozen artifacts
# ============================================================

for path in [
    TARGETS,
    TARGET_ID,
    SPLITS,
    SPLIT_ID,
    SPATIAL,
    SPATIAL_ID,
    MOTION,
    MAPPING,
    B0,
    AMENDMENT,
    RESERVE_REGRESSION,
]:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


target_id = load_json(TARGET_ID)

require(
    sha256(TARGETS)
    == target_id["output_sha256"],
    "Target dataset SHA mismatch.",
)


split_id = load_json(SPLIT_ID)

require(
    sha256(SPLITS)
    == split_id["split_sha256"],
    "CV split SHA mismatch.",
)


spatial_id = load_json(SPATIAL_ID)

require(
    spatial_id["status"] == "FROZEN",
    "Spatial cache identity is not frozen.",
)

require(
    sha256(SPATIAL)
    == spatial_id["output_sha256"],
    "Spatial cache SHA mismatch.",
)

require(
    spatial_id["gate1_regression_pass"] is True,
    "Gate-1 spatial regression not passed.",
)

require(
    spatial_id["gate1_reserve_rows"] == 186_953,
    "Gate-1 reserve row count changed.",
)


amendment = load_json(AMENDMENT)

require(
    amendment["status"] == "FROZEN",
    "Data-role amendment not frozen.",
)


reserve_regression = load_json(
    RESERVE_REGRESSION
)

require(
    reserve_regression["status"] == "PASS",
    "Gate-2 reserve regression is not PASS.",
)


# ============================================================
# Frozen macro mapping
# ============================================================

mapping = load_json(MAPPING)

require(
    mapping["status"] == "FROZEN",
    "Macro mapping is not frozen.",
)


class_order = list(
    mapping["zones"].keys()
)

require(
    len(class_order) == 15,
    "Expected 15 macro-zones.",
)


zone_to_index = {
    zone: i
    for i, zone in enumerate(class_order)
}


place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate fine place: {place}",
        )
        place_to_zone[place] = zone


require(
    len(place_to_zone) == 23,
    "Expected 23 fine places.",
)


# ============================================================
# Freeze execution contract BEFORE B1 scoring
# ============================================================

contract = {
    "version":
        "V3_B1_EXECUTION_CONTRACT_V2",

    "status":
        "FROZEN_BEFORE_SCORING",

    "supersedes_prescore_attempt":
        "docs/v3_b1_execution_contract_v1.json",

    "supersession_reason":
        (
            "The first pre-score attempt omitted parts of "
            "the frozen Gate-1 active-player sampling filter. "
            "Its regression guard aborted before any B1 OOF "
            "predictions or B1 metrics were produced."
        ),

    "motion": {
        "history":
            "approximately one second earlier causal bomb state",

        "velocity":
            "v=(x_t-x_prev)/actual_elapsed_seconds",

        "projection":
            "x_hat(t+h)=x_t+h*v",

        "horizons_seconds":
            [5, 10],

        "motion_cache_expected_rows":
            14869,

        "motion_cache_expected_unresolved":
            0,

        "fallback_to_persistence":
            False,
    },

    "spatial_resolver": {
        "algorithm":
            "scipy.spatial.cKDTree 1-nearest-neighbor",

        "distance":
            "raw XYZ Euclidean distance",

        "fit_data":
            "training-match player semantic samples only",

        "heldout_match_samples_forbidden":
            True,

        "ground_truth_uses_xyz_resolver":
            False,

        "gate1_regression":
            "13 reserves = 186953 exact samples; 23 places",
    },

    "probability_adapter": {
        "formula":
            "q=(n_correct_train+1)/(n_train+2)",

        "predicted_class_probability":
            "q",

        "each_other_class_probability":
            "(1-q)/14",

        "fit_separately_per_fold_and_horizon":
            True,
    },

    "evaluation": {
        "primary":
            "multiclass log loss",

        "secondary": [
            "accuracy",
            "macro F1",
            "multiclass Brier sum_j",
            "equal-match log loss",
            "per-zone recall/support",
            "confidence calibration",
        ],

        "top2_accuracy":
            None,

        "top2_note":
            (
                "Undefined for this deterministic probability "
                "adapter because all 14 non-predicted classes "
                "have exactly tied probability."
            ),

        "paired_comparator":
            "frozen B0 OOF predictions",

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
        load_json(CONTRACT) == contract,
        "Existing B1 V2 execution contract differs.",
    )
else:
    CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
        )
        + "\n"
    )


print("V3 B1 CONSTANT MOTION")
print()
print("Frozen inputs verified.")
print("Execution contract V2 frozen before scoring.")


# ============================================================
# Load datasets
# ============================================================

targets = pl.read_parquet(TARGETS)
splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)
spatial = pl.read_parquet(SPATIAL)
motion = pl.read_parquet(MOTION)
b0 = pl.read_parquet(B0)


require(
    targets.height == 29_065,
    f"Target rows changed: {targets.height}",
)

require(
    b0.height == targets.height,
    "B0 row count differs from targets.",
)

require(
    spatial.height == 774_111,
    f"Spatial rows changed: {spatial.height}",
)

require(
    spatial["demo_filename"].n_unique() == 53,
    "Spatial cache must contain 53 matches.",
)

require(
    spatial["place"].n_unique() == 23,
    "Spatial cache must contain 23 places.",
)

require(
    motion.height == 14_869,
    f"Motion rows changed: {motion.height}",
)

require(
    motion.filter(
        pl.col("status") != "RESOLVED"
    ).height == 0,
    "Motion cache contains unresolved rows.",
)


# Verify frozen target class encoding.
bad_target_encoding = (
    targets
    .with_columns(
        pl.col("target_macro_zone")
        .replace_strict(
            zone_to_index
        )
        .alias("_expected_target_index")
    )
    .filter(
        pl.col("target_class_index")
        != pl.col("_expected_target_index")
    )
)

require(
    bad_target_encoding.height == 0,
    "Target class indices do not match frozen zone order.",
)


# ============================================================
# Key uniqueness
# ============================================================

target_key = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]

motion_key = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]


require(
    targets.unique(
        subset=target_key
    ).height
    == targets.height,
    "Target evaluation key is not unique.",
)

require(
    b0.unique(
        subset=target_key
    ).height
    == b0.height,
    "B0 evaluation key is not unique.",
)

require(
    motion.unique(
        subset=motion_key
    ).height
    == motion.height,
    "Motion key is not unique.",
)


# ============================================================
# Join folds and causal motion
# ============================================================

spatial = spatial.join(
    splits.select([
        "demo_filename",
        "fold",
    ]),
    on="demo_filename",
    how="left",
)

require(
    spatial["fold"].null_count() == 0,
    "Spatial sample missing fold.",
)


motion_small = (
    motion
    .select([
        *motion_key,
        "prior_tick",
        "prior_source",
        "elapsed_sec",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ])
)


data = (
    targets
    .join(
        motion_small,
        on=motion_key,
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


for col in [
    "fold",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
]:
    require(
        data[col].null_count() == 0,
        f"Joined B1 field has nulls: {col}",
    )


# ============================================================
# Prediction helper
# ============================================================

def hard_predict(
    frame,
    horizon,
    tree,
    train_places,
):
    xyz = np.column_stack([
        frame["current_bomb_X"].to_numpy(),
        frame["current_bomb_Y"].to_numpy(),
        frame["current_bomb_Z"].to_numpy(),
    ]).astype(np.float64, copy=False)

    velocity = np.column_stack([
        frame["velocity_X"].to_numpy(),
        frame["velocity_Y"].to_numpy(),
        frame["velocity_Z"].to_numpy(),
    ]).astype(np.float64, copy=False)

    projected = (
        xyz
        + float(horizon) * velocity
    )

    require(
        np.isfinite(projected).all(),
        "Non-finite projected XYZ.",
    )

    distance, nn_index = tree.query(
        projected,
        k=1,
        workers=-1,
    )

    pred_place = train_places[
        nn_index
    ]

    pred_zone = np.asarray(
        [
            place_to_zone[str(x)]
            for x in pred_place
        ],
        dtype=object,
    )

    pred_index = np.asarray(
        [
            zone_to_index[str(x)]
            for x in pred_zone
        ],
        dtype=np.int64,
    )

    return (
        projected,
        np.asarray(
            distance,
            dtype=np.float64,
        ),
        pred_place,
        pred_zone,
        pred_index,
    )


# ============================================================
# 5-fold OOF B1
# ============================================================

prediction_frames = []
fold_records = []


for fold in range(N_FOLDS):

    resolver_train = spatial.filter(
        pl.col("fold") != fold
    )

    train_xyz = np.column_stack([
        resolver_train["X"].to_numpy(),
        resolver_train["Y"].to_numpy(),
        resolver_train["Z"].to_numpy(),
    ]).astype(np.float64, copy=False)

    train_places = np.asarray(
        resolver_train["place"].to_list(),
        dtype=object,
    )

    require(
        len(train_xyz) > 0,
        f"Empty resolver training set fold={fold}",
    )

    tree = cKDTree(train_xyz)

    print()
    print(
        f"FOLD {fold} "
        f"resolver_train={resolver_train.height:,}"
    )

    for horizon in HORIZONS:

        horizon_data = data.filter(
            pl.col("horizon_sec")
            == horizon
        )

        train = horizon_data.filter(
            pl.col("fold") != fold
        )

        valid = horizon_data.filter(
            pl.col("fold") == fold
        )

        require(
            train.height > 0
            and valid.height > 0,
            "Empty train/validation fold.",
        )

        (
            _,
            _,
            _,
            _,
            train_pred_index,
        ) = hard_predict(
            train,
            horizon,
            tree,
            train_places,
        )

        train_true = (
            train["target_class_index"]
            .to_numpy()
            .astype(np.int64)
        )

        n_correct = int(
            np.sum(
                train_pred_index
                == train_true
            )
        )

        q = (
            n_correct + 1
        ) / (
            train.height + 2
        )

        other_p = (
            1.0 - q
        ) / 14.0

        (
            projected,
            nn_distance,
            pred_place,
            pred_zone,
            pred_index,
        ) = hard_predict(
            valid,
            horizon,
            tree,
            train_places,
        )

        true_index = (
            valid["target_class_index"]
            .to_numpy()
            .astype(np.int64)
        )

        n_valid = valid.height

        probs = np.full(
            (n_valid, N_CLASSES),
            other_p,
            dtype=np.float64,
        )

        probs[
            np.arange(n_valid),
            pred_index,
        ] = q

        require(
            np.allclose(
                probs.sum(axis=1),
                1.0,
                atol=1e-12,
                rtol=0.0,
            ),
            "Probability rows do not sum to 1.",
        )

        true_p = probs[
            np.arange(n_valid),
            true_index,
        ]

        row_ll = -np.log(true_p)

        row_brier = (
            np.sum(
                probs ** 2,
                axis=1,
            )
            - 2.0 * true_p
            + 1.0
        )

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
            "prior_tick",
            "prior_source",
            "velocity_X",
            "velocity_Y",
            "velocity_Z",
        ]).to_dict(
            as_series=False
        )

        base.update({
            "predicted_fine_place":
                pred_place.tolist(),

            "predicted_macro_zone":
                pred_zone.tolist(),

            "predicted_class_index":
                pred_index.tolist(),

            "projected_X":
                projected[:, 0].tolist(),

            "projected_Y":
                projected[:, 1].tolist(),

            "projected_Z":
                projected[:, 2].tolist(),

            "resolver_neighbor_distance":
                nn_distance.tolist(),

            "train_accuracy_q":
                [float(q)] * n_valid,

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
            ] = probs[
                :, class_idx
            ].tolist()

        prediction_frames.append(
            pl.DataFrame(base)
        )

        valid_acc = float(
            np.mean(
                pred_index
                == true_index
            )
        )

        fold_records.append({
            "fold":
                fold,

            "horizon_sec":
                horizon,

            "resolver_training_samples":
                resolver_train.height,

            "n_train":
                train.height,

            "n_validation":
                valid.height,

            "training_hard_accuracy":
                n_correct / train.height,

            "train_accuracy_q":
                q,

            "validation_hard_accuracy":
                valid_acc,

            "validation_log_loss":
                float(row_ll.mean()),

            "neighbor_distance_mean":
                float(
                    nn_distance.mean()
                ),

            "neighbor_distance_p95":
                float(
                    np.quantile(
                        nn_distance,
                        0.95,
                    )
                ),
        })

        print(
            f"  +{horizon}s "
            f"train={train.height:,} "
            f"valid={valid.height:,} "
            f"q={q:.6f} "
            f"acc={valid_acc:.4%} "
            f"LL={row_ll.mean():.6f} "
            f"NNp95="
            f"{np.quantile(nn_distance, 0.95):.2f}"
        )


pred = pl.concat(
    prediction_frames,
    how="vertical",
)


require(
    pred.height == targets.height,
    (
        f"OOF rows {pred.height} "
        f"!= target rows {targets.height}"
    ),
)

require(
    pred.unique(
        subset=target_key
    ).height
    == pred.height,
    "B1 OOF key is not unique.",
)


# ============================================================
# Exact paired B0 comparison
# ============================================================

b0_small = b0.select([
    *target_key,

    pl.col(
        "predicted_class_index"
    ).alias(
        "b0_predicted_class_index"
    ),

    pl.col(
        "log_loss"
    ).alias(
        "b0_log_loss"
    ),

    pl.col(
        "multiclass_brier"
    ).alias(
        "b0_multiclass_brier"
    ),
])


pred = (
    pred
    .join(
        b0_small,
        on=target_key,
        how="left",
    )
    .with_columns([
        (
            pl.col("log_loss")
            - pl.col("b0_log_loss")
        ).alias(
            "delta_ll_b1_minus_b0"
        ),

        (
            pl.col("multiclass_brier")
            - pl.col("b0_multiclass_brier")
        ).alias(
            "delta_brier_b1_minus_b0"
        ),
    ])
)


require(
    pred["b0_log_loss"].null_count()
    == 0,
    "B0/B1 paired alignment failed.",
)


# ============================================================
# Aggregate metrics
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

    y_b0 = (
        p["b0_predicted_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    per_match = (
        p
        .group_by("demo_filename")
        .agg([
            pl.col("log_loss")
            .mean()
            .alias("b1_ll"),

            pl.col("b0_log_loss")
            .mean()
            .alias("b0_ll"),

            pl.col("delta_ll_b1_minus_b0")
            .mean()
            .alias("delta_ll"),
        ])
    )

    match_delta = (
        per_match["delta_ll"]
        .to_numpy()
        .astype(np.float64)
    )

    require(
        len(match_delta) == 53,
        "Expected 53 match-level deltas.",
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + horizon
    )

    indices = rng.integers(
        0,
        len(match_delta),
        size=(
            BOOTSTRAP_REPS,
            len(match_delta),
        ),
    )

    boot = match_delta[
        indices
    ].mean(axis=1)

    ci_low, ci_high = np.quantile(
        boot,
        [0.025, 0.975],
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


    fold_calibration = [
        {
            "fold":
                r["fold"],

            "n":
                r["n_validation"],

            "mean_confidence_q":
                r["train_accuracy_q"],

            "observed_accuracy":
                r[
                    "validation_hard_accuracy"
                ],
        }
        for r in fold_records
        if r["horizon_sec"] == horizon
    ]

    calibration_ece = (
        sum(
            x["n"]
            * abs(
                x["mean_confidence_q"]
                - x["observed_accuracy"]
            )
            for x in fold_calibration
        )
        / p.height
    )


    moving = p.filter(
        pl.col("stay_same_zone")
        == False
    )

    staying = p.filter(
        pl.col("stay_same_zone")
        == True
    )


    predicted_change_rate = float(
        (
            p["predicted_class_index"]
            != p["current_class_index"]
        ).mean()
    )


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
                per_match["b1_ll"].mean()
            ),

        "multiclass_brier":
            float(
                p["multiclass_brier"]
                .mean()
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
                        range(N_CLASSES)
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top2_accuracy":
            None,

        "top2_note":
            (
                "Undefined because all "
                "non-predicted classes tie."
            ),

        "per_zone":
            per_zone,

        "calibration": {
            "fold_bins":
                fold_calibration,

            "fold_weighted_ece":
                float(
                    calibration_ece
                ),
        },

        "predicted_change_rate":
            predicted_change_rate,

        "moving_rows":
            moving.height,

        "moving_accuracy":
            float(
                (
                    moving["predicted_class_index"]
                    ==
                    moving["target_class_index"]
                ).mean()
            ),

        "staying_rows":
            staying.height,

        "staying_accuracy":
            float(
                (
                    staying["predicted_class_index"]
                    ==
                    staying["target_class_index"]
                ).mean()
            ),

        "resolver_neighbor_distance_mean":
            float(
                p[
                    "resolver_neighbor_distance"
                ].mean()
            ),

        "resolver_neighbor_distance_p95":
            float(
                p[
                    "resolver_neighbor_distance"
                ].quantile(0.95)
            ),

        "b0_log_loss_same_rows":
            float(
                p["b0_log_loss"].mean()
            ),

        "delta_ll_b1_minus_b0":
            float(
                p[
                    "delta_ll_b1_minus_b0"
                ].mean()
            ),

        "equal_match_delta_ll_b1_minus_b0":
            float(
                per_match["delta_ll"].mean()
            ),

        "equal_match_delta_ll_bootstrap_95ci":
            [
                float(ci_low),
                float(ci_high),
            ],

        "delta_brier_b1_minus_b0":
            float(
                p[
                    "delta_brier_b1_minus_b0"
                ].mean()
            ),

        "accuracy_delta_b1_minus_b0":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
                -
                accuracy_score(
                    y_true,
                    y_b0,
                )
            ),
    }

    results_by_horizon[
        str(horizon)
    ] = result


# ============================================================
# Save outputs
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
        "B1_CONSTANT_MOTION",

    "status":
        "DEVELOPMENT_OOF_COMPLETE",

    "evaluation":
        "5-fold grouped-by-match OOF development",

    "target_dataset_sha256":
        sha256(TARGETS),

    "cv_split_sha256":
        sha256(SPLITS),

    "spatial_cache_sha256":
        sha256(SPATIAL),

    "motion_cache_sha256":
        sha256(MOTION),

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
# Final console summary
# ============================================================

print()
print("B1 CONSTANT MOTION — OOF DEVELOPMENT RESULTS")
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
        f"  predicted change rate: "
        f"{r['predicted_change_rate']:.6%}"
    )

    print(
        f"  moving accuracy: "
        f"{r['moving_accuracy']:.6%}"
    )

    print(
        f"  staying accuracy: "
        f"{r['staying_accuracy']:.6%}"
    )

    print(
        f"  NN p95: "
        f"{r['resolver_neighbor_distance_p95']:.3f}"
    )

    print(
        f"  calibration ECE: "
        f"{r['calibration']['fold_weighted_ece']:.6f}"
    )

    print()

    print(
        f"  B0 LL same rows: "
        f"{r['b0_log_loss_same_rows']:.6f}"
    )

    print(
        f"  ΔLL B1-B0: "
        f"{r['delta_ll_b1_minus_b0']:+.6f}"
    )

    print(
        f"  equal-match ΔLL: "
        f"{r['equal_match_delta_ll_b1_minus_b0']:+.6f}"
    )

    ci = (
        r[
            "equal_match_delta_ll_bootstrap_95ci"
        ]
    )

    print(
        "  match-bootstrap 95% CI: "
        f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
    )

    print(
        f"  ΔBrier: "
        f"{r['delta_brier_b1_minus_b0']:+.6f}"
    )

    print(
        f"  ΔAccuracy: "
        f"{r['accuracy_delta_b1_minus_b0']:+.6%}"
    )

    print()


print("B1_OOF_COMPLETE")
print("NO_V2_D_CONFIRM_USED")
