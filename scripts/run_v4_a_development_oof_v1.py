"""V4-A: controlled 24D versus 56D development OOF experiment.

Uses the frozen V4-A experiment protocol.

20 fits:
    2 horizons x 5 folds x 2 model variants.

No demo parsing, confirmation access, hyperparameter tuning,
or modification of frozen V3 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import xgboost
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


PROTOCOL_PATH = Path(
    "docs/v4_a_model_experiment_protocol_v1_frozen.json"
)

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

B3_FEATURE = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

V4_FEATURE = Path(
    "docs/v4_team_context_feature_contract_v1_frozen.json"
)

MODEL_PROTOCOL = Path(
    "docs/v3_b3_model_protocol_v1.json"
)

OUTPUT = Path(
    "data/interim/v4_a_control_vs_team_oof_v1.parquet"
)

RESULTS = Path(
    "docs/v4_a_development_oof_results_v1.json"
)

KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]

MOTION_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]

TEAM_KEY = [
    "demo_filename",
    "round_num",
    "current_tick",
]

N_CLASSES = 15
HORIZONS = (5, 10)
FOLDS = range(5)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def free_gib():
    free = shutil.disk_usage(".").free / 1024**3

    require(
        free >= 12,
        f"Free disk below 12 GiB: {free:.2f} GiB",
    )

    return free


# ============================================================
# 1. PRE-FIT VALIDATION
# ============================================================

print("\n=== V4-A DEVELOPMENT OOF EXPERIMENT ===")
print(f"Free disk: {free_gib():.2f} GiB")

for path in (
    PROTOCOL_PATH,
    TARGETS,
    MOTION,
    TEAM,
    SPLITS,
    B3_FEATURE,
    V4_FEATURE,
    MODEL_PROTOCOL,
):
    require(
        path.is_file(),
        f"Missing required input: {path}",
    )

for path in (OUTPUT, RESULTS):
    require(
        not path.exists(),
        f"Output already exists; refusing to overwrite: {path}",
    )

protocol = read_json(PROTOCOL_PATH)
b3_protocol = read_json(B3_FEATURE)
v4_protocol = read_json(V4_FEATURE)
parent_model = read_json(MODEL_PROTOCOL)

require(
    protocol["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "V4-A model experiment protocol is not frozen.",
)

require(
    sha256(TARGETS) == protocol["provenance"]["target_sha256"],
    "Frozen target SHA mismatch.",
)

require(
    sha256(MOTION) == protocol["provenance"]["motion_sha256"],
    "Frozen motion SHA mismatch.",
)

require(
    sha256(TEAM)
    == protocol["provenance"]["team_context_sha256"],
    "Frozen Team Context SHA mismatch.",
)

require(
    sha256(SPLITS)
    == protocol["provenance"]["cv_split_sha256"],
    "Frozen CV split SHA mismatch.",
)

require(
    sha256(B3_FEATURE)
    == protocol["provenance"]["b3_feature_protocol_sha256"],
    "Frozen B3 feature protocol SHA mismatch.",
)

require(
    sha256(V4_FEATURE)
    == protocol["provenance"]["v4_feature_protocol_sha256"],
    "Frozen V4 feature protocol SHA mismatch.",
)

require(
    sha256(MODEL_PROTOCOL)
    == protocol["model_configuration"]["source_sha256"],
    "Parent XGBoost model protocol SHA mismatch.",
)

require(
    protocol["model_configuration"]["xgboost_config"]
    == parent_model["xgboost_config"],
    "Frozen model configuration mismatch.",
)

require(
    protocol["control"]["feature_order"]
    == b3_protocol["feature_order"],
    "Control feature order mismatch.",
)

require(
    protocol["candidate"]["feature_order"]
    == (
        b3_protocol["feature_order"]
        + v4_protocol["team_context_feature_order"]
    ),
    "Candidate feature order mismatch.",
)

require(
    protocol["control"]["dimensions"] == 24
    and protocol["candidate"]["dimensions"] == 56,
    "Unexpected feature dimensions.",
)

require(
    len(protocol["scope"]["target_classes"]) == N_CLASSES,
    "Unexpected target class count.",
)

require(
    protocol["training"]["hyperparameter_search"] is False
    and protocol["training"]["early_stopping"] is False
    and protocol["training"]["use_confirmation_data"] is False,
    "Unexpected frozen training policy.",
)

CLASS_ORDER = protocol["scope"]["target_classes"]
TEAM_COLUMNS = v4_protocol["team_context_feature_order"]

ZONE_TO_INDEX = {
    zone: index
    for index, zone in enumerate(CLASS_ORDER)
}

XGB_CONFIG = dict(
    protocol["model_configuration"]["xgboost_config"]
)

print("Frozen input identities: PASS")
print("Frozen model configuration: PASS")
print("Installed XGBoost:", xgboost.__version__)


# ============================================================
# 2. BUILD THE SAME DEVELOPMENT DATA FOR BOTH MODELS
# ============================================================

targets = pl.read_parquet(TARGETS)

motion = pl.read_parquet(MOTION).select(
    MOTION_KEY
    + [
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "status",
    ]
)

team = pl.read_parquet(TEAM)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)

require(
    targets.height == 29065,
    "Unexpected target row count.",
)

require(
    targets.select(KEY).unique().height == targets.height,
    "Duplicate target key.",
)

require(
    motion.select(MOTION_KEY).unique().height == motion.height,
    "Duplicate motion key.",
)

require(
    team.select(TEAM_KEY).unique().height == team.height,
    "Duplicate Team Context key.",
)

require(
    team.columns == TEAM_KEY + TEAM_COLUMNS,
    "Unexpected Team Context schema.",
)

require(
    splits.height == 53
    and splits["demo_filename"].n_unique() == 53,
    "Unexpected development CV split coverage.",
)

require(
    set(splits["fold"].unique().to_list()) == set(FOLDS),
    "Expected five folds numbered 0–4.",
)

data = (
    targets
    .join(
        motion,
        on=MOTION_KEY,
        how="left",
        validate="m:1",
    )
    .join(
        team,
        on=TEAM_KEY,
        how="left",
        validate="m:1",
    )
    .join(
        splits.select(["demo_filename", "fold"]),
        on="demo_filename",
        how="left",
        validate="m:1",
    )
)

require(
    data.height == 29065,
    "Joining features changed target row count.",
)

require(
    data["fold"].null_count() == 0,
    "Missing CV fold.",
)

require(
    data["status"].null_count() == 0
    and data.filter(pl.col("status") != "RESOLVED").height == 0,
    "Unresolved causal motion.",
)

require(
    all(data[column].null_count() == 0 for column in TEAM_COLUMNS),
    "Missing Team Context feature.",
)

require(
    data["demo_filename"].n_unique() == 53,
    "Development match coverage mismatch.",
)

for horizon, expected_rows in ((5, 14869), (10, 14196)):
    part = data.filter(pl.col("horizon_sec") == horizon)

    require(
        part.height == expected_rows
        and part["demo_filename"].n_unique() == 53,
        f"Unexpected +{horizon}s target coverage.",
    )

    require(
        set(part["fold"].unique().to_list()) == set(FOLDS),
        f"Missing fold at +{horizon}s.",
    )

    require(
        set(part["target_class_index"].unique().to_list())
        == set(range(N_CLASSES)),
        f"Unexpected target class coverage at +{horizon}s.",
    )

print("Development data joins: PASS")
print("Same target rows and CV folds for both models: PASS")


# ============================================================
# 3. REPRODUCE FROZEN V3 B3 FEATURE CONSTRUCTION
# ============================================================

def build_control_features(frame):
    """Exact 24D B3 representation used by the V3 implementation."""

    n = frame.height

    X = np.zeros(
        (n, 24),
        dtype=np.float32,
    )

    current_zones = frame["current_macro_zone"].to_list()

    for row_index, zone in enumerate(current_zones):
        require(
            zone in ZONE_TO_INDEX,
            f"Unknown current macro-zone: {zone}",
        )

        X[row_index, ZONE_TO_INDEX[zone]] = 1.0

    sources = frame["current_source"].to_list()

    for row_index, source in enumerate(sources):
        if source in (
            "CARRIED_INVENTORY",
            "CARRIED_PICKUP_FALLBACK",
        ):
            X[row_index, 15] = 1.0

        elif source == "DROPPED":
            X[row_index, 16] = 1.0

        else:
            raise RuntimeError(
                f"STOP: Unexpected current_source: {source}"
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
        velocity.astype(np.float64),
        axis=1,
    ).astype(np.float32)

    X[:, 17:20] = xyz
    X[:, 20:23] = velocity
    X[:, 23] = speed

    require(
        X.shape == (n, 24)
        and np.isfinite(X).all(),
        "Invalid 24D Control feature matrix.",
    )

    return X


def build_candidate_features(frame, control):
    """Append the frozen 32D current-time Team Context."""

    team_matrix = np.column_stack([
        frame[column].to_numpy()
        for column in TEAM_COLUMNS
    ]).astype(
        np.float32,
        copy=False,
    )

    X = np.concatenate(
        [control, team_matrix],
        axis=1,
    )

    require(
        X.shape == (frame.height, 56)
        and np.isfinite(X).all(),
        "Invalid 56D Candidate feature matrix.",
    )

    require(
        np.array_equal(X[:, :24], control),
        "Candidate does not preserve the exact Control features.",
    )

    return X


# ============================================================
# 4. RUN MATCH-GROUPED OOF FITS
# ============================================================

def evaluate_predictions(probabilities, y_true):
    """Return per-row metrics with the same class ordering."""

    n = len(y_true)

    require(
        probabilities.shape == (n, N_CLASSES),
        "Unexpected probability matrix shape.",
    )

    require(
        np.isfinite(probabilities).all()
        and (probabilities >= 0).all(),
        "Invalid model probabilities.",
    )

    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ),
        "Predicted class probabilities do not sum to one.",
    )

    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    true_probability = probabilities[
        np.arange(n),
        y_true,
    ]

    require(
        (true_probability > 0).all(),
        "Zero true-class probability encountered.",
    )

    log_loss = -np.log(
        true_probability.astype(np.float64)
    )

    probabilities64 = probabilities.astype(np.float64)

    brier = (
        np.sum(probabilities64**2, axis=1)
        - 2.0 * true_probability.astype(np.float64)
        + 1.0
    )

    top2 = np.argsort(
        -probabilities,
        axis=1,
        kind="stable",
    )[:, :2]

    top2_correct = (
        top2 == y_true[:, None]
    ).any(axis=1)

    return {
        "predicted": predicted,
        "log_loss": log_loss,
        "brier": brier,
        "top2_correct": top2_correct,
        "confidence": probabilities.max(axis=1),
    }


prediction_frames = []
fold_records = []

print("\n=== BEGIN MODEL FITTING ===")
print("Horizon-specific models: 2")
print("Grouped folds per horizon: 5")
print("Model variants per fold: 2")
print("Total fits: 20")
print("Hyperparameter tuning: NO")
print("Confirmation data: NO")

for horizon in HORIZONS:

    horizon_data = data.filter(
        pl.col("horizon_sec") == horizon
    )

    X_control = build_control_features(horizon_data)

    X_candidate = build_candidate_features(
        horizon_data,
        X_control,
    )

    y = horizon_data["target_class_index"].to_numpy().astype(
        np.int64
    )

    fold_ids = horizon_data["fold"].to_numpy()

    print(
        f"\n=== +{horizon}s | {horizon_data.height:,} rows ===",
        flush=True,
    )

    for fold in FOLDS:

        free_gib()

        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)

        require(
            len(train_idx) > 0 and len(valid_idx) > 0,
            f"Empty train/validation split: +{horizon}s fold {fold}",
        )

        valid = horizon_data.filter(
            pl.col("fold") == fold
        )

        require(
            valid.height == len(valid_idx),
            "Validation rows are not aligned with feature matrices.",
        )

        train_matches = set(
            horizon_data.filter(
                pl.col("fold") != fold
            )["demo_filename"].unique().to_list()
        )

        valid_matches = set(
            valid["demo_filename"].unique().to_list()
        )

        require(
            train_matches.isdisjoint(valid_matches),
            f"Match leakage: +{horizon}s fold {fold}",
        )

        require(
            set(y[train_idx]) == set(range(N_CLASSES)),
            f"Training split lacks a target class: "
            f"+{horizon}s fold {fold}",
        )

        print(
            f"\n+{horizon}s | fold {fold} | "
            f"train={len(train_idx):,} "
            f"validation={len(valid_idx):,}",
            flush=True,
        )

        base = valid.select([
            "demo_filename",
            "round_num",
            "current_nominal_tick",
            "current_tick",
            "horizon_sec",
            "fold",
            "current_macro_zone",
            "target_macro_zone",
            "target_class_index",
            "stay_same_zone",
        ]).to_dict(as_series=False)

        fold_result = {
            "horizon_sec": horizon,
            "fold": fold,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx),
            "train_matches": len(train_matches),
            "validation_matches": len(valid_matches),
        }

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # Each model is fitted from scratch.
        # Both receive identical rows, labels, folds and parameters.

        for model_name, X in (
            ("control", X_control),
            ("candidate", X_candidate),
        ):

            print(
                f"  Fitting {model_name} ({X.shape[1]}D)...",
                flush=True,
            )

            model = XGBClassifier(
                **XGB_CONFIG
            )

            model.fit(
                X[train_idx],
                y_train,
            )

            probabilities = model.predict_proba(
                X[valid_idx]
            )

            metrics = evaluate_predictions(
                probabilities,
                y_valid,
            )

            base[f"{model_name}_predicted_class_index"] = (
                metrics["predicted"].tolist()
            )

            base[f"{model_name}_log_loss"] = (
                metrics["log_loss"].tolist()
            )

            base[f"{model_name}_multiclass_brier"] = (
                metrics["brier"].tolist()
            )

            base[f"{model_name}_top2_correct"] = (
                metrics["top2_correct"].tolist()
            )

            base[f"{model_name}_top1_confidence"] = (
                metrics["confidence"].tolist()
            )

            for class_index, zone in enumerate(CLASS_ORDER):
                base[f"{model_name}_p_{zone}"] = (
                    probabilities[:, class_index].tolist()
                )

            fold_result[f"{model_name}_log_loss"] = float(
                metrics["log_loss"].mean()
            )

            fold_result[f"{model_name}_accuracy"] = float(
                np.mean(metrics["predicted"] == y_valid)
            )

            fold_result[f"{model_name}_multiclass_brier"] = float(
                metrics["brier"].mean()
            )

            print(
                f"    LL={fold_result[f'{model_name}_log_loss']:.6f}"
                f" | Acc={fold_result[f'{model_name}_accuracy']:.4%}",
                flush=True,
            )

            del model

        fold_result["candidate_minus_control_log_loss"] = (
            fold_result["candidate_log_loss"]
            - fold_result["control_log_loss"]
        )

        fold_records.append(fold_result)

        prediction_frames.append(
            pl.DataFrame(base)
        )


# ============================================================
# 5. VERIFY COMPLETE OOF COVERAGE
# ============================================================

predictions = pl.concat(
    prediction_frames,
    how="vertical",
)

require(
    predictions.height == targets.height == 29065,
    "OOF prediction row count mismatch.",
)

require(
    predictions.select(KEY).unique().height
    == predictions.height,
    "Duplicate OOF prediction key.",
)

require(
    predictions["demo_filename"].n_unique() == 53,
    "OOF match coverage mismatch.",
)

require(
    predictions.select(KEY).join(
        targets.select(KEY),
        on=KEY,
        how="anti",
    ).height == 0,
    "OOF predictions contain unexpected target keys.",
)

require(
    targets.select(KEY).join(
        predictions.select(KEY),
        on=KEY,
        how="anti",
    ).height == 0,
    "Some frozen target keys lack OOF predictions.",
)

require(
    len(fold_records) == 10,
    "Expected ten horizon/fold records.",
)

predictions = predictions.with_columns(
    (
        pl.col("candidate_log_loss")
        - pl.col("control_log_loss")
    ).alias("candidate_minus_control_log_loss")
)

print("\nComplete paired OOF coverage: PASS")


# ============================================================
# 6. AGGREGATE METRICS
# ============================================================

def ece_10_bins(frame, prefix):
    confidence = frame[
        f"{prefix}_top1_confidence"
    ].to_numpy()

    correct = (
        frame[f"{prefix}_predicted_class_index"].to_numpy()
        == frame["target_class_index"].to_numpy()
    )

    edges = np.linspace(0.0, 1.0, 11)
    total = len(confidence)

    ece = 0.0
    bins = []

    for i in range(10):

        lo = edges[i]
        hi = edges[i + 1]

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

        count = int(mask.sum())

        if count == 0:
            bins.append({
                "bin": i,
                "n": 0,
                "mean_confidence": None,
                "accuracy": None,
            })
            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        accuracy = float(
            correct[mask].mean()
        )

        ece += (
            count / total
        ) * abs(mean_confidence - accuracy)

        bins.append({
            "bin": i,
            "n": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })

    return {
        "top1_ece": float(ece),
        "bins": bins,
    }


def aggregate_model(frame, prefix):
    y_true = frame["target_class_index"].to_numpy()
    y_pred = frame[
        f"{prefix}_predicted_class_index"
    ].to_numpy()

    per_match = (
        frame
        .group_by("demo_filename")
        .agg(
            pl.col(f"{prefix}_log_loss")
            .mean()
            .alias("match_mean_log_loss")
        )
    )

    require(
        per_match.height == 53,
        "Expected 53 match-level results.",
    )

    per_zone = []

    for index, zone in enumerate(CLASS_ORDER):
        mask = y_true == index
        support = int(mask.sum())

        per_zone.append({
            "zone": zone,
            "support": support,
            "recall": (
                float(np.mean(y_pred[mask] == index))
                if support
                else None
            ),
        })

    return {
        "rows": frame.height,
        "matches": 53,
        "log_loss": float(
            frame[f"{prefix}_log_loss"].mean()
        ),
        "equal_match_log_loss": float(
            per_match["match_mean_log_loss"].mean()
        ),
        "multiclass_brier": float(
            frame[f"{prefix}_multiclass_brier"].mean()
        ),
        "accuracy": float(
            np.mean(y_true == y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "top2_accuracy": float(
            frame[f"{prefix}_top2_correct"].mean()
        ),
        "calibration": ece_10_bins(frame, prefix),
        "per_zone": per_zone,
    }


results_by_horizon = {}

for horizon in HORIZONS:

    part = predictions.filter(
        pl.col("horizon_sec") == horizon
    )

    control_metrics = aggregate_model(
        part,
        "control",
    )

    candidate_metrics = aggregate_model(
        part,
        "candidate",
    )

    # Bootstrap MATCH-LEVEL paired differences, not individual rows.
    match_deltas = (
        part
        .group_by("demo_filename")
        .agg(
            pl.col("candidate_minus_control_log_loss")
            .mean()
            .alias("delta")
        )
        .sort("demo_filename")
    )

    require(
        match_deltas.height == 53,
        "Expected 53 paired match differences.",
    )

    delta_values = match_deltas["delta"].to_numpy()

    rng = np.random.default_rng(
        protocol["evaluation"]["bootstrap_seed"]
    )

    indices = rng.integers(
        0,
        len(delta_values),
        size=(
            protocol["evaluation"]["bootstrap_repetitions"],
            len(delta_values),
        ),
    )

    bootstrap_means = delta_values[
        indices
    ].mean(axis=1)

    ci = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    row_delta = float(
        part["candidate_minus_control_log_loss"].mean()
    )

    equal_match_delta = float(
        delta_values.mean()
    )

    results_by_horizon[str(horizon)] = {
        "control": control_metrics,
        "candidate": candidate_metrics,
        "paired_comparison": {
            "metric": "candidate_minus_control_log_loss",
            "row_weighted_mean": row_delta,
            "equal_match_mean": equal_match_delta,
            "equal_match_bootstrap_95ci": [
                float(ci[0]),
                float(ci[1]),
            ],
            "bootstrap_unit": "match",
            "bootstrap_repetitions": (
                protocol["evaluation"]["bootstrap_repetitions"]
            ),
            "bootstrap_seed": (
                protocol["evaluation"]["bootstrap_seed"]
            ),
        },
    }

    print(
        f"\n+{horizon}s development OOF:"
        f"\n  Control LL:   {control_metrics['log_loss']:.6f}"
        f"\n  Candidate LL: {candidate_metrics['log_loss']:.6f}"
        f"\n  Candidate - Control: {row_delta:+.6f}"
        f"\n  Equal-match delta:  {equal_match_delta:+.6f}"
        f"\n  Match-bootstrap 95% CI:"
        f" [{ci[0]:+.6f}, {ci[1]:+.6f}]",
        flush=True,
    )


# ============================================================
# 7. WRITE NEW V4-ONLY OUTPUTS
# ============================================================

require(
    not OUTPUT.exists() and not RESULTS.exists(),
    "An output appeared during execution; refusing to overwrite.",
)

free_gib()

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

predictions = predictions.sort(KEY)

predictions.write_parquet(OUTPUT)

report = {
    "version": "V4_A_DEVELOPMENT_OOF_RESULTS_V1",
    "status": "DEVELOPMENT_OOF_COMPLETE",
    "experiment_protocol": str(PROTOCOL_PATH),
    "experiment_protocol_sha256": sha256(PROTOCOL_PATH),
    "feature_dimensions": {
        "control": 24,
        "candidate": 56,
    },
    "model_family": "XGBClassifier",
    "installed_xgboost_version": xgboost.__version__,
    "xgboost_configuration": XGB_CONFIG,
    "development_matches": 53,
    "development_target_rows": 29065,
    "horizons_sec": list(HORIZONS),
    "fold_records": fold_records,
    "results_by_horizon": results_by_horizon,
    "oof_predictions": str(OUTPUT),
    "oof_predictions_sha256": sha256(OUTPUT),
    "input_provenance": {
        "targets_sha256": sha256(TARGETS),
        "motion_sha256": sha256(MOTION),
        "team_context_sha256": sha256(TEAM),
        "cv_splits_sha256": sha256(SPLITS),
        "b3_feature_protocol_sha256": sha256(B3_FEATURE),
        "v4_feature_protocol_sha256": sha256(V4_FEATURE),
        "v3_b3_model_protocol_sha256": sha256(MODEL_PROTOCOL),
    },
    "scientific_boundaries": {
        "development_only": True,
        "v2_confirmation_used": False,
        "v3_confirmation_used": False,
        "fresh_v4_confirmation_used": False,
        "hyperparameter_search_performed": False,
        "frozen_v3_artifacts_modified": False,
        "historical_v3_b3_model_modified": False,
    },
    "interpretation": (
        "Paired grouped development OOF comparison only. "
        "This is not independent V4 confirmation evidence."
    ),
}

with RESULTS.open("x") as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )
    file.write("\n")


print("\n" + "=" * 64)
print("V4-A DEVELOPMENT OOF — FINAL SUMMARY")
print("=" * 64)

print("Development matches: 53")
print("Target rows:", predictions.height)
print("Control dimensions: 24")
print("Candidate dimensions: 56")
print("Completed model fits: 20")
print("OOF artifact:", OUTPUT)
print("OOF SHA256:", sha256(OUTPUT))
print("Results:", RESULTS)
print(f"Free disk: {free_gib():.2f} GiB")

print("\nV4_A_DEVELOPMENT_OOF_COMPLETE")
print("No confirmation data was accessed.")
print("No frozen V3 research artifacts were modified.")
