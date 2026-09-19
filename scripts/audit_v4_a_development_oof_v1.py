"""V4-A Gate 4B: independently audit saved development OOF predictions.

Read-only. No model fitting, demo parsing, or confirmation access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


PROTOCOL = Path(
    "docs/v4_a_model_experiment_protocol_v1_frozen.json"
)

REPORT = Path(
    "docs/v4_a_development_oof_results_v1.json"
)

OOF = Path(
    "data/interim/v4_a_control_vs_team_oof_v1.parquet"
)

TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

SPLITS = Path(
    "docs/v3_dev_cv_splits_v1.csv"
)

HISTORICAL_B3 = Path(
    "docs/v3_b3_tabular_map_aware_results.json"
)

KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


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


def load_json(path):
    return json.loads(path.read_text())


def close(actual, expected, label, atol=1e-8):
    require(
        np.isclose(
            actual,
            expected,
            rtol=0.0,
            atol=atol,
        ),
        f"{label}: expected {expected}, got {actual}",
    )


print("\n=== V4-A GATE 4B: DEVELOPMENT OOF AUDIT ===")

for path in (
    PROTOCOL,
    REPORT,
    OOF,
    TARGETS,
    SPLITS,
    HISTORICAL_B3,
):
    require(
        path.is_file(),
        f"Missing input: {path}",
    )

protocol = load_json(PROTOCOL)
report = load_json(REPORT)
historical = load_json(HISTORICAL_B3)

require(
    protocol["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "Unexpected experiment protocol status.",
)

require(
    report["status"] == "DEVELOPMENT_OOF_COMPLETE",
    "Development OOF report is incomplete.",
)

require(
    report["experiment_protocol_sha256"] == sha256(PROTOCOL),
    "Result report references another experiment protocol.",
)

require(
    report["oof_predictions_sha256"] == sha256(OOF),
    "OOF parquet SHA256 mismatch.",
)

require(
    report["input_provenance"]["targets_sha256"]
    == sha256(TARGETS),
    "Frozen development targets changed.",
)

require(
    report["input_provenance"]["cv_splits_sha256"]
    == sha256(SPLITS),
    "Frozen CV splits changed.",
)

require(
    report["feature_dimensions"]
    == {"control": 24, "candidate": 56},
    "Unexpected model dimensions.",
)

require(
    report["xgboost_configuration"]
    == protocol["model_configuration"]["xgboost_config"],
    "Model configuration differs from frozen protocol.",
)

print("Artifact identities: PASS")


# ------------------------------------------------------------
# 1. Verify exact OOF row coverage and CV assignments.
# ------------------------------------------------------------

pred = pl.read_parquet(OOF)

targets = pl.read_parquet(
    TARGETS,
    columns=KEY + ["target_class_index"],
)

splits = pl.read_csv(
    SPLITS,
    infer_schema_length=None,
)

require(
    pred.height == targets.height == 29065,
    "Unexpected OOF/target row count.",
)

require(
    pred.select(KEY).unique().height == pred.height,
    "Duplicate OOF prediction keys.",
)

require(
    targets.select(KEY).unique().height == targets.height,
    "Duplicate target keys.",
)

require(
    pred["demo_filename"].n_unique() == 53,
    "Unexpected OOF match coverage.",
)

require(
    pred.select(KEY).join(
        targets.select(KEY),
        on=KEY,
        how="anti",
    ).height == 0,
    "OOF contains unexpected target keys.",
)

require(
    targets.select(KEY).join(
        pred.select(KEY),
        on=KEY,
        how="anti",
    ).height == 0,
    "Some frozen target keys lack OOF predictions.",
)

checked = pred.join(
    targets.rename({
        "target_class_index": "_frozen_target_class_index",
    }),
    on=KEY,
    how="left",
    validate="1:1",
)

require(
    checked["_frozen_target_class_index"].null_count() == 0,
    "A frozen target label is missing.",
)

require(
    (
        checked["target_class_index"]
        == checked["_frozen_target_class_index"]
    ).all(),
    "OOF target labels differ from frozen targets.",
)

require(
    splits["demo_filename"].n_unique() == splits.height == 53,
    "Unexpected CV split coverage.",
)

checked = checked.join(
    splits.select([
        "demo_filename",
        pl.col("fold").alias("_frozen_fold"),
    ]),
    on="demo_filename",
    how="left",
    validate="m:1",
)

require(
    checked["_frozen_fold"].null_count() == 0,
    "Missing frozen fold assignment.",
)

require(
    (checked["fold"] == checked["_frozen_fold"]).all(),
    "OOF fold assignments differ from frozen CV splits.",
)

require(
    len(report["fold_records"]) == 10,
    "Expected ten horizon/fold records.",
)

print("Exact target and grouped-fold pairing: PASS")


# ------------------------------------------------------------
# 2. Recompute per-row probability metrics.
# ------------------------------------------------------------

classes = protocol["scope"]["target_classes"]

require(
    len(classes) == 15,
    "Unexpected class order.",
)

y = checked["target_class_index"].to_numpy().astype(np.int64)

require(
    ((y >= 0) & (y < 15)).all(),
    "Invalid target class index.",
)

for name in ("control", "candidate"):

    probability_columns = [
        f"{name}_p_{zone}"
        for zone in classes
    ]

    require(
        all(column in checked.columns
            for column in probability_columns),
        f"Missing {name} probability columns.",
    )

    probabilities = checked.select(
        probability_columns
    ).to_numpy().astype(np.float64)

    require(
        probabilities.shape == (29065, 15),
        f"Unexpected {name} probability matrix shape.",
    )

    require(
        np.isfinite(probabilities).all()
        and (probabilities >= 0).all(),
        f"Invalid {name} probabilities.",
    )

    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ),
        f"{name} probabilities do not sum to one.",
    )

    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    require(
        np.array_equal(
            predicted,
            checked[
                f"{name}_predicted_class_index"
            ].to_numpy(),
        ),
        f"{name} hard predictions differ from probability argmax.",
    )

    true_probability = probabilities[
        np.arange(len(y)),
        y,
    ]

    require(
        (true_probability > 0).all(),
        f"{name} has a zero true-class probability.",
    )

    recalculated_ll = -np.log(
        true_probability
    )

    stored_ll = checked[
        f"{name}_log_loss"
    ].to_numpy()

    require(
        np.allclose(
            recalculated_ll,
            stored_ll,
            rtol=0.0,
            atol=1e-7,
        ),
        f"{name} saved per-row Log Loss does not match probabilities.",
    )

    recalculated_brier = (
        np.sum(probabilities**2, axis=1)
        - 2.0 * true_probability
        + 1.0
    )

    require(
        np.allclose(
            recalculated_brier,
            checked[
                f"{name}_multiclass_brier"
            ].to_numpy(),
            rtol=0.0,
            atol=1e-6,
        ),
        f"{name} saved per-row Brier scores are inconsistent.",
    )

    print(f"{name}: probability and row-metric checks PASS")


# ------------------------------------------------------------
# 3. Recompute paired horizon-level results.
# ------------------------------------------------------------

require(
    np.allclose(
        checked["candidate_log_loss"].to_numpy()
        - checked["control_log_loss"].to_numpy(),
        checked[
            "candidate_minus_control_log_loss"
        ].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    ),
    "Stored paired Log Loss differences are inconsistent.",
)

for horizon, expected_rows in ((5, 14869), (10, 14196)):

    part = checked.filter(
        pl.col("horizon_sec") == horizon
    )

    require(
        part.height == expected_rows
        and part["demo_filename"].n_unique() == 53,
        f"Unexpected +{horizon}s OOF coverage.",
    )

    recorded = report["results_by_horizon"][str(horizon)]

    control_ll = float(
        part["control_log_loss"].mean()
    )

    candidate_ll = float(
        part["candidate_log_loss"].mean()
    )

    delta = float(
        part["candidate_minus_control_log_loss"].mean()
    )

    close(
        control_ll,
        recorded["control"]["log_loss"],
        f"+{horizon}s Control LL",
    )

    close(
        candidate_ll,
        recorded["candidate"]["log_loss"],
        f"+{horizon}s Candidate LL",
    )

    close(
        delta,
        recorded["paired_comparison"]["row_weighted_mean"],
        f"+{horizon}s paired delta",
    )

    # Check that the newly fitted 24D control reproduces
    # the historical V3 B3 aggregate metric.

    close(
        control_ll,
        historical["results_by_horizon"][str(horizon)]["log_loss"],
        f"+{horizon}s historical B3 reproduction",
        atol=1e-6,
    )

    print(
        f"\n+{horizon}s:"
        f"\n  Control LL:   {control_ll:.6f}"
        f"\n  Candidate LL: {candidate_ll:.6f}"
        f"\n  Paired delta: {delta:+.6f}"
        "\n  Historical B3 aggregate reproduction: PASS"
    )

print("\n=== GATE 4B FINAL SUMMARY ===")
print("OOF rows: 29,065")
print("Development matches: 53")
print("Exact target/fold alignment: PASS")
print("Probability distributions: PASS")
print("Per-row Log Loss and Brier: PASS")
print("Paired horizon metrics: PASS")
print("Historical B3 aggregate reproduction: PASS")
print("OOF SHA256: PASS")

print("\nGATE_4B_OOF_ARTIFACT_VALIDATED")
print("No models were fitted or scored on confirmation data.")
print("No files were written or modified.")
