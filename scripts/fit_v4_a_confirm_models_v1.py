"""V4-A Gate 6B: fit and freeze four full-development models.

Uses the frozen V4-A confirmation fit contract.
Development data only. No confirmation acquisition or scoring.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost
from xgboost import XGBClassifier


CONTRACT = Path("docs/v4_a_confirm_fit_contract_v1_frozen.json")
CLOSURE = Path("docs/v4_a_development_closure_v1.json")
EXPERIMENT = Path("docs/v4_a_model_experiment_protocol_v1_frozen.json")
FEATURE = Path("docs/v4_team_context_feature_contract_v1_frozen.json")
PARENT_MODEL = Path("docs/v3_b3_model_protocol_v1.json")

TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
MOTION = Path("data/interim/v3_b1_motion_inputs_v1.parquet")
TEAM = Path("data/interim/v4_dev_team_context_v1.parquet")

THIS_SCRIPT = Path("scripts/fit_v4_a_confirm_models_v1.py")

OUT_DIR = Path("artifacts/v4_a_confirm_frozen")
TMP_DIR = Path("artifacts/.v4_a_confirm_fit_tmp")
FREEZE = Path("docs/v4_a_confirm_model_freeze_v1.json")

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

EXPECTED_ROWS = {5: 14869, 10: 14196}


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


def free_gib():
    free = shutil.disk_usage(".").free / 1024**3
    require(free >= 12, f"Free disk below 12 GiB: {free:.2f}")
    return free


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


print("\n=== V4-A GATE 6B: FINAL MODEL FIT ===")
print(f"Free disk: {free_gib():.2f} GiB")

check_only = sys.argv[1:] == ["--check-only"]

require(
    sys.argv[1:] in ([], ["--check-only"]),
    "Usage: fit_v4_a_confirm_models_v1.py [--check-only]",
)

inputs = {
    "contract": CONTRACT,
    "closure": CLOSURE,
    "experiment_protocol": EXPERIMENT,
    "feature_contract": FEATURE,
    "parent_model_protocol": PARENT_MODEL,
    "targets": TARGETS,
    "motion": MOTION,
    "team_context": TEAM,
    "fit_script": THIS_SCRIPT,
}

for name, path in inputs.items():
    require(path.is_file(), f"Missing {name}: {path}")

require(
    not FREEZE.exists(),
    f"Freeze record already exists: {FREEZE}",
)

require(
    not TMP_DIR.exists(),
    f"Temporary fit directory already exists: {TMP_DIR}. "
    "Inspect it before any retry.",
)

contract = read_json(CONTRACT)
closure = read_json(CLOSURE)
experiment = read_json(EXPERIMENT)
feature = read_json(FEATURE)
parent_model = read_json(PARENT_MODEL)

require(
    contract["status"] == "FROZEN_BEFORE_V4_CONFIRMATION_ACCESS",
    "Final fit contract is not frozen.",
)

require(
    closure["status"]
    == "DEVELOPMENT_CLOSED_NOT_INDEPENDENTLY_CONFIRMED",
    "V4-A development closure status mismatch.",
)

require(
    experiment["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "V4-A experiment protocol status mismatch.",
)

require(
    feature["status"] == "FROZEN_FOR_V4_A_DEVELOPMENT",
    "Team Context feature contract status mismatch.",
)

source_paths = {
    "development_closure": CLOSURE,
    "experiment_protocol": EXPERIMENT,
    "feature_contract": FEATURE,
    "parent_model_protocol": PARENT_MODEL,
    "targets": TARGETS,
    "motion": MOTION,
    "team_context": TEAM,
}

for name, path in source_paths.items():
    require(
        sha256(path) == contract["source_sha256"][name],
        f"Frozen source SHA256 mismatch: {name}",
    )

config = contract["fitting"]["xgboost_config"]
class_order = contract["development_population"]["target_classes"]
control_columns = contract["model_variants"]["control"]["feature_order"]
candidate_columns = contract["model_variants"]["candidate"]["feature_order"]
team_columns = feature["team_context_feature_order"]

require(
    config == experiment["model_configuration"]["xgboost_config"]
    == parent_model["xgboost_config"],
    "XGBoost configuration mismatch.",
)

require(
    control_columns == experiment["control"]["feature_order"],
    "Control feature order mismatch.",
)

require(
    candidate_columns == control_columns + team_columns
    == experiment["candidate"]["feature_order"],
    "Candidate feature order mismatch.",
)

require(
    class_order == experiment["scope"]["target_classes"]
    and len(class_order) == 15
    and len(set(class_order)) == 15,
    "Target class order mismatch.",
)

require(
    len(control_columns) == 24
    and len(candidate_columns) == 56
    and len(team_columns) == 32,
    "Unexpected feature dimensions.",
)

require(
    xgboost.__version__
    == experiment["model_configuration"]["parent_recorded_version"],
    "Installed XGBoost differs from frozen experiment version.",
)

require(
    contract["fitting"]["expected_final_model_fits"] == 4
    and contract["fitting"]["hyperparameter_search"] is False
    and contract["fitting"]["early_stopping"] is False
    and contract["fitting"]["confirmation_rows_used_for_training"] is False,
    "Unexpected final fitting policy.",
)

artifact_paths = {
    name: Path(path)
    for name, path in contract["artifact_plan"].items()
}

require(
    set(artifact_paths) == {
        "control_plus5",
        "candidate_plus5",
        "control_plus10",
        "candidate_plus10",
    },
    "Unexpected final artifact plan.",
)

require(
    all(path.parent == OUT_DIR for path in artifact_paths.values()),
    "Unexpected final artifact directory.",
)

for path in artifact_paths.values():
    require(
        not path.exists(),
        f"Final model artifact already exists: {path}",
    )

print("Frozen contract and source identities: PASS")
print("Final model destinations are unused: PASS")
print("Installed XGBoost:", xgboost.__version__)


# ------------------------------------------------------------
# Assemble the exact frozen development training population.
# ------------------------------------------------------------

targets = pl.read_parquet(TARGETS)

motion = pl.read_parquet(MOTION).select(
    MOTION_KEY
    + ["velocity_X", "velocity_Y", "velocity_Z", "status"]
)

team = pl.read_parquet(TEAM)

require(
    targets.height == 29065
    and targets["demo_filename"].n_unique() == 53,
    "Unexpected target population.",
)

require(
    motion.select(MOTION_KEY).unique().height == motion.height,
    "Duplicate motion keys.",
)

require(
    team.select(TEAM_KEY).unique().height == team.height,
    "Duplicate Team Context keys.",
)

require(
    team.columns == TEAM_KEY + team_columns,
    "Team Context schema differs from frozen feature order.",
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
)

require(
    data.height == 29065
    and data["demo_filename"].n_unique() == 53,
    "Feature joining changed the training population.",
)

require(
    data["status"].null_count() == 0
    and data.filter(pl.col("status") != "RESOLVED").height == 0,
    "Causal motion is missing or unresolved.",
)

require(
    all(data[column].null_count() == 0 for column in team_columns),
    "Missing Team Context features.",
)

for horizon, expected in EXPECTED_ROWS.items():
    frame = data.filter(pl.col("horizon_sec") == horizon)

    require(
        frame.height == expected
        and frame["demo_filename"].n_unique() == 53,
        f"Unexpected +{horizon}s training coverage.",
    )

    require(
        set(frame["target_class_index"].unique().to_list())
        == set(range(15)),
        f"Missing target class at +{horizon}s.",
    )

print("Full-development training joins and class coverage: PASS")


# ------------------------------------------------------------
# Reproduce the exact development OOF feature construction.
# ------------------------------------------------------------

zone_to_index = {
    zone: index
    for index, zone in enumerate(class_order)
}


def build_control_features(frame):
    n = frame.height
    X = np.zeros((n, 24), dtype=np.float32)

    for index, zone in enumerate(frame["current_macro_zone"].to_list()):
        require(zone in zone_to_index, f"Unknown current zone: {zone}")
        X[index, zone_to_index[zone]] = 1.0

    for index, source in enumerate(frame["current_source"].to_list()):
        if source in ("CARRIED_INVENTORY", "CARRIED_PICKUP_FALLBACK"):
            X[index, 15] = 1.0
        elif source == "DROPPED":
            X[index, 16] = 1.0
        else:
            raise RuntimeError(f"STOP: Unexpected bomb source: {source}")

    xyz = np.column_stack([
        frame["current_bomb_X"].to_numpy(),
        frame["current_bomb_Y"].to_numpy(),
        frame["current_bomb_Z"].to_numpy(),
    ]).astype(np.float32, copy=False)

    velocity = np.column_stack([
        frame["velocity_X"].to_numpy(),
        frame["velocity_Y"].to_numpy(),
        frame["velocity_Z"].to_numpy(),
    ]).astype(np.float32, copy=False)

    speed = np.linalg.norm(
        velocity.astype(np.float64),
        axis=1,
    ).astype(np.float32)

    X[:, 17:20] = xyz
    X[:, 20:23] = velocity
    X[:, 23] = speed

    require(
        X.shape == (n, 24) and np.isfinite(X).all(),
        "Invalid Control features.",
    )

    return X


def build_candidate_features(frame, control):
    team_matrix = np.column_stack([
        frame[column].to_numpy()
        for column in team_columns
    ]).astype(np.float32, copy=False)

    X = np.concatenate([control, team_matrix], axis=1)

    require(
        X.shape == (frame.height, 56)
        and np.isfinite(X).all()
        and np.array_equal(X[:, :24], control),
        "Invalid Candidate features.",
    )

    return X


if check_only:
    for horizon in EXPECTED_ROWS:
        frame = data.filter(pl.col("horizon_sec") == horizon)
        control = build_control_features(frame)
        candidate = build_candidate_features(frame, control)

        require(
            control.shape == (EXPECTED_ROWS[horizon], 24)
            and candidate.shape == (EXPECTED_ROWS[horizon], 56),
            f"+{horizon}s feature matrix mismatch.",
        )

    print("\nV4_A_GATE_6B_PREFLIGHT_PASS")
    print("No models were fitted.")
    print("No files were written.")
    raise SystemExit(0)


# ------------------------------------------------------------
# Fit and validate all four models in a NEW staging directory.
# ------------------------------------------------------------

TMP_DIR.mkdir(parents=True)

model_metadata = {}

print("\n=== BEGIN FOUR FINAL MODEL FITS ===")

for horizon in EXPECTED_ROWS:
    frame = data.filter(pl.col("horizon_sec") == horizon)

    X_control = build_control_features(frame)
    X_candidate = build_candidate_features(frame, X_control)

    y = frame["target_class_index"].to_numpy().astype(np.int64)

    for variant, X in (
        ("control", X_control),
        ("candidate", X_candidate),
    ):
        name = f"{variant}_plus{horizon}"
        destination = artifact_paths[name]
        temporary = TMP_DIR / destination.name

        print(
            f"Fitting {name}: rows={len(y):,}, "
            f"features={X.shape[1]}",
            flush=True,
        )

        free_gib()

        model = XGBClassifier(**config)
        model.fit(X, y)

        require(
            np.array_equal(model.classes_, np.arange(15)),
            f"{name}: unexpected class order.",
        )

        sample = X[: min(16, len(X))]
        probabilities_before = model.predict_proba(sample)

        require(
            probabilities_before.shape == (len(sample), 15)
            and np.isfinite(probabilities_before).all()
            and (probabilities_before >= 0).all()
            and np.allclose(
                probabilities_before.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-6,
            ),
            f"{name}: invalid probabilities before saving.",
        )

        model.save_model(str(temporary))

        loaded = XGBClassifier()
        loaded.load_model(str(temporary))

        probabilities_after = loaded.predict_proba(sample)

        require(
            probabilities_after.shape == probabilities_before.shape
            and np.allclose(
                probabilities_after,
                probabilities_before,
                rtol=0.0,
                atol=1e-7,
            ),
            f"{name}: loaded model predictions differ.",
        )

        model_metadata[name] = {
            "artifact": str(destination),
            "sha256": sha256(temporary),
            "training_rows": len(y),
            "training_matches": frame["demo_filename"].n_unique(),
            "feature_dimensions": X.shape[1],
            "feature_order": (
                control_columns if variant == "control"
                else candidate_columns
            ),
            "class_order": class_order,
            "reload_probability_check": "PASS",
        }

        print(
            f"  Saved and reloaded: PASS"
            f"\n  SHA256: {model_metadata[name]['sha256']}",
            flush=True,
        )


# ------------------------------------------------------------
# Promote verified artifacts and create the freeze record.
# ------------------------------------------------------------

require(
    len(model_metadata) == 4,
    "Expected four completed final models.",
)

free_gib()

OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, destination in artifact_paths.items():
    temporary = TMP_DIR / destination.name

    require(
        temporary.is_file() and not destination.exists(),
        f"Cannot safely promote model: {name}",
    )

for name, destination in artifact_paths.items():
    temporary = TMP_DIR / destination.name
    temporary.rename(destination)

    require(
        sha256(destination) == model_metadata[name]["sha256"],
        f"Promoted model SHA256 mismatch: {name}",
    )

freeze = {
    "version": "V4_A_CONFIRM_MODEL_FREEZE_V1",
    "status": "FOUR_FULL_DEVELOPMENT_MODELS_FITTED",
    "fit_git_commit": git_head(),
    "fit_script": str(THIS_SCRIPT),
    "fit_script_sha256": sha256(THIS_SCRIPT),
    "fit_contract": str(CONTRACT),
    "fit_contract_sha256": sha256(CONTRACT),
    "development_matches": 53,
    "development_rows": {
        "5": 14869,
        "10": 14196,
    },
    "xgboost_config": config,
    "package_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "polars",
            "scikit-learn",
            "xgboost",
        )
    },
    "source_artifacts": {
        name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name, path in source_paths.items()
    },
    "models": model_metadata,
    "confirmation_data_accessed": False,
    "confirmation_scoring_performed": False,
    "regeneration_policy": (
        "Do not refit, replace, or recalibrate these models "
        "based on confirmation results."
    ),
}

with FREEZE.open("x") as file:
    json.dump(freeze, file, indent=2, ensure_ascii=False)
    file.write("\n")

TMP_DIR.rmdir()

print("\n=== V4-A GATE 6B FINAL SUMMARY ===")
print("Full-development matches: 53")
print("Final model fits: 4")
print("Model reload and probability checks: PASS")
print("Model SHA256 recorded: 4 / 4")
print("Freeze record:", FREEZE)
print(f"Free disk: {free_gib():.2f} GiB")
print("\nV4_A_FOUR_FINAL_MODELS_FITTED_AND_RECORDED")
print("No confirmation data was accessed or scored.")
