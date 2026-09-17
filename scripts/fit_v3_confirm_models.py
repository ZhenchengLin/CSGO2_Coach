from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

CONTRACT = Path(
    "docs/v3_confirm_fit_contract_v1.json"
)

TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

MOTION = Path(
    "data/interim/v3_b1_motion_inputs_v1.parquet"
)

SPATIAL = Path(
    "data/interim/"
    "v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)

MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

FEATURE_PROTOCOL = Path(
    "docs/v3_b3_feature_protocol_v1.json"
)

MODEL_PROTOCOL = Path(
    "docs/v3_b3_model_protocol_v1.json"
)

OUT_DIR = Path(
    "artifacts/v3_confirm_frozen"
)

TMP_DIR = Path(
    "artifacts/.v3_confirm_fit_tmp"
)

FREEZE = Path(
    "docs/v3_confirm_model_freeze.json"
)

THIS_SCRIPT = Path(
    "scripts/fit_v3_confirm_models.py"
)


ARTIFACT_NAMES = {
    "b0": "b0_adapter.json",
    "b1": "b1_constant_motion.json",
    "b2": "b2_zone_markov.json",
    "b3_plus5": "b3_plus5.json",
    "b3_plus10": "b3_plus10.json",
}


HORIZONS = [5, 10]
N_CLASSES = 15


# ============================================================
# Helpers
# ============================================================

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


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
        )
        + "\n"
    )


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


# ============================================================
# Preconditions
# ============================================================

for path in [
    CONTRACT,
    TARGETS,
    MOTION,
    SPATIAL,
    MAPPING,
    FEATURE_PROTOCOL,
    MODEL_PROTOCOL,
    THIS_SCRIPT,
]:
    require(
        path.exists(),
        f"Missing required file: {path}",
    )


require(
    not FREEZE.exists(),
    (
        "Frozen V3 confirmation model record "
        "already exists. Refusing regeneration."
    ),
)


for name in ARTIFACT_NAMES.values():
    require(
        not (OUT_DIR / name).exists(),
        (
            "Existing confirmation artifact found: "
            f"{OUT_DIR / name}"
        ),
    )


require(
    not TMP_DIR.exists(),
    (
        f"Temporary fit directory already exists: {TMP_DIR}. "
        "Inspect before deleting or retrying."
    ),
)


contract = read_json(CONTRACT)
mapping = read_json(MAPPING)
feature_protocol = read_json(
    FEATURE_PROTOCOL
)
model_protocol = read_json(
    MODEL_PROTOCOL
)


require(
    contract["status"]
    == "FROZEN_BEFORE_CONFIRMATION_ACCESS",
    "Confirmation fit contract is not frozen.",
)

require(
    contract[
        "development_boundary"
    ][
        "development_closed"
    ]
    is True,
    "Development is not closed.",
)

require(
    contract[
        "development_boundary"
    ][
        "D_V3_CONFIRM_accessed"
    ]
    is False,
    "Contract records confirmation access.",
)


source_sha = contract["source_sha256"]


require(
    sha256(TARGETS)
    == source_sha["targets"],
    "Target SHA differs from frozen contract.",
)

require(
    sha256(MOTION)
    == source_sha["motion"],
    "Motion SHA differs from frozen contract.",
)

require(
    sha256(SPATIAL)
    == source_sha["spatial"],
    "Spatial SHA differs from frozen contract.",
)

require(
    sha256(MAPPING)
    == source_sha["macro_mapping"],
    "Macro mapping SHA differs from frozen contract.",
)

require(
    sha256(FEATURE_PROTOCOL)
    == source_sha["b3_feature_protocol"],
    "Feature protocol SHA differs from frozen contract.",
)

require(
    sha256(MODEL_PROTOCOL)
    == source_sha["b3_model_protocol"],
    "Model protocol SHA differs from frozen contract.",
)


# ============================================================
# Frozen class / feature semantics
# ============================================================

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

xgb_config = dict(
    model_protocol["xgboost_config"]
)


require(
    len(class_order) == N_CLASSES,
    "Expected 15 frozen classes.",
)

require(
    list(mapping["zones"].keys())
    == class_order,
    "Macro-zone order differs from frozen class order.",
)

require(
    len(feature_order) == 24,
    "Expected 24 frozen B3 features.",
)

require(
    xgb_config["num_class"]
    == N_CLASSES,
    "Expected 15-class B3 model.",
)


zone_to_index = {
    zone: i
    for i, zone in enumerate(
        class_order
    )
}


place_to_zone = {}

for zone, places in mapping[
    "zones"
].items():

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate fine place: {place}",
        )

        place_to_zone[
            place
        ] = zone


require(
    len(place_to_zone) == 23,
    "Expected 23 frozen fine places.",
)


expected_feature_order = (
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
    feature_order
    == expected_feature_order,
    "Frozen B3 feature order changed.",
)


# ============================================================
# Load development data
# ============================================================

targets = pl.read_parquet(
    TARGETS
)

motion = pl.read_parquet(
    MOTION
)

spatial = pl.read_parquet(
    SPATIAL
)


require(
    targets.height == 29_065,
    f"Unexpected target rows: {targets.height}",
)

require(
    targets[
        "demo_filename"
    ].n_unique() == 53,
    "Expected exactly 53 development matches.",
)

require(
    motion.height == 14_869,
    f"Unexpected motion rows: {motion.height}",
)

require(
    spatial.height == 774_111,
    f"Unexpected spatial rows: {spatial.height}",
)

require(
    spatial[
        "demo_filename"
    ].n_unique() == 53,
    "Expected spatial samples from 53 matches.",
)

require(
    spatial[
        "place"
    ].n_unique() == 23,
    "Expected 23 spatial fine places.",
)


for horizon, expected in [
    (5, 14_869),
    (10, 14_196),
]:
    n = targets.filter(
        pl.col("horizon_sec")
        == horizon
    ).height

    require(
        n == expected,
        (
            f"+{horizon}s rows changed: "
            f"{n} != {expected}"
        ),
    )


MOTION_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
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


data = targets.join(
    motion_small,
    on=MOTION_KEY,
    how="left",
)


require(
    data.height == targets.height,
    "Motion join changed row count.",
)

require(
    data["status"].null_count()
    == 0,
    "Missing joined motion state.",
)

require(
    data.filter(
        pl.col("status")
        != "RESOLVED"
    ).height == 0,
    "Unresolved motion state.",
)


# ============================================================
# B3 feature builder — identical frozen 24-D representation
# ============================================================

def build_b3_features(frame: pl.DataFrame) -> np.ndarray:

    n = frame.height

    X = np.zeros(
        (n, 24),
        dtype=np.float32,
    )


    zones = frame[
        "current_macro_zone"
    ].to_list()

    for row_i, zone in enumerate(
        zones
    ):
        require(
            zone in zone_to_index,
            f"Unknown current zone: {zone}",
        )

        X[
            row_i,
            zone_to_index[zone],
        ] = 1.0


    sources = frame[
        "current_source"
    ].to_list()

    for row_i, source in enumerate(
        sources
    ):

        if source in {
            "CARRIED_INVENTORY",
            "CARRIED_PICKUP_FALLBACK",
        }:
            X[row_i, 15] = 1.0

        elif source == "DROPPED":
            X[row_i, 16] = 1.0

        else:
            raise RuntimeError(
                f"Unexpected current_source: {source}"
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

    return X


# ============================================================
# Prepare temporary artifact directory
# ============================================================

TMP_DIR.mkdir(
    parents=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# B0 — full-development persistence adapter
# ============================================================

b0_horizons = {}


for horizon in HORIZONS:

    h = data.filter(
        pl.col("horizon_sec")
        == horizon
    )

    n_train = h.height

    n_correct = int(
        h[
            "stay_same_zone"
        ].sum()
    )

    q = (
        n_correct + 1
    ) / (
        n_train + 2
    )

    other_p = (
        1.0 - q
    ) / 14.0


    require(
        0.0 < q < 1.0,
        "Invalid B0 q.",
    )

    require(
        0.0 < other_p < 1.0,
        "Invalid B0 other probability.",
    )


    b0_horizons[
        str(horizon)
    ] = {
        "n_train":
            n_train,

        "n_correct":
            n_correct,

        "training_hard_accuracy":
            n_correct / n_train,

        "q":
            q,

        "other_probability":
            other_p,
    }


b0_record = {
    "artifact":
        "V3 B0 full-development persistence adapter",

    "fit_population":
        "D_V3_DEV only",

    "class_order":
        class_order,

    "formula":
        "q=(n_correct_train+1)/(n_train+2)",

    "horizons":
        b0_horizons,

    "target_dataset_sha256":
        sha256(TARGETS),

    "confirmation_data_accessed":
        False,
}


write_json(
    TMP_DIR / ARTIFACT_NAMES["b0"],
    b0_record,
)


# ============================================================
# B1 — full-development resolver + deterministic adapter
# ============================================================

spatial_xyz = np.column_stack([
    spatial["X"].to_numpy(),
    spatial["Y"].to_numpy(),
    spatial["Z"].to_numpy(),
]).astype(
    np.float64,
    copy=False,
)


spatial_places = np.asarray(
    spatial["place"].to_list(),
    dtype=object,
)


tree = cKDTree(
    spatial_xyz
)


b1_horizons = {}


for horizon in HORIZONS:

    h = data.filter(
        pl.col("horizon_sec")
        == horizon
    )


    xyz = np.column_stack([
        h["current_bomb_X"].to_numpy(),
        h["current_bomb_Y"].to_numpy(),
        h["current_bomb_Z"].to_numpy(),
    ]).astype(
        np.float64,
        copy=False,
    )


    velocity = np.column_stack([
        h["velocity_X"].to_numpy(),
        h["velocity_Y"].to_numpy(),
        h["velocity_Z"].to_numpy(),
    ]).astype(
        np.float64,
        copy=False,
    )


    projected = (
        xyz
        + float(horizon)
        * velocity
    )


    require(
        np.isfinite(projected).all(),
        "Non-finite B1 projected XYZ.",
    )


    distance, nn_index = tree.query(
        projected,
        k=1,
        workers=-1,
    )


    pred_place = spatial_places[
        nn_index
    ]


    pred_zone = np.asarray(
        [
            place_to_zone[
                str(place)
            ]
            for place in pred_place
        ],
        dtype=object,
    )


    pred_index = np.asarray(
        [
            zone_to_index[
                str(zone)
            ]
            for zone in pred_zone
        ],
        dtype=np.int64,
    )


    truth = (
        h[
            "target_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )


    n_train = h.height

    n_correct = int(
        np.sum(
            pred_index == truth
        )
    )


    q = (
        n_correct + 1
    ) / (
        n_train + 2
    )

    other_p = (
        1.0 - q
    ) / 14.0


    b1_horizons[
        str(horizon)
    ] = {
        "n_train":
            n_train,

        "n_correct":
            n_correct,

        "training_hard_accuracy":
            n_correct / n_train,

        "q":
            q,

        "other_probability":
            other_p,

        "resolver_neighbor_distance_mean":
            float(
                np.mean(distance)
            ),

        "resolver_neighbor_distance_p95":
            float(
                np.quantile(
                    distance,
                    0.95,
                )
            ),
    }


b1_record = {
    "artifact":
        "V3 B1 full-development constant-motion adapter",

    "fit_population":
        "D_V3_DEV only",

    "resolver": {
        "library":
            "scipy.spatial.cKDTree",

        "training_rows":
            spatial.height,

        "training_matches":
            53,

        "query_k":
            1,

        "query_workers":
            -1,

        "distance":
            "default Euclidean raw XYZ",

        "projection":
            "current_xyz + horizon_sec * velocity_xyz",

        "tree_serialized":
            False,

        "rebuild_source":
            str(SPATIAL),

        "rebuild_source_sha256":
            sha256(SPATIAL),
    },

    "macro_mapping_sha256":
        sha256(MAPPING),

    "class_order":
        class_order,

    "formula":
        "q=(n_correct_train+1)/(n_train+2)",

    "horizons":
        b1_horizons,

    "confirmation_data_accessed":
        False,
}


write_json(
    TMP_DIR / ARTIFACT_NAMES["b1"],
    b1_record,
)


# ============================================================
# B2 — full-development transition matrices
# ============================================================

b2_horizons = {}


for horizon in HORIZONS:

    h = data.filter(
        pl.col("horizon_sec")
        == horizon
    )


    current_index = (
        h[
            "current_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )

    target_index = (
        h[
            "target_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )


    counts = np.zeros(
        (
            N_CLASSES,
            N_CLASSES,
        ),
        dtype=np.int64,
    )


    np.add.at(
        counts,
        (
            current_index,
            target_index,
        ),
        1,
    )


    probabilities = (
        counts.astype(np.float64)
        + 1.0
    ) / (
        counts.sum(
            axis=1,
            keepdims=True,
        )
        + N_CLASSES
    )


    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ),
        "B2 row probabilities do not sum to one.",
    )


    b2_horizons[
        str(horizon)
    ] = {
        "n_train":
            h.height,

        "counts":
            counts.tolist(),

        "probabilities":
            probabilities.tolist(),
    }


b2_record = {
    "artifact":
        "V3 B2 full-development zone-Markov model",

    "fit_population":
        "D_V3_DEV only",

    "class_order":
        class_order,

    "smoothing":
        "(count_ij+1)/(row_count_i+15)",

    "horizons":
        b2_horizons,

    "target_dataset_sha256":
        sha256(TARGETS),

    "confirmation_data_accessed":
        False,
}


write_json(
    TMP_DIR / ARTIFACT_NAMES["b2"],
    b2_record,
)


# ============================================================
# B3 — two full-development XGBoost models
# ============================================================

b3_metadata = {}


for horizon in HORIZONS:

    h = data.filter(
        pl.col("horizon_sec")
        == horizon
    )


    X = build_b3_features(
        h
    )

    y = (
        h[
            "target_class_index"
        ]
        .to_numpy()
        .astype(np.int64)
    )


    require(
        set(
            np.unique(y).tolist()
        )
        == set(
            range(N_CLASSES)
        ),
        (
            f"+{horizon}s full-development "
            "training set lacks a class."
        ),
    )


    model = XGBClassifier(
        **xgb_config
    )


    model.fit(
        X,
        y,
    )


    key = (
        "b3_plus5"
        if horizon == 5
        else "b3_plus10"
    )


    model_path = (
        TMP_DIR
        / ARTIFACT_NAMES[key]
    )


    model.save_model(
        model_path
    )


    # Round-trip verification.
    loaded = XGBClassifier()
    loaded.load_model(
        model_path
    )


    n_check = min(
        512,
        h.height,
    )


    p_original = model.predict_proba(
        X[:n_check]
    )

    p_loaded = loaded.predict_proba(
        X[:n_check]
    )


    require(
        p_original.shape
        == (
            n_check,
            N_CLASSES,
        ),
        "Unexpected B3 probability shape.",
    )


    require(
        np.allclose(
            p_original,
            p_loaded,
            atol=1e-8,
            rtol=1e-7,
        ),
        (
            f"B3 +{horizon}s model "
            "round-trip predictions differ."
        ),
    )


    require(
        np.allclose(
            p_loaded.sum(axis=1),
            1.0,
            atol=1e-6,
            rtol=0.0,
        ),
        "B3 probabilities do not sum to one.",
    )


    b3_metadata[
        str(horizon)
    ] = {
        "training_rows":
            h.height,

        "training_matches":
            53,

        "feature_dimensions":
            X.shape[1],

        "classes_observed":
            len(
                np.unique(y)
            ),

        "artifact":
            str(
                OUT_DIR
                / ARTIFACT_NAMES[key]
            ),
    }


# ============================================================
# Construct final freeze record using temporary artifact hashes
# ============================================================

tmp_paths = {
    key:
        TMP_DIR / filename
    for key, filename
    in ARTIFACT_NAMES.items()
}


for path in tmp_paths.values():
    require(
        path.exists(),
        f"Missing fitted artifact: {path}",
    )


artifact_records = {
    key: {
        "path":
            str(
                OUT_DIR
                / ARTIFACT_NAMES[key]
            ),

        "sha256":
            sha256(path),
    }
    for key, path
    in tmp_paths.items()
}


package_versions = {
    name: importlib.metadata.version(name)
    for name in [
        "numpy",
        "polars",
        "scipy",
        "scikit-learn",
        "xgboost",
    ]
}


freeze_record = {
    "freeze_name":
        "V3 independent confirmation model freeze",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "fit_git_commit":
        git_head(),

    "fit_script":
        str(THIS_SCRIPT),

    "fit_script_sha256":
        sha256(THIS_SCRIPT),

    "fit_contract":
        str(CONTRACT),

    "fit_contract_sha256":
        sha256(CONTRACT),

    "training_dataset":
        str(TARGETS),

    "training_dataset_sha256":
        sha256(TARGETS),

    "training_rows_total_horizon_rows":
        targets.height,

    "training_rows_by_horizon": {
        "5": 14869,
        "10": 14196,
    },

    "training_matches":
        53,

    "class_order":
        class_order,

    "package_versions":
        package_versions,

    "b0":
        b0_horizons,

    "b1": {
        "spatial_training_rows":
            spatial.height,

        "spatial_training_matches":
            53,

        "spatial_cache_sha256":
            sha256(SPATIAL),

        "motion_cache_sha256":
            sha256(MOTION),

        "macro_mapping_sha256":
            sha256(MAPPING),

        "horizons":
            b1_horizons,
    },

    "b2": {
        "matrix_shape":
            [
                N_CLASSES,
                N_CLASSES,
            ],

        "smoothing":
            "add-one Laplace",

        "fit_separately_by_horizon":
            True,
    },

    "b3": {
        "candidate":
            "B3_TABULAR_MAP_AWARE_V1",

        "feature_protocol_sha256":
            sha256(FEATURE_PROTOCOL),

        "model_protocol_sha256":
            sha256(MODEL_PROTOCOL),

        "feature_order":
            feature_order,

        "xgboost_config":
            xgb_config,

        "models":
            b3_metadata,
    },

    "artifacts":
        artifact_records,

    "confirmation_data_accessed":
        False,

    "confirmation_performance_calculated":
        False,

    "regeneration_policy":
        (
            "These artifacts must not be regenerated "
            "because of D_V3_CONFIRM performance."
        ),

    "scientific_note":
        (
            "All final models/adapters were fitted only "
            "from the frozen 53-match D_V3_DEV corpus. "
            "No V3 confirmation features, labels, "
            "predictions, or performance were accessed."
        ),
}


tmp_freeze = (
    TMP_DIR
    / "v3_confirm_model_freeze.json"
)

write_json(
    tmp_freeze,
    freeze_record,
)


# ============================================================
# Promote temporary artifacts only after all checks pass
# ============================================================

for key, tmp_path in tmp_paths.items():

    final_path = (
        OUT_DIR
        / ARTIFACT_NAMES[key]
    )

    os.replace(
        tmp_path,
        final_path,
    )


FREEZE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


os.replace(
    tmp_freeze,
    FREEZE,
)


# Temporary directory should now be empty.
require(
    len(
        list(
            TMP_DIR.iterdir()
        )
    )
    == 0,
    "Temporary directory not empty after promotion.",
)


TMP_DIR.rmdir()


# ============================================================
# Final on-disk hash verification
# ============================================================

for key, metadata in artifact_records.items():

    path = Path(
        metadata["path"]
    )

    require(
        path.exists(),
        f"Promoted artifact missing: {path}",
    )

    require(
        sha256(path)
        == metadata["sha256"],
        f"Promoted artifact hash mismatch: {path}",
    )


print("V3 FULL-DEVELOPMENT CONFIRMATION FIT")
print()

print("B0")
for h in ["5", "10"]:
    x = b0_horizons[h]
    print(
        f"  +{h}s "
        f"n={x['n_train']:,} "
        f"correct={x['n_correct']:,} "
        f"acc={x['training_hard_accuracy']:.6%} "
        f"q={x['q']:.9f}"
    )

print()
print("B1")
for h in ["5", "10"]:
    x = b1_horizons[h]
    print(
        f"  +{h}s "
        f"n={x['n_train']:,} "
        f"correct={x['n_correct']:,} "
        f"acc={x['training_hard_accuracy']:.6%} "
        f"q={x['q']:.9f} "
        f"NN_p95={x['resolver_neighbor_distance_p95']:.3f}"
    )

print()
print("B2")
print("  +5s 15x15 Laplace matrix: FIT")
print("  +10s 15x15 Laplace matrix: FIT")

print()
print("B3")
print("  +5s rows: 14,869 model: FIT + ROUNDTRIP PASS")
print("  +10s rows: 14,196 model: FIT + ROUNDTRIP PASS")

print()
print("artifacts:", len(artifact_records))
print("confirmation accessed: NO")
print("confirmation performance calculated: NO")
print()
print("V3_CONFIRM_MODELS_FROZEN")
