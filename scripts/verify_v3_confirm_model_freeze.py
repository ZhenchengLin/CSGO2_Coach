from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from xgboost import XGBClassifier


FREEZE = Path("docs/v3_confirm_model_freeze.json")
CONTRACT = Path("docs/v3_confirm_fit_contract_v1.json")

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

FITTER = Path(
    "scripts/fit_v3_confirm_models.py"
)

OUT = Path(
    "docs/v3_confirm_model_verification.json"
)

THIS_SCRIPT = Path(
    "scripts/verify_v3_confirm_model_freeze.py"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text()
    )


def committed_file_sha(commit, path):
    result = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:{path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        return None

    return hashlib.sha256(
        result.stdout
    ).hexdigest()


for path in [
    FREEZE,
    CONTRACT,
    TARGETS,
    MOTION,
    SPATIAL,
    MAPPING,
    FEATURE_PROTOCOL,
    MODEL_PROTOCOL,
    FITTER,
    THIS_SCRIPT,
]:
    require(
        path.exists(),
        f"Missing required file: {path}",
    )


require(
    not OUT.exists(),
    (
        "Verification record already exists. "
        "Refusing silent overwrite."
    ),
)


freeze = read_json(FREEZE)
contract = read_json(CONTRACT)
mapping = read_json(MAPPING)
model_protocol = read_json(
    MODEL_PROTOCOL
)


require(
    freeze["confirmation_data_accessed"]
    is False,
    "Freeze record says confirmation was accessed.",
)

require(
    freeze[
        "confirmation_performance_calculated"
    ]
    is False,
    "Freeze record says confirmation performance was calculated.",
)

require(
    sha256(CONTRACT)
    == freeze["fit_contract_sha256"],
    "Fit contract SHA mismatch.",
)

require(
    sha256(TARGETS)
    == freeze["training_dataset_sha256"],
    "Target dataset SHA mismatch.",
)

require(
    sha256(FITTER)
    == freeze["fit_script_sha256"],
    "Working-tree fitter differs from fit-time SHA.",
)


current_head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()


current_committed_fitter_sha = (
    committed_file_sha(
        current_head,
        str(FITTER),
    )
)


require(
    current_committed_fitter_sha
    == freeze["fit_script_sha256"],
    (
        "Committed fitter source does not match "
        "the exact fit-time source."
    ),
)


fit_commit = freeze["fit_git_commit"]

ancestor_rc = subprocess.run(
    [
        "git",
        "merge-base",
        "--is-ancestor",
        fit_commit,
        current_head,
    ],
    check=False,
).returncode


require(
    ancestor_rc == 0,
    "Fit-time commit is not an ancestor of current HEAD.",
)


fit_commit_fitter_sha = (
    committed_file_sha(
        fit_commit,
        str(FITTER),
    )
)


artifact_paths = {
    key: Path(meta["path"])
    for key, meta
    in freeze["artifacts"].items()
}


for key, path in artifact_paths.items():

    require(
        path.exists(),
        f"Missing frozen artifact {key}: {path}",
    )

    actual = sha256(path)

    expected = (
        freeze[
            "artifacts"
        ][
            key
        ][
            "sha256"
        ]
    )

    require(
        actual == expected,
        f"Artifact SHA mismatch: {key}",
    )


b0 = read_json(
    artifact_paths["b0"]
)

b1 = read_json(
    artifact_paths["b1"]
)

b2 = read_json(
    artifact_paths["b2"]
)


class_order = freeze["class_order"]

require(
    len(class_order) == 15,
    "Expected 15 classes.",
)

require(
    class_order
    == model_protocol[
        "output"
    ][
        "class_order"
    ],
    "Class order differs from frozen model protocol.",
)

require(
    list(
        mapping["zones"].keys()
    )
    == class_order,
    "Macro-zone order mismatch.",
)


zone_to_index = {
    zone: i
    for i, zone
    in enumerate(class_order)
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
    "Target row count changed.",
)

require(
    targets["demo_filename"].n_unique()
    == 53,
    "Development match count changed.",
)

require(
    spatial.height == 774_111,
    "Spatial cache row count changed.",
)


MOTION_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]


data = targets.join(
    motion.select([
        *MOTION_KEY,
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "status",
    ]),
    on=MOTION_KEY,
    how="left",
)


require(
    data.height == targets.height,
    "Motion join changed row count.",
)

require(
    data["status"].null_count() == 0,
    "Missing motion state.",
)

require(
    data.filter(
        pl.col("status") != "RESOLVED"
    ).height == 0,
    "Unresolved motion state.",
)


# ============================================================
# B0 independent recomputation
# ============================================================

b0_checks = {}


for horizon in [5, 10]:

    h = data.filter(
        pl.col("horizon_sec") == horizon
    )

    n = h.height

    correct = int(
        h["stay_same_zone"].sum()
    )

    q = (
        correct + 1
    ) / (
        n + 2
    )

    other_p = (
        1.0 - q
    ) / 14.0

    stored = b0[
        "horizons"
    ][
        str(horizon)
    ]

    require(
        stored["n_train"] == n,
        f"B0 +{horizon}s n mismatch.",
    )

    require(
        stored["n_correct"] == correct,
        f"B0 +{horizon}s correct mismatch.",
    )

    require(
        np.isclose(
            stored["q"],
            q,
            atol=1e-15,
            rtol=0.0,
        ),
        f"B0 +{horizon}s q mismatch.",
    )

    require(
        np.isclose(
            stored["other_probability"],
            other_p,
            atol=1e-15,
            rtol=0.0,
        ),
        f"B0 +{horizon}s other_p mismatch.",
    )

    b0_checks[
        str(horizon)
    ] = {
        "n": n,
        "correct": correct,
        "q": q,
    }


# ============================================================
# B1 independent recomputation
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


b1_checks = {}


for horizon in [5, 10]:

    h = data.filter(
        pl.col("horizon_sec") == horizon
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
        h["target_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    n = h.height

    correct = int(
        np.sum(
            pred_index == truth
        )
    )

    q = (
        correct + 1
    ) / (
        n + 2
    )

    mean_distance = float(
        np.mean(distance)
    )

    p95_distance = float(
        np.quantile(
            distance,
            0.95,
        )
    )

    stored = b1[
        "horizons"
    ][
        str(horizon)
    ]

    require(
        stored["n_train"] == n,
        f"B1 +{horizon}s n mismatch.",
    )

    require(
        stored["n_correct"] == correct,
        f"B1 +{horizon}s correct mismatch.",
    )

    require(
        np.isclose(
            stored["q"],
            q,
            atol=1e-15,
            rtol=0.0,
        ),
        f"B1 +{horizon}s q mismatch.",
    )

    require(
        np.isclose(
            stored[
                "resolver_neighbor_distance_mean"
            ],
            mean_distance,
            atol=1e-10,
            rtol=1e-12,
        ),
        f"B1 +{horizon}s mean NN distance mismatch.",
    )

    require(
        np.isclose(
            stored[
                "resolver_neighbor_distance_p95"
            ],
            p95_distance,
            atol=1e-10,
            rtol=1e-12,
        ),
        f"B1 +{horizon}s p95 NN distance mismatch.",
    )

    b1_checks[
        str(horizon)
    ] = {
        "n": n,
        "correct": correct,
        "q": q,
        "nn_p95": p95_distance,
    }


# ============================================================
# B2 independent recomputation
# ============================================================

b2_checks = {}


for horizon in [5, 10]:

    h = data.filter(
        pl.col("horizon_sec") == horizon
    )

    current_idx = (
        h["current_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    target_idx = (
        h["target_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    counts = np.zeros(
        (15, 15),
        dtype=np.int64,
    )

    np.add.at(
        counts,
        (
            current_idx,
            target_idx,
        ),
        1,
    )

    probs = (
        counts.astype(np.float64)
        + 1.0
    ) / (
        counts.sum(
            axis=1,
            keepdims=True,
        )
        + 15
    )

    stored = b2[
        "horizons"
    ][
        str(horizon)
    ]

    require(
        np.array_equal(
            np.asarray(
                stored["counts"],
                dtype=np.int64,
            ),
            counts,
        ),
        f"B2 +{horizon}s count matrix mismatch.",
    )

    require(
        np.allclose(
            np.asarray(
                stored["probabilities"],
                dtype=np.float64,
            ),
            probs,
            atol=1e-15,
            rtol=0.0,
        ),
        f"B2 +{horizon}s probability matrix mismatch.",
    )

    require(
        np.allclose(
            probs.sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ),
        f"B2 +{horizon}s row sums invalid.",
    )

    b2_checks[
        str(horizon)
    ] = {
        "training_rows": h.height,
        "matrix_shape": [15, 15],
    }


# ============================================================
# B3 independent load + inference sanity test
# ============================================================

feature_order = model_protocol[
    "input"
][
    "feature_order"
]


def build_features(frame):

    X = np.zeros(
        (frame.height, 24),
        dtype=np.float32,
    )

    for i, zone in enumerate(
        frame["current_macro_zone"].to_list()
    ):
        X[
            i,
            zone_to_index[zone],
        ] = 1.0

    for i, source in enumerate(
        frame["current_source"].to_list()
    ):

        if source in {
            "CARRIED_INVENTORY",
            "CARRIED_PICKUP_FALLBACK",
        }:
            X[i, 15] = 1.0

        elif source == "DROPPED":
            X[i, 16] = 1.0

        else:
            raise RuntimeError(
                f"Unexpected source: {source}"
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
        np.isfinite(X).all(),
        "Non-finite B3 verification feature.",
    )

    return X


require(
    len(feature_order) == 24,
    "Expected 24 B3 features.",
)


b3_checks = {}


for horizon, key in [
    (5, "b3_plus5"),
    (10, "b3_plus10"),
]:

    model = XGBClassifier()

    model.load_model(
        artifact_paths[key]
    )

    frame = data.filter(
        pl.col("horizon_sec") == horizon
    ).head(512)

    X = build_features(
        frame
    )

    probs = model.predict_proba(
        X
    )

    require(
        probs.shape
        == (
            frame.height,
            15,
        ),
        f"B3 +{horizon}s probability shape invalid.",
    )

    require(
        np.isfinite(probs).all(),
        f"B3 +{horizon}s non-finite probabilities.",
    )

    require(
        np.allclose(
            probs.sum(axis=1),
            1.0,
            atol=1e-6,
            rtol=0.0,
        ),
        f"B3 +{horizon}s probability rows invalid.",
    )

    b3_checks[
        str(horizon)
    ] = {
        "rows_checked":
            frame.height,

        "probability_shape":
            list(probs.shape),

        "load_and_predict":
            "PASS",
    }


verification = {
    "verification_name":
        "V3 confirmation model freeze verification",

    "verified_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "verifier_git_commit":
        current_head,

    "verifier_script":
        str(THIS_SCRIPT),

    "verifier_script_sha256":
        sha256(THIS_SCRIPT),

    "freeze_record":
        str(FREEZE),

    "freeze_record_sha256":
        sha256(FREEZE),

    "fit_time_git_commit":
        fit_commit,

    "fit_time_fitter_present_in_commit":
        fit_commit_fitter_sha
        is not None,

    "fit_time_fitter_sha256":
        freeze["fit_script_sha256"],

    "currently_committed_fitter_sha256":
        current_committed_fitter_sha,

    "exact_fit_source_recovered_and_committed":
        (
            current_committed_fitter_sha
            == freeze["fit_script_sha256"]
        ),

    "artifact_sha256_verification":
        "PASS",

    "B0_recomputation":
        b0_checks,

    "B1_recomputation":
        b1_checks,

    "B2_recomputation":
        b2_checks,

    "B3_load_and_inference":
        b3_checks,

    "confirmation_data_accessed":
        False,

    "confirmation_performance_calculated":
        False,

    "overall_status":
        "PASS",
}


OUT.write_text(
    json.dumps(
        verification,
        indent=2,
    )
    + "\n"
)


print("V3 CONFIRMATION MODEL VERIFICATION")
print()
print("artifact SHA256: PASS")
print("fit contract SHA256: PASS")
print("committed fitter == fit-time fitter: PASS")

print()
print("B0")
for h in ["5", "10"]:
    x = b0_checks[h]
    print(
        f"  +{h}s "
        f"correct={x['correct']:,}/{x['n']:,} "
        f"q={x['q']:.9f} PASS"
    )

print()
print("B1")
for h in ["5", "10"]:
    x = b1_checks[h]
    print(
        f"  +{h}s "
        f"correct={x['correct']:,}/{x['n']:,} "
        f"q={x['q']:.9f} "
        f"NN_p95={x['nn_p95']:.3f} PASS"
    )

print()
print("B2")
print("  +5s count/probability matrix: PASS")
print("  +10s count/probability matrix: PASS")

print()
print("B3")
print("  +5s load + predict_proba: PASS")
print("  +10s load + predict_proba: PASS")

print()
print("confirmation accessed: NO")
print("confirmation performance calculated: NO")
print()
print("V3_CONFIRM_MODEL_VERIFICATION_PASS")
