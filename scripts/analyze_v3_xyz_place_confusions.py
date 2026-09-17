from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from scipy.spatial import cKDTree


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")

PLACE_AUDIT = Path(
    "data/interim/v3_place_audit_by_place.csv"
)

CONFUSION_OUTPUT = Path(
    "data/interim/v3_xyz_place_confusions.csv"
)

SYMMETRIC_OUTPUT = Path(
    "data/interim/v3_xyz_place_confusions_symmetric.csv"
)

CLASS_OUTPUT = Path(
    "data/interim/v3_xyz_place_per_class.csv"
)

PREDICTION_OUTPUT = Path(
    "data/interim/v3_xyz_place_predictions.csv"
)

TICK_RATE = 64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Reserve identity
# ============================================================

confirm = set(
    pd.read_csv(
        MANIFEST
    )["demo_filename"]
)

reserve = sorted(
    p
    for p in INCOMING.glob("*.dem")
    if p.name not in confirm
)

require(
    len(reserve) == 13,
    f"Expected 13 reserves, found {len(reserve)}",
)


# ============================================================
# Build same frozen Gate 1A samples
# ============================================================

print("=" * 100)
print("V3 GATE 1B — XYZ PLACE BOUNDARY ERROR ANALYSIS")
print("=" * 100)

frames = []

for index, path in enumerate(
    reserve,
    start=1,
):

    print(
        f"[{index:02d}/13] "
        f"{path.name}"
    )

    parser = DemoParser(
        str(path)
    )

    df = parser.parse_ticks([
        "X",
        "Y",
        "Z",
        "last_place_name",
        "health",
        "is_alive",
        "team_num",
        "is_warmup_period",
        "is_freeze_period",
    ])

    place_valid = (
        df["last_place_name"].notna()
        &
        (
            df["last_place_name"]
            .astype(str)
            .str.len()
            > 0
        )
    )

    active = (
        df["is_alive"]
        .fillna(False)
        .astype(bool)
        &
        (
            df["health"]
            .fillna(0)
            > 0
        )
        &
        df["team_num"].isin([2, 3])
        &
        ~df["is_warmup_period"]
        .fillna(False)
        .astype(bool)
        &
        ~df["is_freeze_period"]
        .fillna(False)
        .astype(bool)
    )

    sample = (
        df.loc[
            active
            & place_valid
            & (
                df["tick"]
                % TICK_RATE
                == 0
            ),
            [
                "tick",
                "X",
                "Y",
                "Z",
                "last_place_name",
            ],
        ]
        .copy()
    )

    sample = sample.rename(
        columns={
            "last_place_name":
                "true_place",
        }
    )

    sample["demo_filename"] = (
        path.name
    )

    frames.append(
        sample
    )


samples = pd.concat(
    frames,
    ignore_index=True,
)

print()
print(
    "Total samples:",
    f"{len(samples):,}",
)


# ============================================================
# Leave-one-demo-out predictions
# ============================================================

prediction_frames = []

for fold, heldout in enumerate(
    reserve,
    start=1,
):

    name = heldout.name

    print()
    print(
        f"[FOLD {fold:02d}/13] "
        f"{name}"
    )

    train = samples[
        samples["demo_filename"]
        != name
    ].reset_index(
        drop=True
    )

    test = samples[
        samples["demo_filename"]
        == name
    ].copy()

    train_xyz = (
        train[
            ["X", "Y", "Z"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    test_xyz = (
        test[
            ["X", "Y", "Z"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    tree = cKDTree(
        train_xyz
    )

    distance, index = tree.query(
        test_xyz,
        k=1,
        workers=-1,
    )

    predicted = (
        train.iloc[index][
            "true_place"
        ]
        .to_numpy()
    )

    test["predicted_place"] = (
        predicted
    )

    test["nn_distance"] = (
        distance
    )

    test["correct"] = (
        test["true_place"]
        ==
        test["predicted_place"]
    )

    prediction_frames.append(
        test
    )


predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

PREDICTION_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

predictions.to_csv(
    PREDICTION_OUTPUT,
    index=False,
)


# ============================================================
# Basic accuracy
# ============================================================

n_total = len(
    predictions
)

n_correct = int(
    predictions[
        "correct"
    ].sum()
)

n_error = (
    n_total
    - n_correct
)

accuracy = (
    n_correct
    / n_total
)

errors = (
    predictions[
        ~predictions["correct"]
    ]
    .copy()
)


# ============================================================
# Directed confusion pairs
# ============================================================

confusions = (
    errors
    .groupby(
        [
            "true_place",
            "predicted_place",
        ]
    )
    .size()
    .reset_index(
        name="n_errors"
    )
    .sort_values(
        "n_errors",
        ascending=False,
    )
)

confusions[
    "fraction_all_errors"
] = (
    confusions[
        "n_errors"
    ]
    / n_error
)


# ============================================================
# Add physical centroid separation
# ============================================================

centroids = pd.read_csv(
    PLACE_AUDIT
).set_index(
    "place"
)


def centroid_distance(
    true_place,
    predicted_place,
    use_z=False,
):

    a = centroids.loc[
        true_place
    ]

    b = centroids.loc[
        predicted_place
    ]

    dx = (
        a["global_x_mean"]
        - b["global_x_mean"]
    )

    dy = (
        a["global_y_mean"]
        - b["global_y_mean"]
    )

    if not use_z:
        return float(
            np.sqrt(
                dx * dx
                + dy * dy
            )
        )

    dz = (
        a["global_z_mean"]
        - b["global_z_mean"]
    )

    return float(
        np.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )
    )


confusions[
    "place_centroid_distance_xy"
] = [
    centroid_distance(
        row.true_place,
        row.predicted_place,
        use_z=False,
    )
    for row in confusions.itertuples()
]

confusions[
    "place_centroid_distance_xyz"
] = [
    centroid_distance(
        row.true_place,
        row.predicted_place,
        use_z=True,
    )
    for row in confusions.itertuples()
]

confusions.to_csv(
    CONFUSION_OUTPUT,
    index=False,
)


# ============================================================
# Symmetric confusion pairs
#
# Middle -> Connector
# Connector -> Middle
#
# become one pair.
# ============================================================

symmetric = (
    errors[
        [
            "true_place",
            "predicted_place",
        ]
    ]
    .copy()
)

pair_values = symmetric.apply(
    lambda row: sorted([
        str(row["true_place"]),
        str(row["predicted_place"]),
    ]),
    axis=1,
)

symmetric["place_a"] = [
    pair[0]
    for pair in pair_values
]

symmetric["place_b"] = [
    pair[1]
    for pair in pair_values
]

symmetric = (
    symmetric
    .groupby(
        [
            "place_a",
            "place_b",
        ]
    )
    .size()
    .reset_index(
        name="n_errors"
    )
    .sort_values(
        "n_errors",
        ascending=False,
    )
)

symmetric[
    "fraction_all_errors"
] = (
    symmetric[
        "n_errors"
    ]
    / n_error
)

symmetric[
    "place_centroid_distance_xy"
] = [
    centroid_distance(
        row.place_a,
        row.place_b,
        use_z=False,
    )
    for row in symmetric.itertuples()
]

symmetric[
    "place_centroid_distance_xyz"
] = [
    centroid_distance(
        row.place_a,
        row.place_b,
        use_z=True,
    )
    for row in symmetric.itertuples()
]

symmetric.to_csv(
    SYMMETRIC_OUTPUT,
    index=False,
)


# ============================================================
# Per-place performance
# ============================================================

class_rows = []

for place in sorted(
    predictions[
        "true_place"
    ].unique()
):

    subset = predictions[
        predictions[
            "true_place"
        ]
        == place
    ]

    correct = int(
        subset[
            "correct"
        ].sum()
    )

    total = len(
        subset
    )

    error = (
        total
        - correct
    )

    class_rows.append({
        "place":
            place,

        "n":
            total,

        "correct":
            correct,

        "errors":
            error,

        "recall":
            correct / total,
    })


class_result = (
    pd.DataFrame(
        class_rows
    )
    .sort_values(
        "recall"
    )
)

class_result.to_csv(
    CLASS_OUTPUT,
    index=False,
)


# ============================================================
# Distance diagnostics
# ============================================================

correct_distance = (
    predictions.loc[
        predictions["correct"],
        "nn_distance",
    ]
)

error_distance = (
    predictions.loc[
        ~predictions["correct"],
        "nn_distance",
    ]
)


def describe_distance(values):
    return {
        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                values.median()
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),
    }


correct_stats = (
    describe_distance(
        correct_distance
    )
)

error_stats = (
    describe_distance(
        error_distance
    )
)


# ============================================================
# Error concentration near identical XYZ
# ============================================================

error_within_2 = float(
    (
        error_distance
        <= 2
    ).mean()
)

error_within_5 = float(
    (
        error_distance
        <= 5
    ).mean()
)

error_within_10 = float(
    (
        error_distance
        <= 10
    ).mean()
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 100)
print("GLOBAL RESULT")
print("=" * 100)

print(
    "Samples:",
    f"{n_total:,}",
)

print(
    "Correct:",
    f"{n_correct:,}",
)

print(
    "Errors:",
    f"{n_error:,}",
)

print(
    "Accuracy:",
    f"{accuracy:.6%}",
)


print()
print("=" * 100)
print("TOP SYMMETRIC CONFUSION PAIRS")
print("=" * 100)

print(
    symmetric.head(25)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("LOWEST-RECALL PLACES")
print("=" * 100)

print(
    class_result.head(15)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("NEAREST-NEIGHBOR DISTANCE")
print("=" * 100)

print(
    "Correct predictions:"
)

print(
    f"  mean   {correct_stats['mean']:.3f}"
)

print(
    f"  median {correct_stats['median']:.3f}"
)

print(
    f"  p95    {correct_stats['p95']:.3f}"
)

print()

print(
    "Incorrect predictions:"
)

print(
    f"  mean   {error_stats['mean']:.3f}"
)

print(
    f"  median {error_stats['median']:.3f}"
)

print(
    f"  p95    {error_stats['p95']:.3f}"
)

print()

print(
    "Error NN distance <= 2:",
    f"{error_within_2:.2%}",
)

print(
    "Error NN distance <= 5:",
    f"{error_within_5:.2%}",
)

print(
    "Error NN distance <= 10:",
    f"{error_within_10:.2%}",
)


print()
print("=" * 100)
print("NEXT")
print("=" * 100)

print(
    "Inspect whether the dominant errors are "
    "spatially adjacent semantic boundaries."
)

print(
    "Do NOT merge places yet."
)

print(
    "Do NOT tune the resolver yet."
)

print("=" * 100)

print()
print("Artifacts:")
print(" ", CONFUSION_OUTPUT)
print(" ", SYMMETRIC_OUTPUT)
print(" ", CLASS_OUTPUT)
print(" ", PREDICTION_OUTPUT)
