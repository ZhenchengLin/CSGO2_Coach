from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")

OUTPUT = Path(
    "data/interim/"
    "v3_xyz_place_resolver_lodo.csv"
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
    f"Expected 13 reserve demos, found {len(reserve)}",
)


# ============================================================
# Build one-second semantic samples
# ============================================================

print("=" * 100)
print("V3 GATE 1A — XYZ → PLACE RESOLVER AUDIT")
print("=" * 100)

print()
print("Sampling approximately once per second per player.")
print("V2 D_CONFIRM remains excluded.")
print()


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
        df["last_place_name"]
        .notna()
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
                "X",
                "Y",
                "Z",
                "last_place_name",
            ],
        ]
        .copy()
    )

    sample["demo_filename"] = (
        path.name
    )

    frames.append(
        sample
    )

    print(
        "  samples:",
        f"{len(sample):,}",
    )


samples = pd.concat(
    frames,
    ignore_index=True,
)

require(
    len(samples) > 0,
    "No spatial samples produced.",
)


places = sorted(
    samples[
        "last_place_name"
    ]
    .astype(str)
    .unique()
)

place_to_id = {
    place: i
    for i, place
    in enumerate(places)
}

id_to_place = {
    i: place
    for place, i
    in place_to_id.items()
}

samples["label_id"] = (
    samples[
        "last_place_name"
    ]
    .map(place_to_id)
    .astype(int)
)


print()
print(
    "Total samples:",
    f"{len(samples):,}",
)

print(
    "Places:",
    len(places),
)


# ============================================================
# Leave-one-demo-out
# ============================================================

rows = []

all_truth = []
all_pred_xy = []
all_pred_xyz = []


for fold, heldout in enumerate(
    reserve,
    start=1,
):

    name = heldout.name

    print()
    print(
        f"[FOLD {fold:02d}/13] "
        f"hold out {name}"
    )

    train = samples[
        samples["demo_filename"]
        != name
    ]

    test = samples[
        samples["demo_filename"]
        == name
    ]

    require(
        len(train) > 0,
        "Empty training fold.",
    )

    require(
        len(test) > 0,
        "Empty test fold.",
    )

    y_train = (
        train["label_id"]
        .to_numpy()
    )

    y_test = (
        test["label_id"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Resolver A — XY only
    # --------------------------------------------------------

    train_xy = (
        train[
            ["X", "Y"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    test_xy = (
        test[
            ["X", "Y"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    tree_xy = cKDTree(
        train_xy
    )

    _, idx_xy = tree_xy.query(
        test_xy,
        k=1,
        workers=-1,
    )

    pred_xy = (
        y_train[
            idx_xy
        ]
    )

    # --------------------------------------------------------
    # Resolver B — XYZ
    # --------------------------------------------------------

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

    tree_xyz = cKDTree(
        train_xyz
    )

    distance_xyz, idx_xyz = (
        tree_xyz.query(
            test_xyz,
            k=1,
            workers=-1,
        )
    )

    pred_xyz = (
        y_train[
            idx_xyz
        ]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    acc_xy = accuracy_score(
        y_test,
        pred_xy,
    )

    acc_xyz = accuracy_score(
        y_test,
        pred_xyz,
    )

    f1_xy = f1_score(
        y_test,
        pred_xy,
        average="macro",
        zero_division=0,
    )

    f1_xyz = f1_score(
        y_test,
        pred_xyz,
        average="macro",
        zero_division=0,
    )

    mean_distance = float(
        np.mean(
            distance_xyz
        )
    )

    p95_distance = float(
        np.percentile(
            distance_xyz,
            95,
        )
    )

    rows.append({
        "heldout_demo":
            name,

        "n_train":
            len(train),

        "n_test":
            len(test),

        "xy_accuracy":
            acc_xy,

        "xy_macro_f1":
            f1_xy,

        "xyz_accuracy":
            acc_xyz,

        "xyz_macro_f1":
            f1_xyz,

        "xyz_mean_neighbor_distance":
            mean_distance,

        "xyz_p95_neighbor_distance":
            p95_distance,
    })

    all_truth.append(
        y_test
    )

    all_pred_xy.append(
        pred_xy
    )

    all_pred_xyz.append(
        pred_xyz
    )

    print(
        f"  XY  acc={acc_xy:.4%}"
        f" | macroF1={f1_xy:.4f}"
    )

    print(
        f"  XYZ acc={acc_xyz:.4%}"
        f" | macroF1={f1_xyz:.4f}"
        f" | meanNN={mean_distance:.2f}"
        f" | p95NN={p95_distance:.2f}"
    )


# ============================================================
# Aggregate
# ============================================================

result = pd.DataFrame(
    rows
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)


truth = np.concatenate(
    all_truth
)

pred_xy = np.concatenate(
    all_pred_xy
)

pred_xyz = np.concatenate(
    all_pred_xyz
)


weighted_xy_acc = accuracy_score(
    truth,
    pred_xy,
)

weighted_xyz_acc = accuracy_score(
    truth,
    pred_xyz,
)

weighted_xy_f1 = f1_score(
    truth,
    pred_xy,
    average="macro",
    zero_division=0,
)

weighted_xyz_f1 = f1_score(
    truth,
    pred_xyz,
    average="macro",
    zero_division=0,
)


print()
print("=" * 100)
print("LODO SUMMARY")
print("=" * 100)

print(
    result[
        [
            "heldout_demo",
            "xy_accuracy",
            "xyz_accuracy",
            "xyz_macro_f1",
            "xyz_p95_neighbor_distance",
        ]
    ]
    .sort_values(
        "xyz_accuracy"
    )
    .to_string(
        index=False
    )
)

print()
print(
    "Weighted XY accuracy:",
    f"{weighted_xy_acc:.6%}",
)

print(
    "Weighted XYZ accuracy:",
    f"{weighted_xyz_acc:.6%}",
)

print(
    "Weighted XY macro F1:",
    f"{weighted_xy_f1:.6f}",
)

print(
    "Weighted XYZ macro F1:",
    f"{weighted_xyz_f1:.6f}",
)

print(
    "Minimum XYZ fold accuracy:",
    f"{result['xyz_accuracy'].min():.6%}",
)

print()
print("=" * 100)

if (
    weighted_xyz_acc >= 0.95
    and result[
        "xyz_accuracy"
    ].min()
    >= 0.90
):

    print(
        "✅ XYZ → PLACE RESOLUTION IS VIABLE"
    )

    print(
        "NEXT_ACTION=ANALYZE_BOUNDARY_ERRORS"
    )

else:

    print(
        "⚠️ XYZ → PLACE RESOLUTION "
        "NEEDS INVESTIGATION"
    )

print("=" * 100)

print()
print("Artifact:")
print(" ", OUTPUT)
