from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PREDICTIONS = Path(
    "data/interim/v3_xyz_place_predictions.csv"
)

CONFUSIONS = Path(
    "data/interim/v3_xyz_place_confusions_symmetric.csv"
)

OUTPUT = Path(
    "data/interim/v3_place_boundary_adjacency.csv"
)

TOP_N = 25


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(
    PREDICTIONS.exists(),
    f"Missing {PREDICTIONS}",
)

require(
    CONFUSIONS.exists(),
    f"Missing {CONFUSIONS}",
)


samples = pd.read_csv(
    PREDICTIONS
)

confusions = (
    pd.read_csv(
        CONFUSIONS
    )
    .head(TOP_N)
    .copy()
)


print("=" * 100)
print("V3 GATE 1C — EMPIRICAL PLACE BOUNDARY AUDIT")
print("=" * 100)

print()
print(
    "Samples:",
    f"{len(samples):,}",
)

print(
    "Confusion pairs:",
    len(confusions),
)


# ============================================================
# Build point clouds once
# ============================================================

point_clouds = {}

for place, group in samples.groupby(
    "true_place"
):

    xyz = (
        group[
            ["X", "Y", "Z"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    point_clouds[
        str(place)
    ] = xyz


# ============================================================
# Estimate boundary distance for each major confusion pair
# ============================================================

rows = []

for index, row in enumerate(
    confusions.itertuples(),
    start=1,
):

    place_a = str(
        row.place_a
    )

    place_b = str(
        row.place_b
    )

    a = point_clouds[
        place_a
    ]

    b = point_clouds[
        place_b
    ]

    # Query the smaller cloud against the larger tree.
    if len(a) <= len(b):

        query_points = a
        reference = b

    else:

        query_points = b
        reference = a

    tree = cKDTree(
        reference
    )

    distance, _ = tree.query(
        query_points,
        k=1,
        workers=-1,
    )

    distance = np.asarray(
        distance,
        dtype=float,
    )

    result = {
        "place_a":
            place_a,

        "place_b":
            place_b,

        "n_errors":
            int(
                row.n_errors
            ),

        "fraction_all_errors":
            float(
                row.fraction_all_errors
            ),

        "n_query_points":
            len(query_points),

        "boundary_min_distance":
            float(
                np.min(distance)
            ),

        "boundary_p01_distance":
            float(
                np.percentile(
                    distance,
                    1,
                )
            ),

        "boundary_p05_distance":
            float(
                np.percentile(
                    distance,
                    5,
                )
            ),

        "boundary_median_distance":
            float(
                np.median(
                    distance
                )
            ),

        "fraction_within_1":
            float(
                np.mean(
                    distance <= 1
                )
            ),

        "fraction_within_5":
            float(
                np.mean(
                    distance <= 5
                )
            ),

        "fraction_within_10":
            float(
                np.mean(
                    distance <= 10
                )
            ),

        "fraction_within_25":
            float(
                np.mean(
                    distance <= 25
                )
            ),
    }

    rows.append(
        result
    )

    print()
    print(
        f"[{index:02d}/{len(confusions):02d}] "
        f"{place_a} ↔ {place_b}"
    )

    print(
        f"  errors={result['n_errors']:,}"
        f" | min={result['boundary_min_distance']:.3f}"
        f" | p01={result['boundary_p01_distance']:.3f}"
        f" | p05={result['boundary_p05_distance']:.3f}"
        f" | <=10={result['fraction_within_10']:.2%}"
    )


result = pd.DataFrame(
    rows
)

result.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("BOUNDARY SUMMARY")
print("=" * 100)

display = result[
    [
        "place_a",
        "place_b",
        "n_errors",
        "boundary_min_distance",
        "boundary_p01_distance",
        "boundary_p05_distance",
        "fraction_within_10",
        "fraction_within_25",
    ]
].copy()

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# How much error mass comes from empirically adjacent regions?
#
# This is diagnostic only.
# It does NOT define the macro-zone merge rule.
# ============================================================

adjacent_10 = result[
    result[
        "boundary_p01_distance"
    ]
    <= 10
]

adjacent_25 = result[
    result[
        "boundary_p01_distance"
    ]
    <= 25
]

error_mass_10 = float(
    adjacent_10[
        "fraction_all_errors"
    ].sum()
)

error_mass_25 = float(
    adjacent_25[
        "fraction_all_errors"
    ].sum()
)


print()
print("=" * 100)
print("ERROR-MASS DIAGNOSTIC")
print("=" * 100)

print(
    "Top-pair error mass with p01 boundary <= 10:",
    f"{error_mass_10:.2%}",
)

print(
    "Top-pair error mass with p01 boundary <= 25:",
    f"{error_mass_25:.2%}",
)

print()
print(
    "NOTE: this is an adjacency diagnostic, "
    "not a macro-zone merge decision."
)

print()
print(
    "NEXT_ACTION=BUILD_SEMANTIC_ADJACENCY_GRAPH"
)

print("=" * 100)

print()
print("Artifact:")
print(" ", OUTPUT)
