from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")

PER_DEMO_OUTPUT = Path(
    "data/interim/v3_place_audit_by_demo.csv"
)

PLACE_OUTPUT = Path(
    "data/interim/v3_place_audit_by_place.csv"
)

CENTROID_OUTPUT = Path(
    "data/interim/v3_place_centroids_by_demo.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Select V2 reserves only
# ============================================================

confirm = set(
    pd.read_csv(
        MANIFEST
    )["demo_filename"]
)

all_demos = sorted(
    INCOMING.glob("*.dem")
)

reserve = [
    path
    for path in all_demos
    if path.name not in confirm
]

print("=" * 100)
print("V3 GATE 0C — MULTI-DEMO PLACE-NAME AUDIT")
print("=" * 100)

print(
    f"All incoming demos: {len(all_demos)}"
)

print(
    f"Sealed V2 D_CONFIRM demos: {len(confirm)}"
)

print(
    f"V3 semantic-audit reserves: {len(reserve)}"
)

require(
    len(reserve) == 13,
    (
        "Expected exactly 13 V2 reserve demos. "
        f"Found {len(reserve)}."
    ),
)

print()
print(
    "✅ D_CONFIRM remains excluded from V3 semantic design"
)


# ============================================================
# Parse reserves
# ============================================================

demo_rows = []
centroid_rows = []

union_places = set()

for index, demo_path in enumerate(
    reserve,
    start=1,
):

    print()
    print(
        f"[{index:02d}/{len(reserve):02d}] "
        f"{demo_path.name}"
    )

    parser = DemoParser(
        str(demo_path)
    )

    df = parser.parse_ticks([
        "X",
        "Y",
        "Z",
        "last_place_name",
    ])

    total_rows = len(df)

    valid_mask = (
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

    valid = (
        df.loc[
            valid_mask
        ]
        .copy()
    )

    labelled_rows = len(valid)

    coverage = (
        labelled_rows
        / total_rows
        if total_rows
        else 0.0
    )

    places = set(
        valid[
            "last_place_name"
        ].astype(str)
    )

    union_places.update(
        places
    )

    demo_rows.append({
        "demo_filename":
            demo_path.name,

        "n_rows":
            total_rows,

        "n_labelled":
            labelled_rows,

        "coverage":
            coverage,

        "n_places":
            len(places),

        "place_names":
            ";".join(
                sorted(
                    places
                )
            ),
    })

    print(
        f"  rows={total_rows:,}"
        f" | labelled={labelled_rows:,}"
        f" | coverage={coverage:.4%}"
        f" | places={len(places)}"
    )

    # --------------------------------------------------------
    # Centroid/range for each place inside this demo
    # --------------------------------------------------------

    grouped = (
        valid
        .groupby(
            "last_place_name",
            dropna=False,
        )
    )

    for place, group in grouped:

        if isinstance(
            place,
            tuple,
        ):
            place = place[0]

        centroid_rows.append({
            "demo_filename":
                demo_path.name,

            "place":
                str(place),

            "n":
                len(group),

            "x_mean":
                float(
                    group["X"].mean()
                ),

            "y_mean":
                float(
                    group["Y"].mean()
                ),

            "z_mean":
                float(
                    group["Z"].mean()
                ),

            "x_min":
                float(
                    group["X"].min()
                ),

            "x_max":
                float(
                    group["X"].max()
                ),

            "y_min":
                float(
                    group["Y"].min()
                ),

            "y_max":
                float(
                    group["Y"].max()
                ),

            "z_min":
                float(
                    group["Z"].min()
                ),

            "z_max":
                float(
                    group["Z"].max()
                ),
        })


# ============================================================
# DataFrames
# ============================================================

demo_audit = pd.DataFrame(
    demo_rows
)

centroids = pd.DataFrame(
    centroid_rows
)


# ============================================================
# Cross-demo place stability
# ============================================================

place_rows = []

for place in sorted(
    union_places
):

    p = centroids[
        centroids["place"]
        == place
    ].copy()

    require(
        len(p) > 0,
        f"No centroid rows for {place}",
    )

    global_x = float(
        np.average(
            p["x_mean"],
            weights=p["n"],
        )
    )

    global_y = float(
        np.average(
            p["y_mean"],
            weights=p["n"],
        )
    )

    global_z = float(
        np.average(
            p["z_mean"],
            weights=p["n"],
        )
    )

    centroid_xy_shift = np.sqrt(
        (
            p["x_mean"]
            - global_x
        ) ** 2
        +
        (
            p["y_mean"]
            - global_y
        ) ** 2
    )

    place_rows.append({
        "place":
            place,

        "demos_present":
            len(p),

        "demo_coverage":
            len(p)
            / len(reserve),

        "total_rows":
            int(
                p["n"].sum()
            ),

        "global_x_mean":
            global_x,

        "global_y_mean":
            global_y,

        "global_z_mean":
            global_z,

        "mean_demo_centroid_xy_shift":
            float(
                centroid_xy_shift.mean()
            ),

        "max_demo_centroid_xy_shift":
            float(
                centroid_xy_shift.max()
            ),

        "x_min":
            float(
                p["x_min"].min()
            ),

        "x_max":
            float(
                p["x_max"].max()
            ),

        "y_min":
            float(
                p["y_min"].min()
            ),

        "y_max":
            float(
                p["y_max"].max()
            ),

        "z_min":
            float(
                p["z_min"].min()
            ),

        "z_max":
            float(
                p["z_max"].max()
            ),
    })


place_audit = (
    pd.DataFrame(
        place_rows
    )
    .sort_values(
        [
            "demos_present",
            "total_rows",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


# ============================================================
# Save artifacts
# ============================================================

PER_DEMO_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

demo_audit.to_csv(
    PER_DEMO_OUTPUT,
    index=False,
)

centroids.to_csv(
    CENTROID_OUTPUT,
    index=False,
)

place_audit.to_csv(
    PLACE_OUTPUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 100)
print("PER-DEMO COVERAGE")
print("=" * 100)

print(
    demo_audit[
        [
            "demo_filename",
            "coverage",
            "n_places",
        ]
    ]
    .to_string(
        index=False
    )
)

print()
print("=" * 100)
print("GLOBAL PLACE STABILITY")
print("=" * 100)

display_columns = [
    "place",
    "demos_present",
    "demo_coverage",
    "total_rows",
    "global_x_mean",
    "global_y_mean",
    "mean_demo_centroid_xy_shift",
    "max_demo_centroid_xy_shift",
]

print(
    place_audit[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# Gate checks
# ============================================================

min_coverage = float(
    demo_audit[
        "coverage"
    ].min()
)

mean_coverage = float(
    demo_audit[
        "coverage"
    ].mean()
)

min_places = int(
    demo_audit[
        "n_places"
    ].min()
)

max_places = int(
    demo_audit[
        "n_places"
    ].max()
)

common_places = (
    place_audit[
        place_audit[
            "demos_present"
        ]
        == len(reserve)
    ]
)

print()
print("=" * 100)
print("GATE SUMMARY")
print("=" * 100)

print(
    f"Reserve demos:       "
    f"{len(reserve)}"
)

print(
    f"Union places:        "
    f"{len(union_places)}"
)

print(
    f"Places in all demos: "
    f"{len(common_places)}"
)

print(
    f"Mean coverage:       "
    f"{mean_coverage:.4%}"
)

print(
    f"Minimum coverage:    "
    f"{min_coverage:.4%}"
)

print(
    f"Places/demo range:   "
    f"{min_places} .. {max_places}"
)

print()

if (
    min_coverage >= 0.95
    and len(union_places) >= 15
):
    print(
        "✅ V3 PLACE SEMANTIC AUDIT PASS"
    )

    print(
        "NEXT_ACTION=DESIGN_FROZEN_MACRO_ZONES"
    )

else:
    print(
        "⚠️ V3 PLACE SEMANTIC AUDIT NEEDS INVESTIGATION"
    )

print("=" * 100)

print()
print("Artifacts:")
print(" ", PER_DEMO_OUTPUT)
print(" ", PLACE_OUTPUT)
print(" ", CENTROID_OUTPUT)
