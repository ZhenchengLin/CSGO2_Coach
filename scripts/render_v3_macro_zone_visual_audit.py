from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MAPPING_PATH = Path(
    "docs/v3_macro_zone_candidate_v1.json"
)

PREDICTIONS_PATH = Path(
    "data/interim/v3_xyz_place_predictions.csv"
)

SUMMARY_PATH = Path(
    "data/interim/"
    "v3_macro_zone_candidate_v1_spatial_summary.csv"
)

OUTPUT_DIR = Path(
    "artifacts/"
    "v3_macro_zone_candidate_v1_visual_audit"
)

MAX_POINTS_PER_ZONE = 3000
RANDOM_STATE = 20260915


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(
    MAPPING_PATH.exists(),
    f"Missing {MAPPING_PATH}",
)

require(
    PREDICTIONS_PATH.exists(),
    f"Missing {PREDICTIONS_PATH}",
)


mapping_record = json.loads(
    MAPPING_PATH.read_text()
)

predictions = pd.read_csv(
    PREDICTIONS_PATH
)


# ============================================================
# Fine place -> macro zone
# ============================================================

place_to_zone = {}

for zone, places in (
    mapping_record[
        "zones"
    ].items()
):

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate place: {place}",
        )

        place_to_zone[
            place
        ] = zone


observed_places = set(
    predictions[
        "true_place"
    ].astype(str)
)

require(
    observed_places
    == set(place_to_zone),
    "Mapping does not exactly cover observed places.",
)


predictions[
    "macro_zone"
] = (
    predictions[
        "true_place"
    ].map(place_to_zone)
)


# ============================================================
# Spatial summary
# ============================================================

summary = (
    predictions
    .groupby(
        "macro_zone"
    )
    .agg(
        n_samples=("macro_zone", "size"),

        x_mean=("X", "mean"),
        x_min=("X", "min"),
        x_max=("X", "max"),

        y_mean=("Y", "mean"),
        y_min=("Y", "min"),
        y_max=("Y", "max"),

        z_mean=("Z", "mean"),
        z_min=("Z", "min"),
        z_max=("Z", "max"),
    )
    .reset_index()
)


summary[
    "x_span"
] = (
    summary["x_max"]
    - summary["x_min"]
)

summary[
    "y_span"
] = (
    summary["y_max"]
    - summary["y_min"]
)

summary[
    "z_span"
] = (
    summary["z_max"]
    - summary["z_min"]
)


summary[
    "fine_places"
] = (
    summary[
        "macro_zone"
    ].map(
        lambda zone:
            ", ".join(
                mapping_record[
                    "zones"
                ][zone]
            )
    )
)


SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Deterministic visual sample
# ============================================================

sample_frames = []

for zone, group in predictions.groupby(
    "macro_zone"
):

    n = min(
        len(group),
        MAX_POINTS_PER_ZONE,
    )

    sample = group.sample(
        n=n,
        random_state=RANDOM_STATE,
    )

    sample_frames.append(
        sample
    )


visual = pd.concat(
    sample_frames,
    ignore_index=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Shared plotting helper
# ============================================================

def render_projection(
    x_column,
    y_column,
    filename,
    title,
    xlabel,
    ylabel,
    equal_aspect=False,
):

    fig, ax = plt.subplots(
        figsize=(13, 10)
    )

    for zone in sorted(
        visual[
            "macro_zone"
        ].unique()
    ):

        subset = visual[
            visual[
                "macro_zone"
            ]
            == zone
        ]

        ax.scatter(
            subset[
                x_column
            ],
            subset[
                y_column
            ],
            s=5,
            alpha=0.20,
            label=zone,
        )

    # ----------------------------------------------
    # Macro-zone labels at full-data centroids
    # ----------------------------------------------

    x_mean_name = (
        x_column.lower()
        + "_mean"
    )

    y_mean_name = (
        y_column.lower()
        + "_mean"
    )

    for row in summary.itertuples():

        ax.annotate(
            row.macro_zone,
            (
                getattr(
                    row,
                    x_mean_name,
                ),
                getattr(
                    row,
                    y_mean_name,
                ),
            ),
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    if equal_aspect:
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    ax.grid(
        alpha=0.15
    )

    ax.legend(
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "  ✅",
        output,
    )


# ============================================================
# Render
# ============================================================

print("=" * 100)
print("V3 GATE 1H — MACRO-ZONE VISUAL SPATIAL AUDIT")
print("=" * 100)

print()
print(
    "Visual samples:",
    f"{len(visual):,}",
)

print(
    "Macro-zones:",
    visual[
        "macro_zone"
    ].nunique(),
)

print()
print(
    "Rendering raw-coordinate projections..."
)


render_projection(
    "X",
    "Y",
    "v3_macro_zones_xy.png",
    (
        "V3 Macro-Zone Candidate V1 "
        "— Top-Down XY"
    ),
    "World X",
    "World Y",
    equal_aspect=True,
)


render_projection(
    "X",
    "Z",
    "v3_macro_zones_xz.png",
    (
        "V3 Macro-Zone Candidate V1 "
        "— Vertical XZ"
    ),
    "World X",
    "World Z",
)


render_projection(
    "Y",
    "Z",
    "v3_macro_zones_yz.png",
    (
        "V3 Macro-Zone Candidate V1 "
        "— Vertical YZ"
    ),
    "World Y",
    "World Z",
)


print()
print("=" * 100)
print("SPATIAL SUMMARY")
print("=" * 100)

print(
    summary[
        [
            "macro_zone",
            "n_samples",
            "x_span",
            "y_span",
            "z_span",
            "fine_places",
        ]
    ]
    .sort_values(
        "macro_zone"
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("MANUAL VISUAL AUDIT CONTRACT")
print("=" * 100)

print(
    "Check 1: no macro-zone bridges unrelated map regions."
)

print(
    "Check 2: merged fine places form visually coherent regions."
)

print(
    "Check 3: A_RAMP and PALACE remain distinguishable."
)

print(
    "Check 4: MID, CONNECTOR, B_SHORT, and UNDERPASS "
    "remain distinguishable."
)

print(
    "Check 5: B_LOBBY and B_APPS remain distinguishable."
)

print(
    "Check 6: CT_SPAWN and MARKET remain distinguishable."
)

print(
    "Check 7: inspect XZ/YZ for important vertical overlap."
)

print(
    "Check 8: pay special attention to MID_WINDOW, "
    "UNDERPASS, PALACE, and B_APPS."
)

print()
print(
    "No mapping changes are made by this script."
)

print(
    "NEXT_ACTION=MANUAL_VISUAL_REVIEW"
)

print("=" * 100)

print()
print("Summary artifact:")
print(" ", SUMMARY_PATH)

print()
print("Image directory:")
print(" ", OUTPUT_DIR)
