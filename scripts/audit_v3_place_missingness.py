from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser


INCOMING = Path("data/raw/v2_incoming")

AUDIT_PATH = Path(
    "data/interim/v3_place_audit_by_demo.csv"
)

OUTPUT_PATH = Path(
    "data/interim/v3_place_missingness_audit.csv"
)

SAMPLE_OUTPUT = Path(
    "data/interim/v3_place_missing_samples.csv"
)


def coverage(mask, place_valid):
    n = int(mask.sum())

    if n == 0:
        return {
            "rows": 0,
            "labelled": 0,
            "coverage": np.nan,
        }

    labelled = int(
        (
            mask
            & place_valid
        ).sum()
    )

    return {
        "rows": n,
        "labelled": labelled,
        "coverage": labelled / n,
    }


# ============================================================
# Pick bad demos + one high-coverage control
# ============================================================

previous = pd.read_csv(
    AUDIT_PATH
)

low = (
    previous[
        previous["coverage"] < 0.95
    ]
    .sort_values("coverage")
)

control = (
    previous
    .sort_values(
        "coverage",
        ascending=False,
    )
    .head(1)
)

selected = (
    pd.concat(
        [
            low,
            control,
        ],
        ignore_index=True,
    )
    .drop_duplicates(
        "demo_filename"
    )
)

print("=" * 100)
print("V3 GATE 0D — PLACE-NAME MISSINGNESS AUDIT")
print("=" * 100)

print()
print("Selected demos:")

for row in selected.itertuples():
    print(
        f"  {row.demo_filename}"
        f" | prior coverage={row.coverage:.4%}"
    )


# ============================================================
# Parse
# ============================================================

result_rows = []
sample_rows = []

for index, row in enumerate(
    selected.itertuples(),
    start=1,
):

    demo_path = (
        INCOMING
        / row.demo_filename
    )

    print()
    print(
        f"[{index:02d}/{len(selected):02d}] "
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
        "health",
        "is_alive",
        "team_num",
        "is_warmup_period",
        "is_freeze_period",
    ])

    # --------------------------------------------------------
    # Basic masks
    # --------------------------------------------------------

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

    xyz_valid = (
        np.isfinite(df["X"])
        & np.isfinite(df["Y"])
        & np.isfinite(df["Z"])
    )

    alive = (
        df["is_alive"]
        .fillna(False)
        .astype(bool)
    )

    health_positive = (
        df["health"]
        .fillna(0)
        > 0
    )

    real_team = (
        df["team_num"]
        .isin([2, 3])
    )

    not_warmup = ~(
        df["is_warmup_period"]
        .fillna(False)
        .astype(bool)
    )

    post_freeze = ~(
        df["is_freeze_period"]
        .fillna(False)
        .astype(bool)
    )

    all_rows = pd.Series(
        True,
        index=df.index,
    )

    alive_rows = (
        alive
        & health_positive
        & xyz_valid
    )

    competitive_player_rows = (
        alive_rows
        & real_team
        & not_warmup
    )

    active_round_rows = (
        competitive_player_rows
        & post_freeze
    )

    # --------------------------------------------------------
    # Coverage slices
    # --------------------------------------------------------

    slices = {
        "ALL":
            all_rows,

        "XYZ_VALID":
            xyz_valid,

        "ALIVE_HEALTH_POSITIVE":
            alive_rows,

        "COMPETITIVE_ALIVE":
            competitive_player_rows,

        "ACTIVE_ROUND_ALIVE":
            active_round_rows,
    }

    print()

    for slice_name, mask in slices.items():

        stats = coverage(
            mask,
            place_valid,
        )

        print(
            f"  {slice_name:<24}"
            f" rows={stats['rows']:>10,}"
            f" | labelled={stats['labelled']:>10,}"
            f" | coverage="
            f"{stats['coverage']:.4%}"
        )

        result_rows.append({
            "demo_filename":
                demo_path.name,

            "slice":
                slice_name,

            **stats,
        })

    # --------------------------------------------------------
    # What are the missing active rows?
    # --------------------------------------------------------

    relevant_missing = (
        active_round_rows
        & ~place_valid
    )

    missing = (
        df.loc[
            relevant_missing,
            [
                "tick",
                "steamid",
                "name",
                "X",
                "Y",
                "Z",
                "health",
                "is_alive",
                "team_num",
                "is_warmup_period",
                "is_freeze_period",
            ],
        ]
        .copy()
    )

    print(
        "  active missing rows:",
        f"{len(missing):,}",
    )

    if len(missing):

        print(
            "  missing tick range:",
            int(missing["tick"].min()),
            "..",
            int(missing["tick"].max()),
        )

        print(
            "  missing players:",
            missing[
                "steamid"
            ].nunique(),
        )

        # Small deterministic sample only.
        sample = (
            missing
            .sort_values(
                [
                    "tick",
                    "steamid",
                ]
            )
            .head(25)
            .copy()
        )

        sample.insert(
            0,
            "demo_filename",
            demo_path.name,
        )

        sample_rows.append(
            sample
        )


# ============================================================
# Save
# ============================================================

results = pd.DataFrame(
    result_rows
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

if sample_rows:

    samples = pd.concat(
        sample_rows,
        ignore_index=True,
    )

else:

    samples = pd.DataFrame()

samples.to_csv(
    SAMPLE_OUTPUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("ACTIVE-ROUND COVERAGE SUMMARY")
print("=" * 100)

active = (
    results[
        results["slice"]
        == "ACTIVE_ROUND_ALIVE"
    ]
    .sort_values(
        "coverage"
    )
)

print(
    active[
        [
            "demo_filename",
            "rows",
            "labelled",
            "coverage",
        ]
    ]
    .to_string(
        index=False
    )
)

minimum = float(
    active["coverage"].min()
)

mean = float(
    active["coverage"].mean()
)

print()
print(
    "Mean active coverage:",
    f"{mean:.4%}",
)

print(
    "Minimum active coverage:",
    f"{minimum:.4%}",
)

print()
print("=" * 100)

if minimum >= 0.99:

    print(
        "✅ ACTIVE PLACE SEMANTICS ARE HIGH-COVERAGE"
    )

    print(
        "NEXT_ACTION=FREEZE_SEMANTIC_SEED"
    )

else:

    print(
        "⚠️ ACTIVE PLACE MISSINGNESS STILL EXISTS"
    )

    print(
        "NEXT_ACTION=INSPECT_MISSING_PATTERN"
    )

print("=" * 100)

print()
print("Artifacts:")
print(" ", OUTPUT_PATH)
print(" ", SAMPLE_OUTPUT)
