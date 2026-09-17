from __future__ import annotations

from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")

OUTPUT = Path(
    "data/interim/"
    "v3_active_place_coverage_all_reserves.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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

print("=" * 100)
print("V3 GATE 0E — ALL-RESERVE ACTIVE PLACE COVERAGE")
print("=" * 100)

print()
print("Reserve demos:", len(reserve))
print("V2 D_CONFIRM excluded:", len(confirm))
print()

results = []

for index, path in enumerate(
    reserve,
    start=1,
):

    print(
        f"[{index:02d}/13] {path.name}"
    )

    parser = DemoParser(
        str(path)
    )

    df = parser.parse_ticks([
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

    n_active = int(
        active.sum()
    )

    n_labelled = int(
        (
            active
            & place_valid
        ).sum()
    )

    n_missing = (
        n_active
        - n_labelled
    )

    coverage = (
        n_labelled / n_active
        if n_active
        else 0.0
    )

    missing_players = int(
        df.loc[
            active & ~place_valid,
            "steamid",
        ].nunique()
    )

    results.append({
        "demo_filename":
            path.name,

        "active_rows":
            n_active,

        "labelled_active_rows":
            n_labelled,

        "missing_active_rows":
            n_missing,

        "missing_players":
            missing_players,

        "coverage":
            coverage,
    })

    print(
        f"  active={n_active:,}"
        f" | missing={n_missing:,}"
        f" | players={missing_players}"
        f" | coverage={coverage:.4%}"
    )


result = pd.DataFrame(
    results
).sort_values(
    "coverage"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)

minimum = float(
    result["coverage"].min()
)

mean = float(
    result["coverage"].mean()
)

total_active = int(
    result["active_rows"].sum()
)

total_labelled = int(
    result["labelled_active_rows"].sum()
)

weighted = (
    total_labelled
    / total_active
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    result.to_string(
        index=False
    )
)

print()
print(
    "Total active rows:",
    f"{total_active:,}",
)

print(
    "Weighted coverage:",
    f"{weighted:.6%}",
)

print(
    "Mean demo coverage:",
    f"{mean:.6%}",
)

print(
    "Minimum demo coverage:",
    f"{minimum:.6%}",
)

print()

if minimum >= 0.99:

    print(
        "✅ V3 GATE 0 ACTIVE SEMANTIC COVERAGE PASS"
    )

    print(
        "NEXT_ACTION=FREEZE_SEMANTIC_SEED"
    )

else:

    print(
        "❌ V3 GATE 0 REQUIRES INVESTIGATION"
    )

print("=" * 100)

print()
print("Artifact:")
print(" ", OUTPUT)
