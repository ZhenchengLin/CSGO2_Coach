from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")


confirm = set(
    pd.read_csv(MANIFEST)["demo_filename"]
)

all_demos = sorted(
    INCOMING.glob("*.dem")
)

reserve = [
    p
    for p in all_demos
    if p.name not in confirm
]

print("=" * 88)
print("V3 GATE 0B — VALVE PLACE-NAME AUDIT")
print("=" * 88)

print("all incoming demos:", len(all_demos))
print("V2 D_CONFIRM demos:", len(confirm))
print("available reserve demos:", len(reserve))

if not reserve:
    raise SystemExit(
        "❌ No reserve demos found."
    )

demo_path = reserve[0]

print()
print("Audit demo:")
print(" ", demo_path.name)

parser = DemoParser(
    str(demo_path)
)

df = parser.parse_ticks(
    [
        "X",
        "Y",
        "Z",
        "last_place_name",
    ]
)

print()
print("rows:", len(df))
print("columns:", list(df.columns))

if "last_place_name" not in df.columns:
    raise SystemExit(
        "❌ last_place_name not returned"
    )

valid = df[
    df["last_place_name"].notna()
].copy()

valid = valid[
    valid["last_place_name"]
    .astype(str)
    .str.len()
    > 0
]

print()
print(
    "place-labelled rows:",
    len(valid),
)

print(
    "coverage:",
    f"{len(valid) / len(df):.4%}",
)

summary = (
    valid
    .groupby(
        "last_place_name",
        dropna=False,
    )
    .agg(
        n=("tick", "size"),
        x_mean=("X", "mean"),
        y_mean=("Y", "mean"),
        z_mean=("Z", "mean"),
        x_min=("X", "min"),
        x_max=("X", "max"),
        y_min=("Y", "min"),
        y_max=("Y", "max"),
        z_min=("Z", "min"),
        z_max=("Z", "max"),
    )
    .sort_values(
        "n",
        ascending=False,
    )
)

print()
print("=" * 88)
print("PLACE SUMMARY")
print("=" * 88)

print(
    summary.to_string()
)

print()
print(
    "unique places:",
    summary.shape[0],
)

print()
print("PLACE NAMES:")
for place in sorted(
    summary.index.astype(str)
):
    print(" ", place)

print()
print("=" * 88)

if (
    len(valid) / len(df) > 0.95
    and summary.shape[0] >= 8
):
    print(
        "✅ PLACE-NAME SEMANTIC SEED AVAILABLE"
    )
    print(
        "NEXT_ACTION=AUDIT_MULTIPLE_RESERVE_DEMOS"
    )
else:
    print(
        "⚠️ PLACE-NAME COVERAGE NEEDS INVESTIGATION"
    )

print("=" * 88)
