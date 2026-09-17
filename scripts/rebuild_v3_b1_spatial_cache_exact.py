from pathlib import Path
import hashlib
import json

import pandas as pd
import polars as pl
from demoparser2 import DemoParser


MANIFEST = Path("docs/v3_dev_manifest.csv")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")

OUTPUT = Path(
    "data/interim/v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)

IDENTITY = Path(
    "docs/v3_b1_spatial_cache_identity_v1.json"
)

EXPECTED_RESERVE_ROWS = 186_953


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


manifest = pd.read_csv(MANIFEST)

require(
    len(manifest) == 53,
    f"Expected 53 demos, got {len(manifest)}",
)


mapping = json.loads(
    MAPPING.read_text()
)

frozen_places = {
    place
    for places in mapping["zones"].values()
    for place in places
}

require(
    len(frozen_places) == 23,
    "Frozen mapping must contain 23 fine places.",
)


frames = []

for i, row in enumerate(
    manifest.sort_values("dev_rank").itertuples(index=False),
    start=1,
):
    print(
        f"[{i:02d}/53] "
        f"{row.dev_source_role:<15} "
        f"{row.demo_filename}"
    )

    parser = DemoParser(str(row.path))

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
            & (df["tick"] % 64 == 0),
            [
                "X",
                "Y",
                "Z",
                "last_place_name",
            ],
        ]
        .copy()
        .rename(
            columns={
                "last_place_name": "place"
            }
        )
    )

    sample["demo_filename"] = row.demo_filename
    sample["dev_source_role"] = row.dev_source_role

    frames.append(sample)

    print("  samples:", f"{len(sample):,}")


samples = pl.from_pandas(
    pd.concat(
        frames,
        ignore_index=True,
    )
)

reserve = samples.filter(
    pl.col("dev_source_role") == "V2_RESERVE"
)

actual_places = set(
    samples["place"].unique().to_list()
)


print()
print("=== EXACT REGRESSION CHECK ===")
print(
    "all matches:",
    samples["demo_filename"].n_unique(),
)
print(
    "all rows:",
    f"{samples.height:,}",
)
print(
    "fine places:",
    len(actual_places),
)
print(
    "reserve matches:",
    reserve["demo_filename"].n_unique(),
)
print(
    "reserve rows:",
    f"{reserve.height:,}",
)
print(
    "expected reserve rows:",
    f"{EXPECTED_RESERVE_ROWS:,}",
)


require(
    samples["demo_filename"].n_unique() == 53,
    "Not all 53 demos were sampled.",
)

require(
    reserve["demo_filename"].n_unique() == 13,
    "Expected 13 reserve demos.",
)

require(
    reserve.height == EXPECTED_RESERVE_ROWS,
    (
        f"Gate-1 regression failed: "
        f"{reserve.height:,} != 186,953"
    ),
)

require(
    actual_places == frozen_places,
    "Fine-place universe differs from frozen mapping.",
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

samples.write_parquet(
    OUTPUT,
    compression="zstd",
)


identity = {
    "version": "V3_B1_SPATIAL_CACHE_V1",
    "status": "FROZEN",
    "matches": 53,
    "rows": samples.height,
    "fine_places": 23,
    "gate1_reserve_rows": reserve.height,
    "gate1_expected_rows": EXPECTED_RESERVE_ROWS,
    "gate1_regression_pass": True,
    "manifest_sha256": sha256(MANIFEST),
    "macro_mapping_sha256": sha256(MAPPING),
    "output": str(OUTPUT),
    "output_sha256": sha256(OUTPUT),
    "v2_d_confirm_used": False,
}

IDENTITY.write_text(
    json.dumps(
        identity,
        indent=2,
    )
    + "\n"
)


print()
print("=== BLOCK A RESULT ===")
print("53 matches: PASS")
print(
    "Gate-1 reserve:",
    f"{reserve.height:,} / 186,953",
)
print("fine places: 23 / 23")
print(
    "output rows:",
    f"{samples.height:,}",
)
print("BLOCK_A_PASS")
