"""V4 Gate 1B: build development-only causal Team Context features.

One feature row per exact current-time observation.
No model fitting, confirmation access, or V3 artifact modification.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo


MANIFEST = Path("docs/v3_dev_manifest.csv")
MANIFEST_IDENTITY = Path("docs/v3_dev_manifest_identity.json")

TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
TARGET_IDENTITY = Path("docs/v3_dev_target_dataset_identity.json")

MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")
B3_PROTOCOL = Path("docs/v3_b3_feature_protocol_v1.json")

GATE0_REPORT = Path(
    "docs/v4_gate0b_development_audit_v1.json"
)

OUTPUT = Path(
    "data/interim/v4_dev_team_context_v1.parquet"
)

IDENTITY = Path(
    "docs/v4_dev_team_context_identity_v1.json"
)

MIN_FREE_BYTES = 12 * 1024**3

AFFECTED_DEMO = (
    "2026-09-12_iowa-stormboar_vs_"
    "sportsbetexpert_mirage.dem"
)

AFFECTED_ROUND = 13
AFFECTED_TICK = 99365


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def disk_check():
    free = shutil.disk_usage(".").free / 1024**3

    require(
        free >= 12,
        f"Free disk below 12 GiB: {free:.2f} GiB",
    )

    return free


def load_json(path):
    return json.loads(path.read_text())


print("\n=== V4 GATE 1B: DEVELOPMENT FEATURE EXTRACTION ===")
print(f"Initial free disk: {disk_check():.2f} GiB")

for path in (
    MANIFEST,
    MANIFEST_IDENTITY,
    TARGETS,
    TARGET_IDENTITY,
    MAPPING,
    B3_PROTOCOL,
    GATE0_REPORT,
):
    require(path.is_file(), f"Missing required input: {path}")

require(
    not OUTPUT.exists(),
    f"Output already exists; refusing to overwrite: {OUTPUT}",
)

require(
    not IDENTITY.exists(),
    f"Identity already exists; refusing to overwrite: {IDENTITY}",
)

manifest_identity = load_json(MANIFEST_IDENTITY)
target_identity = load_json(TARGET_IDENTITY)
mapping = load_json(MAPPING)
b3 = load_json(B3_PROTOCOL)
gate0 = load_json(GATE0_REPORT)

require(
    manifest_identity["status"] == "FROZEN"
    and target_identity["status"] == "FROZEN",
    "Unexpected frozen input status.",
)

manifest_sha = sha256(MANIFEST)
targets_sha = sha256(TARGETS)
mapping_sha = sha256(MAPPING)

require(
    manifest_sha == manifest_identity["manifest_sha256"],
    "Development manifest SHA mismatch.",
)

require(
    manifest_sha == target_identity["development_manifest_sha256"],
    "Target dataset references another development manifest.",
)

require(
    targets_sha == target_identity["output_sha256"],
    "Frozen development targets changed.",
)

require(
    mapping_sha == target_identity["macro_mapping_sha256"],
    "Frozen macro-zone mapping changed.",
)

require(
    gate0["matches_audited"] == 53
    and gate0["quality_issues"] == {"unmapped_player_rows": 1},
    "Unexpected Gate 0 audit evidence.",
)

require(
    gate0["development_manifest_sha256"] == manifest_sha
    and gate0["development_targets_sha256"] == targets_sha
    and gate0["macro_mapping_sha256"] == mapping_sha,
    "Gate 0 evidence does not match current frozen inputs.",
)

zones = b3["features"]["current_macro_zone"]["categories"]

require(
    b3["total_dimensions"] == 24
    and len(zones) == 15
    and len(set(zones)) == 15
    and set(zones) == set(mapping["zones"]),
    "Unexpected V3 B3 feature or zone contract.",
)

place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate fine place: {place}",
        )
        place_to_zone[place] = zone


# The exact 32-dimensional feature order.
FEATURE_COLUMNS = (
    [f"v4_t__{zone}" for zone in zones]
    + [f"v4_ct__{zone}" for zone in zones]
    + [
        "v4_t__UNKNOWN_PLACE",
        "v4_ct__UNKNOWN_PLACE",
    ]
)

require(
    len(FEATURE_COLUMNS) == 32
    and len(set(FEATURE_COLUMNS)) == 32,
    "Team Context feature order is invalid.",
)


def extract_context(players):
    """Extract current-time occupancy, without reading future state."""

    t_counts = Counter()
    ct_counts = Counter()

    unknown_t = 0
    unknown_ct = 0

    living_t = 0
    living_ct = 0

    for player in players:

        side = player["side"]
        health = player["health"]

        if side not in ("t", "ct"):
            continue

        if health is None or health <= 0:
            continue

        if side == "t":
            living_t += 1
            counts = t_counts
        else:
            living_ct += 1
            counts = ct_counts

        zone = place_to_zone.get(player["place"])

        if zone is None:
            if side == "t":
                unknown_t += 1
            else:
                unknown_ct += 1
        else:
            counts[zone] += 1

    require(
        1 <= living_t <= 5 and 1 <= living_ct <= 5,
        "Unexpected living team count.",
    )

    vector = (
        [t_counts[zone] for zone in zones]
        + [ct_counts[zone] for zone in zones]
        + [unknown_t, unknown_ct]
    )

    require(
        len(vector) == 32,
        "Feature dimension mismatch.",
    )

    require(
        all(isinstance(value, int) and value >= 0
            for value in vector),
        "Team Context must contain nonnegative integer counts.",
    )

    require(
        sum(vector[:15]) + vector[30] == living_t,
        "T-side occupancy conservation failed.",
    )

    require(
        sum(vector[15:30]) + vector[31] == living_ct,
        "CT-side occupancy conservation failed.",
    )

    return vector, living_t, living_ct


with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(
    len(manifest) == 53
    and len({row["demo_filename"] for row in manifest}) == 53,
    "Development manifest must contain 53 distinct matches.",
)

targets = pl.read_parquet(
    TARGETS,
    columns=[
        "demo_filename",
        "round_num",
        "current_tick",
        "horizon_sec",
    ],
)

require(
    targets.height == 29065
    and targets["demo_filename"].n_unique() == 53,
    "Unexpected frozen development target coverage.",
)

require(
    set(targets["horizon_sec"].unique().to_list()) == {5, 10},
    "Unexpected target horizon.",
)

observation_keys = [
    "demo_filename",
    "round_num",
    "current_tick",
]

observations = targets.select(
    observation_keys
).unique()

require(
    observations.height == 14869,
    "Expected 14,869 unique current-time observations.",
)

feature_rows = []
totals = Counter()


# ------------------------------------------------------------
# Parse one existing development demo at a time.
# ------------------------------------------------------------

for index, record in enumerate(manifest, start=1):

    print(
        f"\n[{index:02d}/53] {record['demo_filename']}",
        flush=True,
    )

    print(
        f"  Free disk: {disk_check():.2f} GiB",
        flush=True,
    )

    name = record["demo_filename"]
    demo_path = Path(record["path"])

    require(
        demo_path.is_file(),
        f"Missing development demo: {demo_path}",
    )

    require(
        record["map_name"] == "de_mirage"
        and int(record["raw_ticks_per_sec"]) == 64,
        f"Map/timing contract mismatch: {name}",
    )

    match_obs = observations.filter(
        pl.col("demo_filename") == name
    )

    require(
        match_obs.height > 0,
        f"No current-time observations for {name}",
    )

    requested = {
        (int(row["round_num"]), int(row["current_tick"]))
        for row in match_obs.iter_rows(named=True)
    }

    require(
        len(requested) == match_obs.height,
        f"Duplicate current-time keys: {name}",
    )

    ticks = {tick for _, tick in requested}

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )

    require(
        demo.header.get("map_name") == "de_mirage",
        f"Parsed map mismatch: {name}",
    )

    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )

    required_columns = {
        "round_num",
        "tick",
        "side",
        "health",
        "place",
    }

    require(
        required_columns <= set(demo.ticks.columns),
        f"Player snapshot schema incomplete: {name}",
    )

    snapshots = demo.ticks.filter(
        pl.col("tick").is_in(list(ticks))
    ).select(sorted(required_columns))

    recovered = set()
    match_total = Counter()

    for group in snapshots.partition_by(
        ["round_num", "tick"],
        maintain_order=False,
    ):

        key = (
            int(group["round_num"][0]),
            int(group["tick"][0]),
        )

        if key not in requested:
            continue

        require(
            key not in recovered,
            f"Duplicate snapshot group: {name}, {key}",
        )

        recovered.add(key)

        vector, living_t, living_ct = extract_context(
            group.iter_rows(named=True)
        )

        row = {
            "demo_filename": name,
            "round_num": key[0],
            "current_tick": key[1],
        }

        row.update(
            dict(zip(FEATURE_COLUMNS, vector))
        )

        feature_rows.append(row)

        match_total["observations"] += 1
        match_total["living_t"] += living_t
        match_total["living_ct"] += living_ct

        match_total["mapped"] += sum(vector[:30])
        match_total["unknown"] += vector[30] + vector[31]

        match_total["unknown_t"] += vector[30]
        match_total["unknown_ct"] += vector[31]

    require(
        recovered == requested,
        f"Exact snapshot coverage mismatch: {name}",
    )

    # Compare against the independently archived Gate 0 counts.
    gate0_match = [
        item
        for item in gate0["per_match"]
        if item["demo_filename"] == name
    ]

    require(
        len(gate0_match) == 1,
        f"Missing Gate 0 record: {name}",
    )

    reference = gate0_match[0]["counts"]

    require(
        match_total["observations"]
        == reference["requested_snapshots"],
        f"Observation count differs from Gate 0: {name}",
    )

    require(
        match_total["living_t"] == reference["alive_t_rows"],
        f"T player count differs from Gate 0: {name}",
    )

    require(
        match_total["living_ct"] == reference["alive_ct_rows"],
        f"CT player count differs from Gate 0: {name}",
    )

    require(
        match_total["mapped"] == reference["mapped_player_rows"],
        f"Mapped player count differs from Gate 0: {name}",
    )

    require(
        match_total["unknown"]
        == reference.get("unmapped_player_rows", 0),
        f"Unknown player count differs from Gate 0: {name}",
    )

    totals.update(match_total)

    print(
        "  Observations:",
        match_total["observations"],
        "| mapped:",
        match_total["mapped"],
        "| unknown:",
        match_total["unknown"],
        flush=True,
    )

    del demo
    del snapshots

    gc.collect()


# ------------------------------------------------------------
# Check every extracted observation and target join.
# ------------------------------------------------------------

features = pl.DataFrame(feature_rows)

require(
    features.height == 14869,
    "Extracted observation count mismatch.",
)

require(
    features.select(observation_keys).unique().height
    == features.height,
    "Duplicate Team Context observation keys.",
)

require(
    all(features[column].null_count() == 0
        for column in FEATURE_COLUMNS),
    "Null Team Context feature detected.",
)

require(
    totals["living_t"] == 60744
    and totals["living_ct"] == 63166
    and totals["mapped"] == 123909
    and totals["unknown"] == 1,
    "Extracted player totals differ from Gate 0 evidence.",
)

require(
    totals["unknown_t"] == 0
    and totals["unknown_ct"] == 1,
    "Unexpected unknown-place distribution.",
)

affected = features.filter(
    (pl.col("demo_filename") == AFFECTED_DEMO)
    & (pl.col("round_num") == AFFECTED_ROUND)
    & (pl.col("current_tick") == AFFECTED_TICK)
)

require(
    affected.height == 1,
    "Known missing-place observation was not recovered.",
)

affected_row = affected.row(0, named=True)

require(
    affected_row["v4_t__UNKNOWN_PLACE"] == 0
    and affected_row["v4_ct__UNKNOWN_PLACE"] == 1
    and sum(affected_row[col] for col in FEATURE_COLUMNS[:15]) == 5
    and sum(affected_row[col] for col in FEATURE_COLUMNS[15:30]) == 4,
    "Known missing-place observation was encoded incorrectly.",
)

joined = targets.join(
    features,
    on=observation_keys,
    how="left",
    validate="m:1",
)

require(
    joined.height == targets.height == 29065,
    "Joining Team Context changed the number of target rows.",
)

require(
    all(joined[column].null_count() == 0
        for column in FEATURE_COLUMNS),
    "Some target rows did not receive Team Context features.",
)

affected_joined = joined.filter(
    (pl.col("demo_filename") == AFFECTED_DEMO)
    & (pl.col("round_num") == AFFECTED_ROUND)
    & (pl.col("current_tick") == AFFECTED_TICK)
)

require(
    set(affected_joined["horizon_sec"].to_list()) == {5, 10},
    "Known missing-place observation lacks an expected horizon.",
)

require(
    affected_joined["v4_ct__UNKNOWN_PLACE"].to_list() == [1, 1],
    "The two horizons received inconsistent unknown-place counts.",
)


# ------------------------------------------------------------
# Write only the small independent V4 feature artifact.
# ------------------------------------------------------------

disk_check()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

features = features.sort(observation_keys)

features.write_parquet(OUTPUT)

identity = {
    "version": "V4_DEV_TEAM_CONTEXT_V1",
    "status": "GATE_1B_EXTRACTED_NOT_FROZEN",
    "scope": "53 V3 development matches only",
    "development_manifest_sha256": manifest_sha,
    "development_targets_sha256": targets_sha,
    "macro_mapping_sha256": mapping_sha,
    "v3_b3_feature_protocol_sha256": sha256(B3_PROTOCOL),
    "feature_order": FEATURE_COLUMNS,
    "feature_dimensions": len(FEATURE_COLUMNS),
    "observations": features.height,
    "horizon_specific_joined_rows": joined.height,
    "totals": dict(totals),
    "known_missing_place": {
        "demo_filename": AFFECTED_DEMO,
        "round_num": AFFECTED_ROUND,
        "current_tick": AFFECTED_TICK,
        "t_unknown": 0,
        "ct_unknown": 1,
    },
    "output": str(OUTPUT),
    "output_sha256": sha256(OUTPUT),
    "scientific_boundary": {
        "v3_confirmation_used": False,
        "v2_confirmation_used": False,
        "model_training_performed": False,
        "v3_frozen_artifacts_modified": False,
        "v4_feature_contract_frozen": False,
    },
}

with IDENTITY.open("x") as file:
    json.dump(identity, file, indent=2, ensure_ascii=False)
    file.write("\n")


print("\n" + "=" * 64)
print("V4 GATE 1B — FINAL SUMMARY")
print("=" * 64)

print("Development matches:", len(manifest))
print("Unique current-time observations:", features.height)
print("Horizon-specific rows after join:", joined.height)
print("Feature dimensions:", len(FEATURE_COLUMNS))
print("Living T player rows:", totals["living_t"])
print("Living CT player rows:", totals["living_ct"])
print("Mapped player rows:", totals["mapped"])
print("Unknown T player rows:", totals["unknown_t"])
print("Unknown CT player rows:", totals["unknown_ct"])
print("Known empty-place case: VERIFIED")
print("Occupancy conservation: PASS")
print("Exact current-time joins: PASS")
print("Output:", OUTPUT)
print("Identity:", IDENTITY)
print(f"Free disk: {disk_check():.2f} GiB")

print("\nGATE_1B_DEVELOPMENT_FEATURE_EXTRACTION_PASS")
print("Features are extracted but NOT frozen.")
print("No models were trained or scored.")
print("No confirmation data was accessed.")
