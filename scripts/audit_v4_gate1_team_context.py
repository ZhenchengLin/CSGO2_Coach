"""V4 Gate 1A: validate team-context extraction on one development demo."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo


MANIFEST = Path("docs/v3_dev_manifest.csv")
TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")
B3_PROTOCOL = Path("docs/v3_b3_feature_protocol_v1.json")

MIN_FREE_BYTES = 12 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def disk_check():
    free_gib = shutil.disk_usage(".").free / 1024**3
    require(free_gib >= 12, "Free disk space below 12 GiB.")
    return free_gib


mapping = json.loads(MAPPING.read_text())
protocol = json.loads(B3_PROTOCOL.read_text())

zones = protocol["features"]["current_macro_zone"]["categories"]

require(len(zones) == 15, "Expected 15 frozen macro-zones.")
require(
    set(zones) == set(mapping["zones"]),
    "B3 zone order is incompatible with frozen mapping.",
)

place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate fine-place mapping: {place}",
        )
        place_to_zone[place] = zone


def extract_context(players):
    """Return 32 integer features for one exact current-time snapshot."""

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
        0 <= living_t <= 5 and 0 <= living_ct <= 5,
        "Unexpected living player count.",
    )

    vector = (
        [t_counts[z] for z in zones]
        + [ct_counts[z] for z in zones]
        + [unknown_t, unknown_ct]
    )

    require(len(vector) == 32, "Team Context must have 32 features.")

    require(
        sum(vector[:15]) + vector[30] == living_t,
        "T-side occupancy conservation failed.",
    )

    require(
        sum(vector[15:30]) + vector[31] == living_ct,
        "CT-side occupancy conservation failed.",
    )

    require(
        all(isinstance(x, int) and x >= 0 for x in vector),
        "Team Context must contain nonnegative integers.",
    )

    return vector


# ------------------------------------------------------------
# Synthetic tests, including the known empty-place condition.
# ------------------------------------------------------------

normal = [
    {"side": "t", "health": 100, "place": "TSpawn"},
    {"side": "ct", "health": 100, "place": "CTSpawn"},
    {"side": "ct", "health": 0, "place": "Shop"},
]

normal_features = extract_context(normal)

assert normal_features[zones.index("T_SPAWN")] == 1
assert normal_features[15 + zones.index("CT_SPAWN")] == 1
assert normal_features[30:] == [0, 0]

missing_place = [
    {"side": "t", "health": 100, "place": "TopofMid"},
    {"side": "ct", "health": 100, "place": ""},
]

missing_features = extract_context(missing_place)

assert missing_features[30:] == [0, 1]
assert sum(missing_features[:15]) == 1
assert sum(missing_features[15:30]) == 0

print("SYNTHETIC_FEATURE_TESTS_PASS")


# ------------------------------------------------------------
# Select the first existing development demo.
# ------------------------------------------------------------

print("\n=== V4 GATE 1A: SINGLE-DEMO FEATURE AUDIT ===")
print(f"Free disk: {disk_check():.2f} GiB")

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(len(manifest) == 53, "Expected 53 development demos.")

selected = next(
    (
        row
        for row in manifest
        if Path(row["path"]).is_file()
    ),
    None,
)

require(selected is not None, "No development demo available.")

name = selected["demo_filename"]

require(
    selected["map_name"] == "de_mirage"
    and int(selected["raw_ticks_per_sec"]) == 64,
    "Unexpected demo map or timing contract.",
)

targets = pl.read_parquet(
    TARGETS,
    columns=[
        "demo_filename",
        "round_num",
        "current_tick",
        "horizon_sec",
    ],
).filter(
    pl.col("demo_filename") == name
)

require(targets.height > 0, "No matching development targets.")

observations = targets.select(
    "round_num",
    "current_tick",
).unique()

requested = {
    (int(row["round_num"]), int(row["current_tick"]))
    for row in observations.iter_rows(named=True)
}

require(
    len(requested) == observations.height,
    "Current-time observation keys are not unique.",
)

ticks = {tick for _, tick in requested}

print("Demo:", name)
print("Horizon-specific target rows:", targets.height)
print("Unique current observations:", len(requested))
print("Parsing existing development demo...", flush=True)

demo = open_v0_demo(Path(selected["path"]), verbose=False)

require(
    demo.header.get("map_name") == "de_mirage",
    "Parsed map is not Mirage.",
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
    "Missing required current-time player columns.",
)

snapshots = demo.ticks.filter(
    pl.col("tick").is_in(list(ticks))
).select(sorted(required_columns))

features_by_observation = {}

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
        key not in features_by_observation,
        f"Duplicate current-time group: {key}",
    )

    features_by_observation[key] = extract_context(
        group.iter_rows(named=True)
    )

require(
    set(features_by_observation) == requested,
    "Not all exact current-time snapshots were recovered.",
)


# ------------------------------------------------------------
# Join each horizon-specific row to its current observation.
# ------------------------------------------------------------

joined_vectors = {}
horizon_counts = Counter()

for row in targets.iter_rows(named=True):
    observation_key = (
        int(row["round_num"]),
        int(row["current_tick"]),
    )

    horizon = int(row["horizon_sec"])

    require(horizon in (5, 10), "Unexpected prediction horizon.")

    row_key = (*observation_key, horizon)

    require(
        row_key not in joined_vectors,
        f"Duplicate horizon-specific target key: {row_key}",
    )

    # Both horizons refer to the SAME current-time feature vector.
    vector = features_by_observation[observation_key]

    joined_vectors[row_key] = vector
    horizon_counts[horizon] += 1

for observation_key in requested:
    five = joined_vectors.get((*observation_key, 5))
    ten = joined_vectors.get((*observation_key, 10))

    if five is not None and ten is not None:
        require(
            five == ten,
            "The two horizons have different current-time context.",
        )

require(
    len(joined_vectors) == targets.height,
    "Horizon-specific join changed the number of target rows.",
)


print("\n=== GATE 1A VALIDATION ===")
print("Exact observation joins:", len(features_by_observation))
print("Target rows after feature join:", len(joined_vectors))
print("+5s rows:", horizon_counts[5])
print("+10s rows:", horizon_counts[10])
print("Feature dimensions:", len(next(iter(joined_vectors.values()))))
print("Occupancy conservation: PASS")
print("Unknown-place handling: PASS")
print("Horizon consistency: PASS")
print("No future player state used: PASS")

print("\nGATE_1A_SINGLE_DEMO_FEATURE_VALIDATION_PASS")
print("No V3 models were changed or trained.")
print("No confirmation demos were accessed.")
print(f"Free disk after audit: {disk_check():.2f} GiB")
