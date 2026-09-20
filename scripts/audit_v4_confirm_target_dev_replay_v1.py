"""Replay frozen V3 target semantics on one existing Development demo.

Audits retained rows, observation/horizon exclusions, round exclusions,
current bomb XYZ, and +5s/+10s target labels. No model scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import os
from collections import Counter
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
    build_plant_lookup,
)
from cs2_tactical_intelligence.v0.timing import open_v0_demo
from cs2_tactical_intelligence import (
    v4_a_confirm_target_semantics_v1 as target,
)


MANIFEST = Path("docs/v3_dev_manifest.csv")
MANIFEST_IDENTITY = Path("docs/v3_dev_manifest_identity.json")

FROZEN_TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)
TARGET_IDENTITY = Path(
    "docs/v3_dev_target_dataset_identity.json"
)

FROZEN_EXCLUSIONS = Path(
    "data/interim/v3_dev_target_exclusions.csv"
)
FROZEN_ROUND_EXCLUSIONS = Path(
    "data/interim/v3_dev_round_exclusions.csv"
)

TARGET_CONTRACT = Path("docs/v3_future_target_contract.json")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows_from_csv(path, demo_filename):
    with path.open(newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["demo_filename"] == demo_filename
        ]


def exclusion_key(row):
    return (
        int(row["round_num"]),
        int(row["current_nominal_tick"]),
        int(row["horizon_sec"]),
        row["reason"],
    )


def round_exclusion_key(row):
    return (
        int(row["round_num"]),
        row["reason"],
    )


print("\n=== V4 TARGET DEVELOPMENT DEMO REPLAY ===")

free_gib = shutil.disk_usage(".").free / 1024**3
require(free_gib >= 12, f"Free disk below 12 GiB: {free_gib:.2f}")

for path in (
    MANIFEST,
    MANIFEST_IDENTITY,
    FROZEN_TARGETS,
    TARGET_IDENTITY,
    FROZEN_EXCLUSIONS,
    FROZEN_ROUND_EXCLUSIONS,
    TARGET_CONTRACT,
    MAPPING,
):
    require(path.is_file(), f"Missing frozen input: {path}")

manifest_identity = json.loads(MANIFEST_IDENTITY.read_text())
target_identity = json.loads(TARGET_IDENTITY.read_text())
rules = json.loads(TARGET_CONTRACT.read_text())
mapping = json.loads(MAPPING.read_text())

require(
    sha256(MANIFEST) == manifest_identity["manifest_sha256"],
    "Development manifest identity mismatch.",
)
require(
    sha256(FROZEN_TARGETS) == target_identity["output_sha256"],
    "Frozen target dataset identity mismatch.",
)
require(
    sha256(TARGET_CONTRACT) == target_identity["target_contract_sha256"],
    "Frozen target contract identity mismatch.",
)
require(
    sha256(MAPPING) == target_identity["macro_mapping_sha256"],
    "Frozen macro-zone mapping identity mismatch.",
)

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(len(manifest) == 53, "Expected 53 Development demos.")

# Use the same first Development demo as the successful motion replay.
index = int(os.environ.get("V4_DEV_REPLAY_INDEX", "0"))
require(0 <= index < len(manifest), "Invalid Development demo index.")
record = manifest[index]
name = record["demo_filename"]
demo_path = Path(record["path"])

require(demo_path.is_file(), f"Missing Development demo: {demo_path}")

frozen = pl.read_parquet(FROZEN_TARGETS).filter(
    pl.col("demo_filename") == name
)
require(frozen.height > 0, "No frozen target rows for this demo.")

expected_exclusions = Counter(
    exclusion_key(row)
    for row in rows_from_csv(FROZEN_EXCLUSIONS, name)
)
expected_round_exclusions = Counter(
    round_exclusion_key(row)
    for row in rows_from_csv(FROZEN_ROUND_EXCLUSIONS, name)
)

print("Development demo:", name)
print("Frozen target rows:", frozen.height)

demo = open_v0_demo(demo_path, verbose=False)

require(
    demo.header.get("map_name") == "de_mirage",
    "Parsed map is not de_mirage.",
)

demo.parse(
    player_props=[
        *V0_PLAYER_PROPS,
        "last_place_name",
    ]
)

ticks_by_round = target.build_ticks_by_round(demo)
events_by_round = target.build_events_by_round(demo)

plant_lookup, plant_issues = build_plant_lookup(demo)

unresolved_plant_rounds = {
    int(issue["round_num"])
    for issue in plant_issues
    if issue["reason"] == "PLANT_LABEL_UNRESOLVED"
}

drop_lookup, _ = target.build_drop_lookup(demo, name)

start_seconds = int(
    rules["observation_grid"]["first_current_observation_sec"]
)
stride_seconds = int(
    rules["observation_grid"]["stride_sec"]
)
horizons = list(rules["prediction_horizons_sec"])

require(start_seconds == 5, "Unexpected current observation start.")
require(stride_seconds == 5, "Unexpected current observation stride.")
require(horizons == [5, 10], "Unexpected target horizons.")

ticks_per_second = int(rules["tick_clock"]["raw_ticks_per_second"])
require(ticks_per_second == 64, "Unexpected tick rate.")

class_order = list(mapping["zones"])
class_to_index = {
    zone: index for index, zone in enumerate(class_order)
}

require(len(class_order) == 15, "Unexpected target class count.")

timing_records = []
required_snapshot_keys = set()
actual_exclusions = Counter()
actual_round_exclusions = Counter()


def exclude(round_num, nominal, horizon, reason):
    actual_exclusions[
        (int(round_num), int(nominal), int(horizon), reason)
    ] += 1


# Stage 1: independently rebuild all nominal observations and
# horizon-specific timing decisions from the original demo.
for round_row in (
    demo.rounds
    .sort("round_num")
    .iter_rows(named=True)
):
    round_num = int(round_row["round_num"])
    freeze_end = round_row["freeze_end"]
    round_end = round_row["end"]

    if freeze_end is None:
        actual_round_exclusions[
            (round_num, "MISSING_FREEZE_END")
        ] += 1
        continue

    if round_end is None:
        actual_round_exclusions[
            (round_num, "MISSING_ROUND_END")
        ] += 1
        continue

    freeze_end = int(freeze_end)
    round_end = int(round_end)

    if freeze_end >= round_end:
        actual_round_exclusions[
            (round_num, "INVALID_TIMING_ORDER")
        ] += 1
        continue

    if round_num in unresolved_plant_rounds:
        actual_round_exclusions[
            (round_num, "PLANT_LABEL_UNRESOLVED")
        ] += 1
        continue

    plant = plant_lookup.get(round_num)
    plant_tick = None if plant is None else int(plant["plant_tick"])
    plant_label = None if plant is None else str(plant["label"])

    available_ticks = ticks_by_round.get(round_num, [])
    nominal = freeze_end + start_seconds * ticks_per_second
    stride_ticks = stride_seconds * ticks_per_second

    while (
        nominal < round_end
        and (plant_tick is None or nominal < plant_tick)
    ):
        current_resolution = target.resolve_snapshot_tick(
            available_ticks,
            nominal,
            upper_bound_exclusive=round_end,
        )

        if current_resolution is None:
            for horizon in horizons:
                exclude(
                    round_num, nominal, horizon,
                    "CURRENT_SNAPSHOT_UNAVAILABLE",
                )
            nominal += stride_ticks
            continue

        current_tick = int(current_resolution["tick"])

        if plant_tick is not None and current_tick >= plant_tick:
            for horizon in horizons:
                exclude(
                    round_num, nominal, horizon,
                    "CURRENT_STATE_AT_OR_AFTER_PLANT",
                )
            nominal += stride_ticks
            continue

        required_snapshot_keys.add((round_num, current_tick))

        for horizon in horizons:
            target_nominal = (
                current_tick + horizon * ticks_per_second
            )

            if target_nominal >= round_end:
                exclude(
                    round_num, nominal, horizon,
                    "TARGET_AFTER_ROUND_END",
                )
                continue

            future_resolution = target.resolve_snapshot_tick(
                available_ticks,
                target_nominal,
                upper_bound_exclusive=round_end,
            )

            if future_resolution is None:
                exclude(
                    round_num, nominal, horizon,
                    "TARGET_SNAPSHOT_UNAVAILABLE",
                )
                continue

            future_tick = int(future_resolution["tick"])

            required_snapshot_keys.add((round_num, future_tick))

            timing_records.append({
                "round_num": round_num,
                "current_nominal_tick": nominal,
                "current_tick": current_tick,
                "current_lateness": int(
                    current_resolution["lateness"]
                ),
                "horizon_sec": horizon,
                "target_nominal_tick": target_nominal,
                "target_tick": future_tick,
                "target_lateness": int(
                    future_resolution["lateness"]
                ),
                "plant_tick": plant_tick,
                "plant_label": plant_label,
            })

        nominal += stride_ticks


snapshots = target.materialize_snapshots(
    demo,
    required_snapshot_keys,
)


# Stage 2: resolve current and future bomb semantics. Future state
# is used only as the target, never as a current-time feature.
actual_rows = {}

for row in timing_records:
    round_num = row["round_num"]
    nominal = row["current_nominal_tick"]
    horizon = row["horizon_sec"]
    current_tick = row["current_tick"]
    future_tick = row["target_tick"]

    current_snapshot = snapshots.get((round_num, current_tick))
    future_snapshot = snapshots.get((round_num, future_tick))

    if not current_snapshot:
        exclude(
            round_num, nominal, horizon,
            "CURRENT_SNAPSHOT_UNAVAILABLE",
        )
        continue

    if not future_snapshot:
        exclude(
            round_num, nominal, horizon,
            "TARGET_SNAPSHOT_UNAVAILABLE",
        )
        continue

    events = events_by_round.get(round_num, [])

    current = target.resolve_unplanted_bomb(
        round_num=round_num,
        tick=current_tick,
        snapshot=current_snapshot,
        events=events,
        drop_lookup=drop_lookup,
        state_error="CURRENT_BOMB_STATE_UNRESOLVED",
        place_error="CURRENT_BOMB_PLACE_UNRESOLVED",
    )

    if not current["ok"]:
        exclude(round_num, nominal, horizon, current["error"])
        continue

    if row["plant_tick"] is not None and (
        row["plant_tick"] <= future_tick
    ):
        if row["plant_label"] == "A_PLANT":
            future = {
                "ok": True,
                "source": "PLANTED",
                "place": "BombsiteA",
                "macro_zone": "A_SITE_COMPLEX",
            }
        elif row["plant_label"] == "B_PLANT":
            future = {
                "ok": True,
                "source": "PLANTED",
                "place": "BombsiteB",
                "macro_zone": "B_SITE_COMPLEX",
            }
        else:
            raise RuntimeError("Unexpected canonical plant label.")
    else:
        future = target.resolve_unplanted_bomb(
            round_num=round_num,
            tick=future_tick,
            snapshot=future_snapshot,
            events=events,
            drop_lookup=drop_lookup,
            state_error="TARGET_BOMB_STATE_UNRESOLVED",
            place_error="TARGET_BOMB_PLACE_UNRESOLVED",
        )

    if not future["ok"]:
        exclude(round_num, nominal, horizon, future["error"])
        continue

    current_zone = current["macro_zone"]
    future_zone = future["macro_zone"]

    require(
        current_zone in class_to_index
        and future_zone in class_to_index,
        "Unexpected macro-zone.",
    )

    key = (round_num, nominal, horizon)

    require(key not in actual_rows, f"Duplicate target key: {key}")

    actual_rows[key] = {
        "current_tick": current_tick,
        "current_lateness": row["current_lateness"],
        "target_nominal_tick": row["target_nominal_tick"],
        "target_tick": future_tick,
        "target_lateness": row["target_lateness"],
        "current_source": current["source"],
        "current_place": current["place"],
        "current_macro_zone": current_zone,
        "current_class_index": class_to_index[current_zone],
        "current_bomb_X": float(current["X"]),
        "current_bomb_Y": float(current["Y"]),
        "current_bomb_Z": float(current["Z"]),
        "target_source": future["source"],
        "target_place": future["place"],
        "target_macro_zone": future_zone,
        "target_class_index": class_to_index[future_zone],
        "stay_same_zone": current_zone == future_zone,
    }


# Stage 3: compare retained keys, every retained field, and both
# classes of exclusions. A row-count-only match is not sufficient.
frozen_rows = {}

for row in frozen.iter_rows(named=True):
    key = (
        int(row["round_num"]),
        int(row["current_nominal_tick"]),
        int(row["horizon_sec"]),
    )
    require(key not in frozen_rows, f"Duplicate frozen key: {key}")
    frozen_rows[key] = row

require(
    set(actual_rows) == set(frozen_rows),
    "Retained target-row keys differ from frozen V3 dataset.\n"
    f"Only new: {list(set(actual_rows) - set(frozen_rows))[:5]}\n"
    f"Only frozen: {list(set(frozen_rows) - set(actual_rows))[:5]}",
)

for key, actual in actual_rows.items():
    expected = frozen_rows[key]

    for field, value in actual.items():
        frozen_value = expected[field]

        if field in (
            "current_bomb_X",
            "current_bomb_Y",
            "current_bomb_Z",
        ):
            require(
                math.isclose(
                    value, float(frozen_value),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ),
                f"{key} — {field} mismatch: "
                f"{value} != {frozen_value}",
            )
        else:
            require(
                value == frozen_value,
                f"{key} — {field} mismatch: "
                f"{value!r} != {frozen_value!r}",
            )

require(
    actual_exclusions == expected_exclusions,
    "Observation/horizon exclusions differ.\n"
    f"Only new: "
    f"{list((actual_exclusions - expected_exclusions).items())[:5]}\n"
    f"Only frozen: "
    f"{list((expected_exclusions - actual_exclusions).items())[:5]}",
)

require(
    actual_round_exclusions == expected_round_exclusions,
    "Round exclusions differ.\n"
    f"Only new: "
    f"{list((actual_round_exclusions - expected_round_exclusions).items())[:5]}\n"
    f"Only frozen: "
    f"{list((expected_round_exclusions - actual_round_exclusions).items())[:5]}",
)

print("\n=== TARGET REPLAY SUMMARY ===")
print("Retained target rows:", len(actual_rows))
print("Observation/horizon exclusions:", sum(actual_exclusions.values()))
print("Round exclusions:", sum(actual_round_exclusions.values()))
print("Retained row keys and fields: PASS")
print("Current bomb XYZ: PASS")
print("+5s/+10s target labels: PASS")
print("Observation/horizon exclusion identities: PASS")
print("Round exclusion identities: PASS")
print("\nV4_TARGET_SINGLE_DEV_DEMO_REPLAY_PASS")
print("No confirmation data was accessed or scored.")
