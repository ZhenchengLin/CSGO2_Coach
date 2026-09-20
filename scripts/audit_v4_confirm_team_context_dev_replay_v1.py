"""Replay V4 Team Context on one existing Development demo.

Read-only comparison against frozen Development Team Context.
No confirmation access, model loading, prediction, or scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS
from cs2_tactical_intelligence.v0.timing import open_v0_demo

from cs2_tactical_intelligence import (
    v4_a_confirm_target_semantics_v1 as target,
    v4_a_confirm_team_context_v1 as team,
)


MANIFEST = Path("docs/v3_dev_manifest.csv")
FROZEN_TEAM = Path("data/interim/v4_dev_team_context_v1.parquet")
FIT_CONTRACT = Path("docs/v4_a_confirm_fit_contract_v1_frozen.json")

KEY = ["demo_filename", "round_num", "current_tick"]


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


print("\n=== V4 TEAM CONTEXT DEVELOPMENT REPLAY ===")

require(
    shutil.disk_usage(".").free >= 12 * 1024**3,
    "Free disk below 12 GiB.",
)

fit = json.loads(FIT_CONTRACT.read_text())

manifest_identity = json.loads(
    Path("docs/v3_dev_manifest_identity.json").read_text()
)

require(
    manifest_identity["status"] == "FROZEN"
    and sha256(MANIFEST) == manifest_identity["manifest_sha256"],
    "Development manifest identity mismatch.",
)

require(
    sha256(FROZEN_TEAM)
    == fit["source_sha256"]["team_context"],
    "Frozen Team Context SHA256 mismatch.",
)

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(
    len(manifest) == 53,
    "Expected 53 Development demos.",
)

index = int(os.environ.get("V4_DEV_REPLAY_INDEX", "0"))

require(
    0 <= index < len(manifest),
    "Invalid Development demo index.",
)

record = manifest[index]
name = record["demo_filename"]
demo_path = Path(record["path"])

require(
    demo_path.is_file(),
    f"Missing Development demo: {demo_path}",
)

require(
    sha256(demo_path) == record["sha256"],
    "Raw Development demo SHA256 mismatch.",
)

frozen = (
    pl.read_parquet(FROZEN_TEAM)
    .filter(pl.col("demo_filename") == name)
)

require(
    frozen.height > 0,
    "No frozen Team Context rows for this demo.",
)

require(
    frozen.columns == KEY + team.TEAM_COLUMNS,
    "Unexpected frozen Team Context schema.",
)

expected = {}

for row in frozen.iter_rows(named=True):
    key = (int(row["round_num"]), int(row["current_tick"]))

    require(
        key not in expected,
        f"Duplicate frozen snapshot key: {key}",
    )

    expected[key] = [
        int(row[column])
        for column in team.TEAM_COLUMNS
    ]

print("Development demo:", name)
print("Frozen current-time snapshots:", len(expected))

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

snapshots = target.materialize_snapshots(
    demo,
    set(expected),
)

require(
    set(snapshots) == set(expected),
    "Recovered snapshot keys differ from frozen Team Context.\n"
    f"Missing: {list(set(expected) - set(snapshots))[:5]}\n"
    f"Extra: {list(set(snapshots) - set(expected))[:5]}",
)

unknown_snapshots = 0

for key, players in snapshots.items():
    result = team.build_team_context(players)
    actual_vector = result["vector"]
    expected_vector = expected[key]

    require(
        actual_vector == expected_vector,
        f"32D Team Context mismatch at {name}, {key}\n"
        f"Actual:   {actual_vector}\n"
        f"Expected: {expected_vector}",
    )

    if actual_vector[30] or actual_vector[31]:
        unknown_snapshots += 1

print("Recovered snapshot keys: PASS")
print("32D Team Context row-by-row comparison: PASS")
print("Living-player identity and team-size checks: PASS")
print("Snapshots containing UNKNOWN_PLACE:", unknown_snapshots)

print("\nV4_TEAM_CONTEXT_SINGLE_DEV_DEMO_REPLAY_PASS")
print("No confirmation data was accessed or scored.")
