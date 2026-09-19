"""V4 Gate 0B: audit causal team-state availability across V3 development.

Reads 53 existing development demos and frozen V3 target timestamps.
Does not read confirmation demos, train models, or change V3 artifacts.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
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

REPORT = Path("docs/v4_gate0b_development_audit_v1.json")

MIN_FREE_BYTES = 12 * 1024**3

EXPECTED_ROLES = {
    "V0_DEVELOPMENT": 20,
    "V2_RESERVE": 13,
    "V3_FRESH": 20,
}


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


def check_disk():
    free = shutil.disk_usage(".").free

    require(
        free >= MIN_FREE_BYTES,
        f"Disk below 12 GiB: {free / 1024**3:.2f} GiB",
    )

    return free / 1024**3


def load_json(path):
    return json.loads(path.read_text())


print("\n=== V4 GATE 0B: DEVELOPMENT COVERAGE AUDIT ===")

require(
    not REPORT.exists(),
    "Gate 0B report already exists. Do not overwrite it.",
)

for path in (
    MANIFEST,
    MANIFEST_IDENTITY,
    TARGETS,
    TARGET_IDENTITY,
    MAPPING,
):
    require(path.is_file(), f"Missing required file: {path}")

print(f"Initial free disk: {check_disk():.2f} GiB")

manifest_identity = load_json(MANIFEST_IDENTITY)
target_identity = load_json(TARGET_IDENTITY)
mapping = load_json(MAPPING)

require(
    manifest_identity["status"] == "FROZEN",
    "Development manifest is not frozen.",
)

require(
    target_identity["status"] == "FROZEN",
    "Development target dataset is not frozen.",
)

manifest_sha = sha256(MANIFEST)

require(
    manifest_identity["manifest_sha256"] == manifest_sha,
    "Development manifest SHA mismatch.",
)

require(
    target_identity["development_manifest_sha256"] == manifest_sha,
    "Target dataset references a different development manifest.",
)

require(
    target_identity["output_sha256"] == sha256(TARGETS),
    "Frozen development target dataset changed.",
)

require(
    target_identity["macro_mapping_sha256"] == sha256(MAPPING),
    "Frozen macro-zone mapping changed.",
)

require(
    mapping["status"] == "FROZEN"
    and len(mapping["zones"]) == 15,
    "Unexpected frozen macro-zone mapping.",
)

place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate place in frozen mapping: {place}",
        )

        place_to_zone[place] = zone


with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(
    len(manifest) == 53,
    "Expected exactly 53 development matches.",
)

require(
    Counter(
        row["dev_source_role"]
        for row in manifest
    ) == EXPECTED_ROLES,
    "Development role counts differ from frozen manifest.",
)

require(
    len({row["demo_filename"] for row in manifest}) == 53,
    "Duplicate development demo filename.",
)

require(
    len({row["sha256"] for row in manifest}) == 53,
    "Duplicate development demo SHA.",
)

targets = pl.read_parquet(TARGETS)

require(
    targets.height == target_identity["n_rows"],
    "Development target row count mismatch.",
)

require(
    targets["demo_filename"].n_unique() == 53,
    "Development target dataset does not contain 53 matches.",
)

require(
    set(targets["demo_filename"].unique().to_list())
    == {row["demo_filename"] for row in manifest},
    "Target dataset and development manifest contain different demos.",
)

require(
    set(targets["horizon_sec"].unique().to_list()) == {5, 10},
    "Unexpected target horizons.",
)


# ------------------------------------------------------------
# Audit one development demo at a time.
# ------------------------------------------------------------

audit_rows = []

for index, record in enumerate(manifest, start=1):

    free_gib = check_disk()

    name = record["demo_filename"]
    role = record["dev_source_role"]
    demo_path = Path(record["path"])

    print(
        f"\n[{index:02d}/53] {role} | {name}",
        flush=True,
    )

    require(
        demo_path.is_file(),
        f"Development demo missing: {demo_path}",
    )

    require(
        record["map_name"] == "de_mirage"
        and int(record["raw_ticks_per_sec"]) == 64,
        f"Frozen map/timing contract mismatch: {name}",
    )

    require(
        sha256(demo_path) == record["sha256"],
        f"Raw development demo SHA mismatch: {name}",
    )

    match_targets = targets.filter(
        pl.col("demo_filename") == name
    )

    require(
        match_targets.height > 0,
        f"No development targets for {name}",
    )

    current = match_targets.select(
        "round_num",
        "current_nominal_tick",
        "current_tick",
    ).unique()

    lateness = (
        current["current_tick"]
        - current["current_nominal_tick"]
    )

    require(
        ((lateness >= 0) & (lateness <= 1)).all(),
        f"Current snapshot lateness violates contract: {name}",
    )

    requested = {
        (int(row["round_num"]), int(row["current_tick"]))
        for row in current.iter_rows(named=True)
    }

    require(
        len(requested) == current.height,
        f"Ambiguous current-time keys: {name}",
    )

    ticks = {tick for _, tick in requested}

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )

    require(
        demo.header.get("map_name") == "de_mirage",
        f"Parsed demo map mismatch: {name}",
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
        "X",
        "Y",
        "Z",
        "place",
    }

    require(
        required_columns <= set(demo.ticks.columns),
        f"Player snapshot schema incomplete: {name}",
    )

    snapshots = demo.ticks.filter(
        pl.col("tick").is_in(list(ticks))
    ).select(sorted(required_columns))

    groups = {
        (
            int(group["round_num"][0]),
            int(group["tick"][0]),
        ): group
        for group in snapshots.partition_by(
            ["round_num", "tick"],
            maintain_order=False,
        )
        if (
            int(group["round_num"][0]),
            int(group["tick"][0]),
        ) in requested
    }

    missing = requested - set(groups)

    counters = Counter()
    unknown_places = Counter()
    observed_zones = set()

    counters["target_rows"] = match_targets.height
    counters["requested_snapshots"] = len(requested)
    counters["recovered_snapshots"] = len(groups)
    counters["missing_snapshots"] = len(missing)

    for key in requested:

        snapshot = groups.get(key)

        if snapshot is None:
            continue

        # This is the exact current-time player state,
        # never the future target snapshot.
        alive = snapshot.filter(
            pl.col("side").is_in(["t", "ct"])
            & (pl.col("health") > 0)
        )

        t_count = alive.filter(
            pl.col("side") == "t"
        ).height

        ct_count = alive.filter(
            pl.col("side") == "ct"
        ).height

        counters["alive_t_rows"] += t_count
        counters["alive_ct_rows"] += ct_count
        counters["alive_player_rows"] += alive.height

        if t_count == 0:
            counters["snapshots_without_t"] += 1

        if ct_count == 0:
            counters["snapshots_without_ct"] += 1

        if t_count > 5 or ct_count > 5:
            counters["snapshots_over_team_limit"] += 1

        if t_count + ct_count == 0:
            counters["snapshots_without_alive_players"] += 1

        for player in alive.iter_rows(named=True):

            place = player["place"]

            if place in place_to_zone:
                counters["mapped_player_rows"] += 1
                observed_zones.add(place_to_zone[place])
            else:
                counters["unmapped_player_rows"] += 1
                unknown_places[str(place)] += 1

            xyz_ok = True

            for axis in ("X", "Y", "Z"):

                value = player[axis]

                if value is None:
                    xyz_ok = False
                    break

                try:
                    if not math.isfinite(float(value)):
                        xyz_ok = False
                        break
                except (TypeError, ValueError):
                    xyz_ok = False
                    break

            if not xyz_ok:
                counters["invalid_xyz_player_rows"] += 1

    row = {
        "demo_filename": name,
        "dev_source_role": role,
        "raw_demo_sha256": record["sha256"],
        "counts": dict(counters),
        "observed_macro_zones": sorted(observed_zones),
        "unmapped_places": dict(unknown_places),
    }

    audit_rows.append(row)

    print(
        "  exact snapshots:",
        f"{counters['recovered_snapshots']}/"
        f"{counters['requested_snapshots']}",
    )

    print(
        "  mapped alive players:",
        f"{counters['mapped_player_rows']}/"
        f"{counters['alive_player_rows']}",
    )

    print(
        "  invalid XYZ:",
        counters["invalid_xyz_player_rows"],
    )

    print(
        f"  free disk: {check_disk():.2f} GiB",
        flush=True,
    )

    del demo
    del snapshots
    del groups

    gc.collect()


# ------------------------------------------------------------
# Aggregate coverage and write one small audit report.
# ------------------------------------------------------------

total = Counter()

for row in audit_rows:
    total.update(row["counts"])

observed_all = sorted({
    zone
    for row in audit_rows
    for zone in row["observed_macro_zones"]
})

quality_issues = {
    key: total[key]
    for key in (
        "missing_snapshots",
        "unmapped_player_rows",
        "invalid_xyz_player_rows",
        "snapshots_over_team_limit",
        "snapshots_without_alive_players",
    )
    if total[key] > 0
}

status = (
    "GATE_0B_DEV_COVERAGE_PASS"
    if not quality_issues
    else "GATE_0B_NEEDS_REVIEW"
)

report = {
    "version": "V4_GATE_0B_DEVELOPMENT_AUDIT_V1",
    "status": status,
    "scope": "53 frozen V3 development matches only",
    "scientific_boundary": {
        "v3_confirmation_accessed": False,
        "v2_confirmation_accessed": False,
        "model_training_performed": False,
        "confirmation_scoring_performed": False,
        "v3_frozen_artifacts_modified": False,
        "v4_feature_contract_frozen": False,
    },
    "development_manifest_sha256": manifest_sha,
    "development_targets_sha256": sha256(TARGETS),
    "macro_mapping_sha256": sha256(MAPPING),
    "matches_audited": len(audit_rows),
    "role_counts": dict(EXPECTED_ROLES),
    "total_counts": dict(total),
    "observed_macro_zones": observed_all,
    "quality_issues": quality_issues,
    "per_match": audit_rows,
}

with REPORT.open("x") as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )
    file.write("\n")


print("\n" + "=" * 64)
print("V4 GATE 0B — FINAL SUMMARY")
print("=" * 64)

print("Status:", status)
print("Development matches:", len(audit_rows))
print("Target rows:", total["target_rows"])
print("Requested snapshots:", total["requested_snapshots"])
print("Recovered snapshots:", total["recovered_snapshots"])
print("Missing snapshots:", total["missing_snapshots"])
print("Alive T rows:", total["alive_t_rows"])
print("Alive CT rows:", total["alive_ct_rows"])
print("Mapped player rows:", total["mapped_player_rows"])
print("Unmapped player rows:", total["unmapped_player_rows"])
print("Invalid XYZ rows:", total["invalid_xyz_player_rows"])
print(
    "Snapshots without living T:",
    total["snapshots_without_t"],
)
print(
    "Snapshots without living CT:",
    total["snapshots_without_ct"],
)
print("Observed macro-zones:", len(observed_all), "/ 15")
print("Quality issues:", quality_issues)
print("Report:", REPORT)
print(f"Free disk: {check_disk():.2f} GiB")

print(
    "\nDevelopment-only feasibility audit complete. "
    "No V4 feature contract or model has been frozen."
)
