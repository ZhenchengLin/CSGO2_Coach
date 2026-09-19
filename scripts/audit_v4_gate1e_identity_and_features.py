"""V4 Gate 1E: full-development player identity and feature audit.

Read-only for existing research artifacts.
Parses the 53 historical development demos once.
Does not use confirmation data or train models.
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

FEATURES = Path("data/interim/v4_dev_team_context_v1.parquet")
FEATURE_IDENTITY = Path("docs/v4_dev_team_context_identity_v1.json")

MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")
B3_PROTOCOL = Path("docs/v3_b3_feature_protocol_v1.json")

GATE0 = Path("docs/v4_gate0b_development_audit_v1.json")
GATE1C = Path("docs/v4_gate1c_feature_artifact_audit_v1.json")

REPORT = Path("docs/v4_gate1e_identity_and_features_audit_v1.json")

KEYS = ["demo_filename", "round_num", "current_tick"]
MIN_FREE_BYTES = 12 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def digest(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def free_gib():
    free = shutil.disk_usage(".").free

    require(
        free >= MIN_FREE_BYTES,
        f"Free disk below 12 GiB: {free / 1024**3:.2f} GiB",
    )

    return free / 1024**3


def load_json(path):
    return json.loads(path.read_text())


print("\n=== V4 GATE 1E: IDENTITY AND FEATURE AUDIT ===")
print(f"Initial free disk: {free_gib():.2f} GiB")

require(
    not REPORT.exists(),
    f"Report already exists; refusing to overwrite: {REPORT}",
)

for path in (
    MANIFEST,
    MANIFEST_IDENTITY,
    FEATURES,
    FEATURE_IDENTITY,
    MAPPING,
    B3_PROTOCOL,
    GATE0,
    GATE1C,
):
    require(path.is_file(), f"Missing input: {path}")


# ------------------------------------------------------------
# Validate previously frozen and audited inputs.
# ------------------------------------------------------------

manifest_identity = load_json(MANIFEST_IDENTITY)
feature_identity = load_json(FEATURE_IDENTITY)
mapping = load_json(MAPPING)
b3 = load_json(B3_PROTOCOL)
gate0 = load_json(GATE0)
gate1c = load_json(GATE1C)

manifest_sha = digest(MANIFEST)
feature_sha = digest(FEATURES)
mapping_sha = digest(MAPPING)

require(
    manifest_identity["manifest_sha256"] == manifest_sha,
    "Development manifest SHA256 mismatch.",
)

require(
    feature_identity["development_manifest_sha256"] == manifest_sha,
    "Feature artifact references a different development manifest.",
)

require(
    feature_identity["output_sha256"] == feature_sha,
    "Existing Team Context parquet SHA256 mismatch.",
)

require(
    feature_identity["macro_mapping_sha256"] == mapping_sha,
    "Feature artifact references a different macro-zone mapping.",
)

require(
    gate1c["feature_artifact_sha256"] == feature_sha
    and gate1c["status"]
    == "ARTIFACT_VALIDATED_FEATURE_CONTRACT_NOT_FROZEN",
    "Gate 1C evidence does not match the existing feature artifact.",
)

require(
    gate0["matches_audited"] == 53,
    "Unexpected Gate 0 match coverage.",
)

zones = b3["features"]["current_macro_zone"]["categories"]

require(
    len(zones) == 15
    and set(zones) == set(mapping["zones"]),
    "Frozen B3 zones and macro-zone mapping disagree.",
)

place_to_zone = {}

for zone, places in mapping["zones"].items():
    for place in places:
        require(
            place not in place_to_zone,
            f"Duplicate fine-place mapping: {place}",
        )
        place_to_zone[place] = zone

feature_columns = (
    [f"v4_t__{zone}" for zone in zones]
    + [f"v4_ct__{zone}" for zone in zones]
    + ["v4_t__UNKNOWN_PLACE", "v4_ct__UNKNOWN_PLACE"]
)

require(
    feature_identity["feature_order"] == feature_columns,
    "Unexpected Team Context feature ordering.",
)

features = pl.read_parquet(FEATURES)

require(
    features.columns == KEYS + feature_columns,
    "Unexpected saved feature schema.",
)

require(
    features.height == 14869
    and features.select(KEYS).unique().height == features.height,
    "Unexpected saved observation coverage.",
)

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(
    len(manifest) == 53
    and len({row["demo_filename"] for row in manifest}) == 53,
    "Unexpected development manifest.",
)

gate0_by_demo = {
    row["demo_filename"]: row
    for row in gate0["per_match"]
}

require(
    len(gate0_by_demo) == 53,
    "Gate 0 does not contain 53 distinct matches.",
)


# ------------------------------------------------------------
# Independent current-time feature calculation.
# ------------------------------------------------------------

def inspect_snapshot(players):
    counts_t = Counter()
    counts_ct = Counter()

    unknown_t = 0
    unknown_ct = 0

    living_t = 0
    living_ct = 0

    ids = []

    missing_ids = 0

    for player in players:
        side = player["side"]
        health = player["health"]

        if side not in ("t", "ct"):
            continue

        if health is None or health <= 0:
            continue

        if side == "t":
            living_t += 1
            counts = counts_t
        else:
            living_ct += 1
            counts = counts_ct

        raw_id = player["steamid"]

        if (
            raw_id is None
            or str(raw_id).strip() in ("", "0", "0.0")
        ):
            missing_ids += 1
        else:
            ids.append(str(raw_id))

        zone = place_to_zone.get(player["place"])

        if zone is None:
            if side == "t":
                unknown_t += 1
            else:
                unknown_ct += 1
        else:
            counts[zone] += 1

    duplicate_excess = sum(
        count - 1
        for count in Counter(ids).values()
        if count > 1
    )

    vector = (
        [counts_t[zone] for zone in zones]
        + [counts_ct[zone] for zone in zones]
        + [unknown_t, unknown_ct]
    )

    require(
        len(vector) == 32,
        "Recomputed feature vector has incorrect dimensions.",
    )

    require(
        sum(vector[:15]) + vector[30] == living_t,
        "T-side occupancy conservation failed.",
    )

    require(
        sum(vector[15:30]) + vector[31] == living_ct,
        "CT-side occupancy conservation failed.",
    )

    return {
        "vector": vector,
        "living_t": living_t,
        "living_ct": living_ct,
        "missing_ids": missing_ids,
        "duplicate_excess": duplicate_excess,
    }


# ------------------------------------------------------------
# Parse each development demo exactly once.
# ------------------------------------------------------------

all_totals = Counter()
per_match = []
issue_examples = []

for index, record in enumerate(manifest, start=1):

    name = record["demo_filename"]
    path = Path(record["path"])

    print(
        f"\n[{index:02d}/53] {name}",
        flush=True,
    )

    print(
        f"  Free disk: {free_gib():.2f} GiB",
        flush=True,
    )

    require(
        path.is_file(),
        f"Missing development demo: {path}",
    )

    require(
        record["map_name"] == "de_mirage"
        and int(record["raw_ticks_per_sec"]) == 64,
        f"Unexpected map/tick contract: {name}",
    )

    # Verify that this is still the exact manifested raw demo.
    require(
        digest(path) == record["sha256"],
        f"Raw demo SHA256 mismatch: {name}",
    )

    saved = features.filter(
        pl.col("demo_filename") == name
    )

    require(
        saved.height > 0,
        f"No saved Team Context for {name}",
    )

    expected = {}

    for row in saved.iter_rows(named=True):
        key = (
            int(row["round_num"]),
            int(row["current_tick"]),
        )

        require(
            key not in expected,
            f"Duplicate saved observation key: {name} {key}",
        )

        expected[key] = [
            int(row[column])
            for column in feature_columns
        ]

    requested_ticks = {tick for _, tick in expected}

    demo = open_v0_demo(path, verbose=False)

    require(
        demo.header.get("map_name") == "de_mirage",
        f"Parsed demo is not Mirage: {name}",
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
        "steamid",
        "side",
        "health",
        "place",
    }

    require(
        required_columns <= set(demo.ticks.columns),
        f"Missing player identity fields: {name}",
    )

    snapshots = demo.ticks.filter(
        pl.col("tick").is_in(list(requested_ticks))
    ).select(sorted(required_columns))

    seen = set()
    match_totals = Counter()

    for group in snapshots.partition_by(
        ["round_num", "tick"],
        maintain_order=False,
    ):

        key = (
            int(group["round_num"][0]),
            int(group["tick"][0]),
        )

        if key not in expected:
            continue

        require(
            key not in seen,
            f"Duplicate snapshot group: {name} {key}",
        )

        seen.add(key)

        result = inspect_snapshot(
            group.iter_rows(named=True)
        )

        vector = result["vector"]

        match_totals["snapshots"] += 1
        match_totals["living_t"] += result["living_t"]
        match_totals["living_ct"] += result["living_ct"]

        match_totals["missing_ids"] += result["missing_ids"]
        match_totals["duplicate_excess"] += result["duplicate_excess"]

        match_totals["mapped"] += sum(vector[:30])
        match_totals["unknown_t"] += vector[30]
        match_totals["unknown_ct"] += vector[31]

        if result["missing_ids"] > 0:
            match_totals["snapshots_with_missing_ids"] += 1

        if result["duplicate_excess"] > 0:
            match_totals["snapshots_with_duplicate_ids"] += 1

        if not (
            1 <= result["living_t"] <= 5
            and 1 <= result["living_ct"] <= 5
        ):
            match_totals["invalid_team_size_snapshots"] += 1

        if vector != expected[key]:
            match_totals["feature_mismatch_snapshots"] += 1

        if (
            result["missing_ids"] > 0
            or result["duplicate_excess"] > 0
            or vector != expected[key]
            or not (
                1 <= result["living_t"] <= 5
                and 1 <= result["living_ct"] <= 5
            )
        ) and len(issue_examples) < 20:
            issue_examples.append({
                "demo_filename": name,
                "round_num": key[0],
                "current_tick": key[1],
                "missing_ids": result["missing_ids"],
                "duplicate_excess": result["duplicate_excess"],
                "feature_mismatch": vector != expected[key],
                "living_t": result["living_t"],
                "living_ct": result["living_ct"],
            })

    match_totals["missing_snapshots"] = len(
        set(expected) - seen
    )

    reference = gate0_by_demo[name]["counts"]

    require(
        match_totals["snapshots"]
        == reference["recovered_snapshots"],
        f"Snapshot count differs from Gate 0: {name}",
    )

    require(
        match_totals["living_t"] == reference["alive_t_rows"],
        f"Living T count differs from Gate 0: {name}",
    )

    require(
        match_totals["living_ct"] == reference["alive_ct_rows"],
        f"Living CT count differs from Gate 0: {name}",
    )

    require(
        match_totals["mapped"] == reference["mapped_player_rows"],
        f"Mapped-player count differs from Gate 0: {name}",
    )

    require(
        match_totals["unknown_t"] + match_totals["unknown_ct"]
        == reference.get("unmapped_player_rows", 0),
        f"Unknown-place count differs from Gate 0: {name}",
    )

    all_totals.update(match_totals)

    per_match.append({
        "demo_filename": name,
        "dev_source_role": record["dev_source_role"],
        "counts": dict(match_totals),
    })

    print(
        "  Snapshots:",
        match_totals["snapshots"],
        "| Missing IDs:",
        match_totals["missing_ids"],
        "| Duplicate IDs:",
        match_totals["duplicate_excess"],
        "| Feature mismatches:",
        match_totals["feature_mismatch_snapshots"],
        flush=True,
    )

    del demo
    del snapshots

    gc.collect()


# ------------------------------------------------------------
# Final checks and independent audit report.
# ------------------------------------------------------------

require(
    all_totals["snapshots"] == features.height == 14869,
    "Full-development observation count mismatch.",
)

require(
    all_totals["living_t"] == 60744
    and all_totals["living_ct"] == 63166,
    "Full-development living-player totals mismatch.",
)

require(
    all_totals["mapped"] == 123909
    and all_totals["unknown_t"] == 0
    and all_totals["unknown_ct"] == 1,
    "Full-development place mapping totals mismatch.",
)

issues = {
    key: all_totals.get(key, 0)
    for key in (
        "missing_snapshots",
        "missing_ids",
        "duplicate_excess",
        "snapshots_with_missing_ids",
        "snapshots_with_duplicate_ids",
        "invalid_team_size_snapshots",
        "feature_mismatch_snapshots",
    )
    if all_totals.get(key, 0) > 0
}

status = (
    "GATE_1E_IDENTITY_AND_FEATURE_RECONCILIATION_PASS"
    if not issues
    else "GATE_1E_NEEDS_REVIEW"
)

report = {
    "version": "V4_GATE_1E_IDENTITY_AND_FEATURE_AUDIT_V1",
    "status": status,
    "scope": "53 V3 historical development matches only",
    "development_manifest_sha256": manifest_sha,
    "team_context_sha256": feature_sha,
    "macro_mapping_sha256": mapping_sha,
    "matches_audited": len(per_match),
    "feature_dimensions": len(feature_columns),
    "totals": dict(all_totals),
    "issues": issues,
    "issue_examples": issue_examples,
    "per_match": per_match,
    "limitations": [
        "This is a development-data audit.",
        "The audit checks nonempty, nonzero steamid and "
        "uniqueness within each current-time snapshot.",
        "It does not establish cross-demo player identity.",
        "No model training or independent confirmation "
        "was performed.",
    ],
    "scientific_boundary": {
        "v2_confirmation_used": False,
        "v3_confirmation_used": False,
        "model_training_performed": False,
        "v3_frozen_artifacts_modified": False,
        "v4_feature_contract_frozen": False,
    },
}

with REPORT.open("x") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("\n" + "=" * 64)
print("V4 GATE 1E — FINAL SUMMARY")
print("=" * 64)

print("Status:", status)
print("Matches audited:", len(per_match))
print("Current-time observations:", all_totals["snapshots"])
print("Living T rows:", all_totals["living_t"])
print("Living CT rows:", all_totals["living_ct"])
print("Missing/placeholder player IDs:", all_totals["missing_ids"])
print("Duplicate player ID excess rows:", all_totals["duplicate_excess"])
print("Feature mismatch snapshots:", all_totals["feature_mismatch_snapshots"])
print("Missing snapshots:", all_totals["missing_snapshots"])
print("Unknown T places:", all_totals["unknown_t"])
print("Unknown CT places:", all_totals["unknown_ct"])
print("Issues:", issues)
print("Report:", REPORT)
print(f"Free disk: {free_gib():.2f} GiB")

print("\n" + status)
print("V4 feature contract remains NOT FROZEN.")
print("No confirmation data or model training was used.")
