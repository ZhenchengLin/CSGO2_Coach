"""V4 Gate 1C: validate existing development feature artifact.

No demo parsing, model training, confirmation access, or V3 changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl


FEATURES = Path("data/interim/v4_dev_team_context_v1.parquet")
IDENTITY = Path("docs/v4_dev_team_context_identity_v1.json")
TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
GATE0 = Path("docs/v4_gate0b_development_audit_v1.json")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")
B3 = Path("docs/v3_b3_feature_protocol_v1.json")

REPORT = Path("docs/v4_gate1c_feature_artifact_audit_v1.json")

KEYS = ["demo_filename", "round_num", "current_tick"]


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


print("\n=== V4 GATE 1C: FEATURE ARTIFACT AUDIT ===")

for path in (FEATURES, IDENTITY, TARGETS, GATE0, MAPPING, B3):
    require(path.is_file(), f"Missing input: {path}")

require(
    not REPORT.exists(),
    "Audit report already exists; refusing to overwrite it.",
)

identity = json.loads(IDENTITY.read_text())
gate0 = json.loads(GATE0.read_text())
mapping = json.loads(MAPPING.read_text())
b3 = json.loads(B3.read_text())

zones = b3["features"]["current_macro_zone"]["categories"]

require(
    len(zones) == 15 and set(zones) == set(mapping["zones"]),
    "Frozen macro-zone definitions disagree.",
)

columns = (
    [f"v4_t__{z}" for z in zones]
    + [f"v4_ct__{z}" for z in zones]
    + ["v4_t__UNKNOWN_PLACE", "v4_ct__UNKNOWN_PLACE"]
)

require(
    identity["status"] == "GATE_1B_EXTRACTED_NOT_FROZEN",
    "Unexpected feature artifact status.",
)

require(
    identity["feature_order"] == columns
    and identity["feature_dimensions"] == 32,
    "Feature order or dimensions differ from identity.",
)

require(
    identity["output_sha256"] == sha256(FEATURES),
    "Feature artifact SHA256 mismatch.",
)

require(
    identity["development_targets_sha256"] == sha256(TARGETS),
    "Frozen target dataset SHA256 mismatch.",
)

require(
    identity["macro_mapping_sha256"] == sha256(MAPPING),
    "Frozen macro-zone mapping SHA256 mismatch.",
)

require(
    identity["v3_b3_feature_protocol_sha256"] == sha256(B3),
    "V3 B3 feature protocol SHA256 mismatch.",
)

features = pl.read_parquet(FEATURES)

require(
    features.columns == KEYS + columns,
    "Unexpected feature schema or column ordering.",
)

require(
    features.height == identity["observations"] == 14869,
    "Current-time observation count mismatch.",
)

require(
    features.select(KEYS).unique().height == features.height,
    "Duplicate current-time observation key.",
)

require(
    features["demo_filename"].n_unique() == 53,
    "Expected 53 development matches.",
)

for column in columns:
    require(
        features[column].dtype.is_integer(),
        f"Non-integer feature: {column}",
    )

    require(
        features[column].null_count() == 0,
        f"Null feature: {column}",
    )

    require(
        (features[column] >= 0).all(),
        f"Negative occupancy: {column}",
    )


# Independently recompute team totals from the saved feature vector.

t_total = pl.sum_horizontal(
    *[pl.col(c) for c in columns[:15]],
    pl.col("v4_t__UNKNOWN_PLACE"),
)

ct_total = pl.sum_horizontal(
    *[pl.col(c) for c in columns[15:30]],
    pl.col("v4_ct__UNKNOWN_PLACE"),
)

features = features.with_columns(
    t_total.alias("_living_t"),
    ct_total.alias("_living_ct"),
)

require(
    features.select(
        ((pl.col("_living_t") >= 1)
         & (pl.col("_living_t") <= 5)).all()
    ).item(),
    "T-side living-player total outside 1–5.",
)

require(
    features.select(
        ((pl.col("_living_ct") >= 1)
         & (pl.col("_living_ct") <= 5)).all()
    ).item(),
    "CT-side living-player total outside 1–5.",
)

totals = {
    "observations": features.height,
    "living_t": int(features["_living_t"].sum()),
    "living_ct": int(features["_living_ct"].sum()),
    "mapped": int(
        sum(features[c].sum() for c in columns[:30])
    ),
    "unknown_t": int(features["v4_t__UNKNOWN_PLACE"].sum()),
    "unknown_ct": int(features["v4_ct__UNKNOWN_PLACE"].sum()),
}

require(
    totals == {
        "observations": 14869,
        "living_t": 60744,
        "living_ct": 63166,
        "mapped": 123909,
        "unknown_t": 0,
        "unknown_ct": 1,
    },
    f"Aggregate player counts disagree: {totals}",
)


# Independently compare every match against archived Gate 0 evidence.

for match in gate0["per_match"]:
    part = features.filter(
        pl.col("demo_filename") == match["demo_filename"]
    )

    counts = match["counts"]

    require(
        part.height == counts["requested_snapshots"],
        f"Observation count mismatch: {match['demo_filename']}",
    )

    require(
        part["_living_t"].sum() == counts["alive_t_rows"],
        f"T-side count mismatch: {match['demo_filename']}",
    )

    require(
        part["_living_ct"].sum() == counts["alive_ct_rows"],
        f"CT-side count mismatch: {match['demo_filename']}",
    )

    mapped = sum(
        part[column].sum()
        for column in columns[:30]
    )

    unknown = (
        part["v4_t__UNKNOWN_PLACE"].sum()
        + part["v4_ct__UNKNOWN_PLACE"].sum()
    )

    require(
        mapped == counts["mapped_player_rows"],
        f"Mapped count mismatch: {match['demo_filename']}",
    )

    require(
        unknown == counts.get("unmapped_player_rows", 0),
        f"Unknown-place count mismatch: {match['demo_filename']}",
    )


# Verify the unique missing-place observation.

affected = features.filter(
    (pl.col("demo_filename") ==
     "2026-09-12_iowa-stormboar_vs_sportsbetexpert_mirage.dem")
    & (pl.col("round_num") == 13)
    & (pl.col("current_tick") == 99365)
)

require(
    affected.height == 1,
    "Known missing-place observation was not recovered.",
)

require(
    affected["v4_t__UNKNOWN_PLACE"][0] == 0
    and affected["v4_ct__UNKNOWN_PLACE"][0] == 1
    and affected["_living_t"][0] == 5
    and affected["_living_ct"][0] == 5,
    "Known missing-place observation was encoded incorrectly.",
)


# Verify joins against the existing frozen development targets.

targets = pl.read_parquet(
    TARGETS,
    columns=KEYS + ["horizon_sec"],
)

joined = targets.join(
    features.select(KEYS + columns),
    on=KEYS,
    how="left",
    validate="m:1",
)

require(
    targets.height == joined.height == 29065,
    "Team Context join changed target row count.",
)

require(
    all(joined[c].null_count() == 0 for c in columns),
    "Some target rows lack Team Context.",
)

require(
    joined.select(KEYS + ["horizon_sec"]).unique().height
    == joined.height,
    "Duplicate horizon-specific target keys.",
)

affected_horizons = joined.filter(
    (pl.col("demo_filename") ==
     "2026-09-12_iowa-stormboar_vs_sportsbetexpert_mirage.dem")
    & (pl.col("round_num") == 13)
    & (pl.col("current_tick") == 99365)
)

require(
    set(affected_horizons["horizon_sec"].to_list()) == {5, 10},
    "Known missing-place observation has unexpected horizons.",
)

require(
    affected_horizons["v4_ct__UNKNOWN_PLACE"].to_list() == [1, 1],
    "Known missing-place feature differs between horizons.",
)


# Save only a small, independent V4 audit report.

report = {
    "version": "V4_GATE_1C_FEATURE_ARTIFACT_AUDIT_V1",
    "status": "ARTIFACT_VALIDATED_FEATURE_CONTRACT_NOT_FROZEN",
    "scope": "53 historical development matches",
    "feature_artifact_sha256": sha256(FEATURES),
    "feature_identity_sha256": sha256(IDENTITY),
    "development_targets_sha256": sha256(TARGETS),
    "macro_mapping_sha256": sha256(MAPPING),
    "feature_dimensions": 32,
    "feature_columns": columns,
    "total_counts": totals,
    "joined_target_rows": joined.height,
    "per_match_gate0_reconciliation": "PASS",
    "known_empty_place": "CT unknown count = 1 at round 13, tick 99365",
    "limitations": [
        "This checks a saved development feature artifact.",
        "It does not independently reparse all raw demos.",
        "It does not by itself audit duplicate player identities.",
        "It is not model evaluation or confirmation evidence.",
    ],
}

with REPORT.open("x") as file:
    json.dump(report, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("\n=== V4 GATE 1C — FINAL SUMMARY ===")
print("Development matches: 53")
print("Current-time observations:", features.height)
print("Team Context dimensions:", len(columns))
print("Target rows after join:", joined.height)
print("Independent per-match reconciliation: PASS")
print("Known missing-place case: PASS")
print("Feature artifact SHA256: PASS")
print("Report:", REPORT)
print("\nGATE_1C_FEATURE_ARTIFACT_VALIDATION_PASS")
print("Feature contract remains NOT FROZEN.")
