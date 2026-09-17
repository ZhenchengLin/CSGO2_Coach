from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl


MANIFEST = Path("docs/v3_dev_manifest.csv")
TARGETS = Path("data/processed/v3_dev_targets_v1.parquet")
EXCLUSIONS = Path("data/interim/v3_dev_target_exclusions.csv")
ROUND_EXCLUSIONS = Path("data/interim/v3_dev_round_exclusions.csv")
DROP_AUDIT = Path("data/interim/v3_dev_drop_semantic_audit.csv")

ACQUISITION_V2 = Path("docs/v3_dev_acquisition_protocol_v2.json")
GATE2 = Path("docs/v3_gate2_target_freeze.json")
TARGET_CONTRACT = Path("docs/v3_future_target_contract.json")
BASELINE_PROTOCOL = Path("docs/v3_baseline_protocol_v1.json")

AMENDMENT = Path("docs/v3_dev_data_role_amendment_v2.json")
REGRESSION = Path("docs/v3_gate2_reserve_regression.json")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


for path in [
    MANIFEST,
    TARGETS,
    EXCLUSIONS,
    ROUND_EXCLUSIONS,
    DROP_AUDIT,
    ACQUISITION_V2,
    GATE2,
    TARGET_CONTRACT,
    BASELINE_PROTOCOL,
]:
    require(path.exists(), f"Missing required artifact: {path}")


manifest = pl.read_csv(MANIFEST, infer_schema_length=None)
targets = pl.read_parquet(TARGETS)
exclusions = pl.read_csv(EXCLUSIONS, infer_schema_length=None)
round_exclusions = pl.read_csv(ROUND_EXCLUSIONS, infer_schema_length=None)
drop_audit = pl.read_csv(DROP_AUDIT, infer_schema_length=None)


# ============================================================
# Exact role inventory
# ============================================================

role_counts = {
    row["dev_source_role"]: int(row["len"])
    for row in (
        manifest
        .group_by("dev_source_role")
        .len()
        .iter_rows(named=True)
    )
}

require(role_counts.get("V0_DEVELOPMENT") == 20, "Expected 20 V0 development demos.")
require(role_counts.get("V2_RESERVE") == 13, "Expected 13 V2 reserves.")
require(role_counts.get("V3_FRESH") == 20, "Expected 20 fresh V3 demos.")
require(manifest.height == 53, "Expected 53 V3 development demos.")


# ============================================================
# Freeze data-role amendment
# ============================================================

amendment = {
    "version": "V3_DEV_DATA_ROLE_AMENDMENT_V2",
    "status": "FROZEN",
    "scope": "Acquisition/data-role clauses only",
    "supersedes_for_scope": str(ACQUISITION_V2),
    "superseded_document_sha256": sha256_file(ACQUISITION_V2),

    "unchanged": [
        "Gate 0 semantic seed",
        "Gate 1 frozen 15-zone macro mapping",
        "Gate 2 future target contract",
        "Gate 2 baseline protocol",
        "Frozen V3 target semantics",
        "Frozen V3 grouped-match CV policy",
    ],

    "decision_timing": {
        "storage_aware_revision_declared_before_v3_fresh_acquisition": True,
        "declared_before_v3_model_fitting": True,
        "note": (
            "This document records and formalizes the earlier storage-aware "
            "development-role decision; it does not claim the document itself "
            "was authored before acquisition."
        ),
    },

    "roles": {
        "V0_DEVELOPMENT": {
            "n": 20,
            "v3_model_development_eligible": "conditional_on_compatibility_audit",
            "unbiased_representation_or_target_validation": False,
            "reason": (
                "Historical development data may be reused for model development "
                "but already influenced earlier project development."
            ),
        },

        "V2_RESERVE": {
            "n": 13,
            "prior_role": "Gate 0-2 design/audit evidence",
            "v3_model_development_eligible": "conditional_on_compatibility_audit",
            "unbiased_representation_or_target_validation": False,
            "reason": (
                "These matches were used for V3 semantic/target design audits "
                "and therefore are not independent validation evidence."
            ),
        },

        "V3_FRESH": {
            "n": 20,
            "selection": "chronological pre-fit fresh acquisition",
            "v3_model_development_eligible": "conditional_on_compatibility_audit",
        },

        "V2_D_CONFIRM": {
            "n": 30,
            "v3_use": "FORBIDDEN",
            "reason": "Permanent sealed V2 confirmation set.",
        },
    },

    "development_pool": {
        "maximum_matches": 53,
        "split_unit": "match",
        "random_row_split_forbidden": True,
    },

    "future_confirmation": {
        "name": "D_V3_CONFIRM",
        "must_be_later_than_all_development_matches": True,
        "untouched_until_v3_choices_frozen": True,
        "development_feedback_forbidden": True,
    },
}


if AMENDMENT.exists():
    existing = json.loads(AMENDMENT.read_text())
    require(existing == amendment, "Existing data-role amendment differs.")
else:
    AMENDMENT.write_text(json.dumps(amendment, indent=2) + "\n")


# ============================================================
# Gate-2 reserve subset regression
# ============================================================

reserve_names = (
    manifest
    .filter(pl.col("dev_source_role") == "V2_RESERVE")
    ["demo_filename"]
    .to_list()
)

require(len(reserve_names) == 13, "Expected exactly 13 reserve names.")


reserve_targets = targets.filter(
    pl.col("demo_filename").is_in(reserve_names)
)

reserve_exclusions = exclusions.filter(
    pl.col("demo_filename").is_in(reserve_names)
)

reserve_round_exclusions = round_exclusions.filter(
    pl.col("demo_filename").is_in(reserve_names)
)

reserve_drop_audit = drop_audit.filter(
    pl.col("demo_filename").is_in(reserve_names)
)


# Frozen Gate-2 expected totals from the original 13 reserves.
expected = {
    "candidate_horizon_rows": 7432,
    "valid_rows": 6964,
    "excluded_rows": 468,
    "valid_plus5": 3564,
    "valid_plus10": 3400,
    "after_round_end_plus5": 152,
    "after_round_end_plus10": 316,
    "round_exclusions_missing_freeze_end": 4,
    "drop_events_total": 657,
    "drop_events_resolved": 657,
}


actual = {
    "candidate_horizon_rows":
        reserve_targets.height + reserve_exclusions.height,

    "valid_rows":
        reserve_targets.height,

    "excluded_rows":
        reserve_exclusions.height,

    "valid_plus5":
        reserve_targets.filter(pl.col("horizon_sec") == 5).height,

    "valid_plus10":
        reserve_targets.filter(pl.col("horizon_sec") == 10).height,

    "after_round_end_plus5":
        reserve_exclusions.filter(
            (pl.col("horizon_sec") == 5)
            & (pl.col("reason") == "TARGET_AFTER_ROUND_END")
        ).height,

    "after_round_end_plus10":
        reserve_exclusions.filter(
            (pl.col("horizon_sec") == 10)
            & (pl.col("reason") == "TARGET_AFTER_ROUND_END")
        ).height,

    "round_exclusions_missing_freeze_end":
        reserve_round_exclusions.filter(
            pl.col("reason") == "MISSING_FREEZE_END"
        ).height,

    "drop_events_total":
        reserve_drop_audit.height,

    "drop_events_resolved":
        reserve_drop_audit.filter(
            pl.col("resolved") == True
        ).height
        if "resolved" in reserve_drop_audit.columns
        else None,
}


# Some historical audit schemas used status instead of resolved.
if actual["drop_events_resolved"] is None:
    if "status" in reserve_drop_audit.columns:
        actual["drop_events_resolved"] = reserve_drop_audit.filter(
            pl.col("status") == "RESOLVED"
        ).height
    else:
        raise RuntimeError(
            "Cannot identify resolved drop rows in drop audit schema."
        )


for key, expected_value in expected.items():
    require(
        actual[key] == expected_value,
        f"Gate-2 reserve regression failed for {key}: "
        f"expected {expected_value}, got {actual[key]}",
    )


# No non-boundary exclusions are expected in the original reserve seed.
non_boundary = reserve_exclusions.filter(
    pl.col("reason") != "TARGET_AFTER_ROUND_END"
)

require(
    non_boundary.height == 0,
    f"Unexpected non-boundary reserve exclusions: {non_boundary.height}",
)


# Target-source counts are also part of the original Gate-2 fingerprint.
source_counts = {}

for horizon in [5, 10]:
    frame = reserve_targets.filter(pl.col("horizon_sec") == horizon)

    counts = {
        row["target_source"]: int(row["len"])
        for row in (
            frame
            .group_by("target_source")
            .len()
            .iter_rows(named=True)
        )
    }

    source_counts[str(horizon)] = counts


expected_sources = {
    "5": {
        "CARRIED_INVENTORY": 2428,
        "DROPPED": 1008,
        "PLANTED": 128,
    },
    "10": {
        "CARRIED_INVENTORY": 2260,
        "DROPPED": 896,
        "PLANTED": 244,
    },
}

require(
    source_counts == expected_sources,
    f"Gate-2 reserve target-source regression failed: {source_counts}",
)


regression = {
    "version": "V3_GATE2_RESERVE_REGRESSION_V1",
    "status": "PASS",
    "purpose": (
        "Verify that the combined 53-match target builder reproduces "
        "the previously frozen 13-reserve Gate-2 target semantics exactly."
    ),

    "reserve_matches": 13,
    "expected": expected,
    "actual": actual,
    "target_source_counts": source_counts,

    "semantic_conclusion": (
        "The original 13-reserve Gate-2 target behavior is reproduced "
        "exactly; no target-contract change is required."
    ),

    "target_dataset_sha256": sha256_file(TARGETS),
    "manifest_sha256": sha256_file(MANIFEST),
    "target_contract_sha256": sha256_file(TARGET_CONTRACT),
    "baseline_protocol_sha256": sha256_file(BASELINE_PROTOCOL),
    "v2_d_confirm_used": False,
}


if REGRESSION.exists():
    existing = json.loads(REGRESSION.read_text())
    require(existing == regression, "Existing reserve regression differs.")
else:
    REGRESSION.write_text(json.dumps(regression, indent=2) + "\n")


print("=" * 108)
print("V3 DATA-ROLE AMENDMENT")
print("=" * 108)
print("V0 development:", role_counts["V0_DEVELOPMENT"])
print("V2 reserves:", role_counts["V2_RESERVE"])
print("V3 fresh:", role_counts["V3_FRESH"])
print("D_V3_DEV maximum:", manifest.height)
print("V2 D_CONFIRM:", "FORBIDDEN")
print()

print("=" * 108)
print("GATE-2 RESERVE REGRESSION")
print("=" * 108)

for key in expected:
    print(
        f"{key:<42}"
        f"expected={expected[key]:>7} "
        f"actual={actual[key]:>7}"
    )

print()
print("Target sources:")
print(json.dumps(source_counts, indent=2))

print()
print("✅ V3 DATA-ROLE AMENDMENT FROZEN")
print("✅ 13-RESERVE GATE-2 REGRESSION PASS")
print("✅ V2 D_CONFIRM REMAINS FORBIDDEN")
print("NEXT_ACTION=REBUILD_EXACT_B1_SPATIAL_CACHE")
