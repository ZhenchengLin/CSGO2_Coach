"""Freeze V4-A confirmation feature/target extraction rules.

This script reads frozen metadata and model identities only.
It does not parse demos, construct confirmation features, or score models.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SOURCES = {
    "acquisition_protocol": Path(
        "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json"
    ),
    "model_freeze": Path(
        "docs/v4_a_confirm_model_freeze_v1.json"
    ),
    "fit_contract": Path(
        "docs/v4_a_confirm_fit_contract_v1_frozen.json"
    ),
    "v3_feature_protocol": Path(
        "docs/v3_b3_feature_protocol_v1.json"
    ),
    "v4_feature_contract": Path(
        "docs/v4_team_context_feature_contract_v1_frozen.json"
    ),
    "target_contract": Path(
        "docs/v3_future_target_contract.json"
    ),
    "macro_mapping": Path(
        "docs/v3_macro_zone_mapping_v1_frozen.json"
    ),
    "v3_target_builder": Path(
        "scripts/build_v3_confirm_candidate_targets.py"
    ),
    "v4_team_builder": Path(
        "scripts/build_v4_dev_team_context_v1.py"
    ),
    "v4_identity_auditor": Path(
        "scripts/audit_v4_gate1e_identity_and_features.py"
    ),
}

OUTPUT = Path(
    "docs/v4_a_confirm_feature_extraction_contract_v1_frozen.json"
)


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


def load_json(path):
    return json.loads(path.read_text())


print("\n=== V4-A GATE 6D: FEATURE EXTRACTION CONTRACT ===")

free_gib = shutil.disk_usage(".").free / 1024**3

require(
    free_gib >= 12,
    f"Free disk below 12 GiB: {free_gib:.2f}",
)

for name, path in SOURCES.items():
    require(path.is_file(), f"Missing {name}: {path}")

require(
    not OUTPUT.exists(),
    f"Contract already exists: {OUTPUT}",
)

acquisition = load_json(SOURCES["acquisition_protocol"])
freeze = load_json(SOURCES["model_freeze"])
fit = load_json(SOURCES["fit_contract"])
v3 = load_json(SOURCES["v3_feature_protocol"])
v4 = load_json(SOURCES["v4_feature_contract"])
target = load_json(SOURCES["target_contract"])
mapping = load_json(SOURCES["macro_mapping"])

require(
    acquisition["status"]
    == "FROZEN_BEFORE_V4_CONFIRMATION_ACQUISITION",
    "Acquisition protocol is not frozen.",
)

require(
    acquisition["provenance"]["model_freeze_sha256"]
    == sha256(SOURCES["model_freeze"]),
    "Acquisition/model freeze identity mismatch.",
)

require(
    freeze["status"] == "FOUR_FULL_DEVELOPMENT_MODELS_FITTED"
    and freeze["confirmation_data_accessed"] is False
    and freeze["confirmation_scoring_performed"] is False,
    "Unexpected model freeze state.",
)

require(
    freeze["fit_contract_sha256"]
    == sha256(SOURCES["fit_contract"]),
    "Model/fitting contract identity mismatch.",
)

require(
    freeze["source_artifacts"]["feature_contract"]["sha256"]
    == sha256(SOURCES["v4_feature_contract"]),
    "Frozen V4 feature identity mismatch.",
)

require(
    v3["status"] == "FROZEN_BEFORE_FIRST_B3_FIT"
    and v4["status"] == "FROZEN_FOR_V4_A_DEVELOPMENT"
    and target["status"] == "FROZEN",
    "An input protocol is not frozen.",
)

require(
    target["evidence"]["macro_mapping_sha256"]
    == sha256(SOURCES["macro_mapping"]),
    "Target/mapping identity mismatch.",
)

control_order = fit["model_variants"]["control"]["feature_order"]
candidate_order = fit["model_variants"]["candidate"]["feature_order"]
team_order = v4["team_context_feature_order"]

require(
    control_order == v3["feature_order"]
    and candidate_order == control_order + team_order,
    "Frozen feature order mismatch.",
)

require(
    len(control_order) == 24
    and len(team_order) == 32
    and len(candidate_order) == 56,
    "Unexpected feature dimensions.",
)

class_order = fit["development_population"]["target_classes"]

require(
    len(class_order) == 15
    and class_order == list(mapping["zones"])
    and class_order == v4["macro_zone_order"],
    "Macro-zone/class order mismatch.",
)

require(
    target["prediction_horizons_sec"] == [5, 10]
    and target["tick_clock"]["raw_ticks_per_second"] == 64,
    "Frozen target timing mismatch.",
)

expected_models = {
    "control_plus5": 24,
    "candidate_plus5": 56,
    "control_plus10": 24,
    "candidate_plus10": 56,
}

require(
    set(freeze["models"]) == set(expected_models),
    "Unexpected final-model inventory.",
)

for name, dimensions in expected_models.items():
    info = freeze["models"][name]
    path = Path(info["artifact"])

    require(
        info["feature_dimensions"] == dimensions
        and path.is_file()
        and sha256(path) == info["sha256"],
        f"Frozen model identity mismatch: {name}",
    )

require(
    acquisition["temporal_boundary"]["earliest_allowed_match_date"]
    == "2026-09-20",
    "Unexpected new-data date boundary.",
)

print("Frozen protocol identities: PASS")
print("Four final model SHA256 values: PASS")
print("Control/Candidate feature order: PASS")


contract = {
    "version": "V4_A_CONFIRM_FEATURE_EXTRACTION_CONTRACT_V1",
    "status": "FROZEN_BEFORE_CONFIRMATION_FEATURE_EXTRACTION",

    "research_scope": {
        "dataset": "D_V4_A_CONFIRM",
        "map": "de_mirage",
        "earliest_match_date": "2026-09-20",
        "prediction_horizons_sec": [5, 10],
        "target_classes": class_order,
        "new_confirmation_only": True,
        "historical_v2_v3_confirmation_reuse": False,
    },

    "source_of_truth": {
        "acquisition": str(SOURCES["acquisition_protocol"]),
        "target_semantics": str(SOURCES["target_contract"]),
        "macro_zone_mapping": str(SOURCES["macro_mapping"]),
        "control_features": str(SOURCES["v3_feature_protocol"]),
        "team_context": str(SOURCES["v4_feature_contract"]),
        "model_fitting": str(SOURCES["fit_contract"]),
        "model_freeze": str(SOURCES["model_freeze"]),
    },

    "observation_and_target_rules": {
        "reuse_frozen_v3_target_contract": True,
        "raw_ticks_per_second": 64,
        "round_anchor": "round.freeze_end",
        "first_current_observation_seconds_after_anchor": 5,
        "current_observation_stride_seconds": 5,
        "current_snapshot_resolution": (
            "First available player snapshot at or after the nominal "
            "current tick, with at most +1 raw tick lateness."
        ),
        "current_must_be_preplant": True,
        "current_must_be_before_round_end": True,
        "target_nominal_tick": (
            "resolved_current_tick + horizon_sec * 64"
        ),
        "target_snapshot_resolution": (
            "First available player snapshot at or after the nominal "
            "target tick, with at most +1 raw tick lateness."
        ),
        "target_must_be_before_round_end": True,
        "target_semantics": (
            "Use the frozen V3 target-state priority, frozen 15-zone "
            "mapping, and frozen round/observation/horizon exclusions."
        ),
        "future_information_as_feature": False,
        "future_information_as_target_only": True,
        "candidate_eligibility_requires_nonzero_rows_for_both_horizons": True,
        "no_posthoc_row_or_round_removal": True,
    },

    "data_keys": {
        "target_row": [
            "demo_filename",
            "round_num",
            "current_nominal_tick",
            "horizon_sec",
        ],
        "motion_join": [
            "demo_filename",
            "round_num",
            "current_nominal_tick",
            "current_tick",
        ],
        "team_context_join": [
            "demo_filename",
            "round_num",
            "current_tick",
        ],
        "one_team_context_vector_per_unique_current_snapshot": True,
        "same_snapshot_must_have_identical_team_context_for_both_horizons": True,
        "joins_must_not_duplicate_or_drop_eligible_target_rows": True,
    },

    "control_24d": {
        "dimensions": 24,
        "feature_order": control_order,
        "zone_encoding": "Frozen 15-class full one-hot encoding.",
        "bomb_state_encoding": (
            "CARRIED_INVENTORY and CARRIED_PICKUP_FALLBACK map to "
            "CARRIED; DROPPED maps to DROPPED."
        ),
        "xyz": "Current bomb XYZ, raw coordinates.",
        "velocity": (
            "Reproduce the frozen V3 B3 one-second causal motion "
            "construction. Do not use positions or events after "
            "the resolved current tick."
        ),
        "speed": "sqrt(vx^2 + vy^2 + vz^2)",
        "feature_dtype": "float32",
        "speed_clipping": False,
        "new_imputation_or_scaling": False,
        "no_future_state": True,
    },

    "team_context_32d": {
        "dimensions": 32,
        "feature_order": team_order,
        "snapshot": (
            "Exactly the resolved current-time player snapshot "
            "associated with the eligible target observation."
        ),
        "living_player": "side in {t, ct} and health > 0",
        "identity": (
            "For every counted living player, require a usable "
            "steamid and uniqueness within the snapshot."
        ),
        "team_size": "Require 1 to 5 living players on each side.",
        "counts": (
            "Count living T and CT players separately in each "
            "frozen macro-zone, in frozen feature order."
        ),
        "unknown_place": (
            "If a living player's current place is unmapped, "
            "increment that player's team-specific UNKNOWN_PLACE "
            "count. Unknown place alone does not exclude a match."
        ),
        "occupancy_invariants": [
            "All 32 counts are nonnegative integers.",
            "sum(T 15 zones) + T UNKNOWN_PLACE == living T count.",
            "sum(CT 15 zones) + CT UNKNOWN_PLACE == living CT count.",
            "No player is counted more than once in one snapshot.",
        ],
        "time_cutoff": "at_or_before_resolved_current_tick",
        "future_player_state_allowed": False,
        "future_bomb_state_allowed": False,
    },

    "candidate_56d": {
        "dimensions": 56,
        "feature_order": candidate_order,
        "construction": (
            "Append the exact frozen 32D Team Context vector "
            "to the identical frozen 24D Control vector."
        ),
        "first_24_features_equal_control": True,
        "feature_dtype": "float32",
        "missing_feature_imputation": False,
    },

    "technical_integrity": {
        "required": [
            "Unique target-row keys.",
            "Unique resolved-current-snapshot keys within each demo.",
            "Unique motion and Team Context join keys.",
            "All expected target and feature fields available.",
            "No unresolved motion among retained target rows.",
            "No missing or nonfinite model input values.",
            "Every eligible target class index belongs to 0..14.",
            "Independent player identity and occupancy verification.",
            "Exactly 24 Control and 56 Candidate input columns.",
            "Exact feature ordering and identical target rows for both models.",
        ],
        "failure_handling": (
            "Apply only the frozen target-level exclusions and "
            "the frozen acquisition protocol's technical eligibility "
            "rules. Do not silently drop individual rows because "
            "they are difficult to predict or change class balance. "
            "If an unanticipated failure has no predeclared handling "
            "rule, stop and document it before model scoring."
        ),
        "unexpected_failure_policy": (
            "Do not invent a new exclusion or change a frozen feature "
            "definition after viewing confirmation predictions."
        ),
    },

    "execution_order": [
        "Freeze and commit the deterministic candidate metadata queue.",
        "Acquire and technically audit candidates in frozen rank order.",
        "Build candidate target rows using frozen V3 target semantics.",
        "Build and independently audit current-time motion and Team Context.",
        "Apply the frozen acquisition eligibility rules.",
        "Select and freeze the first 20 technically eligible matches.",
        "Verify the selected corpus and its exact feature/target identities.",
        "Only after the final manifest and scoring protocol are frozen, "
        "run frozen-model predictions and confirmation scoring.",
    ],

    "output_policy": {
        "new_v4_paths_only": True,
        "overwrite_historical_v2_v3_data": False,
        "overwrite_development_artifacts": False,
        "overwrite_frozen_models": False,
        "target_output": (
            "data/interim/v4_a_confirm_targets_v1.parquet"
        ),
        "motion_output": (
            "data/interim/v4_a_confirm_motion_v1.parquet"
        ),
        "team_context_output": (
            "data/interim/v4_a_confirm_team_context_v1.parquet"
        ),
        "identity_record": (
            "docs/v4_a_confirm_feature_identity_v1.json"
        ),
        "all_outputs_require_sha256_and_row_count": True,
        "no_model_prediction_during_extraction": True,
        "no_confirmation_metric_during_extraction": True,
    },

    "provenance": {
        name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name, path in SOURCES.items()
    },

    "implementation_boundary": {
        "v4_specific_extractor_written": False,
        "confirmation_demos_downloaded_by_this_script": False,
        "confirmation_features_extracted_by_this_script": False,
        "confirmation_predictions_computed_by_this_script": False,
        "confirmation_scoring_performed_by_this_script": False,
    },

    "revision_policy": (
        "Do not overwrite this frozen V1 contract. "
        "Any changed extraction or integrity rule requires "
        "an explicit versioned amendment before confirmation scoring."
    ),
}

with OUTPUT.open("x") as file:
    json.dump(contract, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("\n=== GATE 6D SUMMARY ===")
print("Current-time feature dimensions: 24D Control / 56D Candidate")
print("Frozen target horizons: +5s and +10s")
print("Team Context identity and occupancy rules: RECORDED")
print("Frozen source SHA256 values: RECORDED")
print("New confirmation demos: NOT ACQUIRED")
print("Feature extraction: NOT PERFORMED")
print("Model prediction/scoring: NOT PERFORMED")
print("Contract:", OUTPUT)
print("\nV4_A_CONFIRM_FEATURE_EXTRACTION_CONTRACT_FROZEN")
