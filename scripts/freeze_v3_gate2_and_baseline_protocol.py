from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl


# ============================================================
# Inputs
# ============================================================

TARGET_CONTRACT = Path(
    "docs/v3_future_target_contract.json"
)

MACRO_MAPPING = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

INTEGRITY_SUMMARY = Path(
    "docs/v3_target_integrity_summary.json"
)

DISTRIBUTION_AUDIT = Path(
    "docs/v3_target_distribution_audit.json"
)

OBSERVATIONS = Path(
    "data/interim/v3_target_integrity_observations.csv"
)

ROUND_EXCLUSIONS = Path(
    "data/interim/v3_target_integrity_round_exclusions.csv"
)


# ============================================================
# Outputs
# ============================================================

GATE2_FREEZE = Path(
    "docs/v3_gate2_target_freeze.json"
)

BASELINE_PROTOCOL = Path(
    "docs/v3_baseline_protocol_v1.json"
)


HORIZONS = [
    5,
    10,
]

N_ZONES = 15


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# Presence / overwrite protection
# ============================================================

inputs = [
    TARGET_CONTRACT,
    MACRO_MAPPING,
    INTEGRITY_SUMMARY,
    DISTRIBUTION_AUDIT,
    OBSERVATIONS,
    ROUND_EXCLUSIONS,
]


for path in inputs:
    require(
        path.exists(),
        f"Missing required Gate 2 evidence: {path}",
    )


require(
    not GATE2_FREEZE.exists(),
    (
        "Refusing to overwrite frozen Gate 2 record: "
        f"{GATE2_FREEZE}"
    ),
)

require(
    not BASELINE_PROTOCOL.exists(),
    (
        "Refusing to overwrite baseline protocol v1: "
        f"{BASELINE_PROTOCOL}"
    ),
)


# ============================================================
# Load records
# ============================================================

target_contract = json.loads(
    TARGET_CONTRACT.read_text()
)

macro_mapping = json.loads(
    MACRO_MAPPING.read_text()
)

integrity = json.loads(
    INTEGRITY_SUMMARY.read_text()
)

distribution = json.loads(
    DISTRIBUTION_AUDIT.read_text()
)


require(
    target_contract["status"]
    == "FROZEN",
    "Future target contract is not frozen.",
)

require(
    macro_mapping["status"]
    == "FROZEN",
    "Macro-zone mapping is not frozen.",
)

require(
    integrity["status"]
    == "AUDIT_COMPLETE",
    "Gate 2D integrity audit is incomplete.",
)

require(
    distribution["status"]
    == "TARGET_DISTRIBUTION_AUDIT_COMPLETE",
    "Gate 2E distribution audit is incomplete.",
)

require(
    integrity[
        "training_performed"
    ]
    is False,
    "Training occurred during Gate 2D.",
)

require(
    distribution[
        "training_performed"
    ]
    is False,
    "Training occurred during Gate 2E.",
)

require(
    integrity[
        "v2_d_confirm_used"
    ]
    is False,
    "V2 D_CONFIRM was used during Gate 2D.",
)

require(
    distribution[
        "v2_d_confirm_used"
    ]
    is False,
    "V2 D_CONFIRM was used during Gate 2E.",
)


# ============================================================
# Read observations
# ============================================================

obs = pl.read_csv(
    OBSERVATIONS,
    infer_schema_length=None,
)

valid = obs.filter(
    pl.col("valid")
    == True
)

invalid = obs.filter(
    pl.col("valid")
    == False
)


require(
    obs.height
    == 7432,
    (
        "Gate 2 candidate-row count changed: "
        f"{obs.height}"
    ),
)

require(
    valid.height
    == 6964,
    (
        "Gate 2 valid-row count changed: "
        f"{valid.height}"
    ),
)

require(
    invalid.height
    == 468,
    (
        "Gate 2 excluded-row count changed: "
        f"{invalid.height}"
    ),
)


# ============================================================
# Exclusion integrity
# ============================================================

exclusion_reasons = sorted(
    invalid[
        "exclusion_reason"
    ]
    .drop_nulls()
    .unique()
    .to_list()
)


require(
    exclusion_reasons
    == [
        "TARGET_AFTER_ROUND_END",
    ],
    (
        "Unexpected Gate 2 exclusion reasons: "
        f"{exclusion_reasons}"
    ),
)


# ============================================================
# Horizon coverage
# ============================================================

coverage_by_horizon = {}

persistence_by_horizon = {}

move_rate_by_horizon = {}

zone_support_by_horizon = {}


for horizon in HORIZONS:

    horizon_all = obs.filter(
        pl.col(
            "horizon_sec"
        )
        == horizon
    )

    horizon_valid = valid.filter(
        pl.col(
            "horizon_sec"
        )
        == horizon
    )


    n_candidate = (
        horizon_all.height
    )

    n_valid = (
        horizon_valid.height
    )

    coverage = (
        n_valid
        / n_candidate
    )


    n_stay = (
        horizon_valid
        .filter(
            pl.col(
                "stay_same_zone"
            )
            == True
        )
        .height
    )

    n_move = (
        n_valid
        - n_stay
    )


    persistence_rate = (
        n_stay
        / n_valid
    )

    move_rate = (
        n_move
        / n_valid
    )


    zone_counts = (
        horizon_valid
        .group_by(
            "target_macro_zone"
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )


    zones_observed = set(
        zone_counts[
            "target_macro_zone"
        ]
        .to_list()
    )


    require(
        len(
            zones_observed
        )
        == N_ZONES,
        (
            f"+{horizon}s does not contain "
            "all 15 target zones."
        ),
    )


    rare_under_20 = (
        zone_counts
        .filter(
            pl.col("len")
            < 20
        )
        .select([
            "target_macro_zone",
            "len",
        ])
        .to_dicts()
    )


    coverage_by_horizon[
        str(
            horizon
        )
    ] = {
        "candidate_rows":
            n_candidate,

        "valid_rows":
            n_valid,

        "coverage":
            coverage,
    }


    persistence_by_horizon[
        str(
            horizon
        )
    ] = persistence_rate


    move_rate_by_horizon[
        str(
            horizon
        )
    ] = move_rate


    zone_support_by_horizon[
        str(
            horizon
        )
    ] = {
        "zones_observed":
            len(
                zones_observed
            ),

        "rare_under_20":
            rare_under_20,
    }


# ============================================================
# Round exclusions
# ============================================================

round_exclusions = pl.read_csv(
    ROUND_EXCLUSIONS,
    infer_schema_length=None,
)


round_exclusion_counts = (
    round_exclusions
    .group_by(
        "reason"
    )
    .len()
    .sort(
        "len",
        descending=True,
    )
    .to_dicts()
)


# ============================================================
# Evidence hashes
# ============================================================

evidence_sha256 = {
    str(path):
        sha256_file(path)
    for path in inputs
}


# ============================================================
# Gate 2 freeze record
# ============================================================

gate2_record = {
    "version":
        "V3",

    "gate":
        "2",

    "artifact":
        "Near-future bomb-zone target definition",

    "status":
        "FROZEN",

    "frozen_date":
        "2026-09-15",

    "research_target":
        (
            "Predict the bomb tactical macro-zone "
            "at +5s and +10s from legitimate "
            "state available at time t."
        ),

    "label_space": {
        "map":
            "de_mirage",

        "macro_mapping_version":
            "v1",

        "n_classes":
            N_ZONES,

        "all_classes_observed_at_5s":
            True,

        "all_classes_observed_at_10s":
            True,

        "rare_class_policy":
            (
                "Rare target classes remain part of the "
                "frozen 15-zone representation. Gate 1 "
                "mapping must not be modified merely to "
                "increase class support."
            ),
    },

    "observation_design": {
        "start":
            "freeze_end + 5 seconds",

        "stride_sec":
            5,

        "horizons_sec":
            HORIZONS,

        "horizons_scored_separately":
            True,

        "post_plant_current_states":
            False,

        "future_target_may_cross_plant":
            True,
    },

    "integrity": {
        "candidate_horizon_rows":
            obs.height,

        "valid_horizon_rows":
            valid.height,

        "excluded_horizon_rows":
            invalid.height,

        "overall_coverage":
            (
                valid.height
                / obs.height
            ),

        "coverage_by_horizon":
            coverage_by_horizon,

        "observed_exclusion_reasons":
            exclusion_reasons,

        "semantic_or_snapshot_integrity_failures":
            0,

        "round_exclusions":
            round_exclusion_counts,
    },

    "target_semantics": {
        "carried":
            "Valve place of current C4 carrier",

        "pickup_fallback":
            (
                "Latest pickup identifies carrier; "
                "Valve place comes from the current snapshot."
            ),

        "dropped":
            (
                "Frozen exact-tick Valve place of the "
                "player who dropped the C4."
            ),

        "planted":
            (
                "Canonical valid plant event maps "
                "BombsiteA/B to the corresponding "
                "site macro-zone."
            ),

        "xyz_classifier_used_for_ground_truth":
            False,
    },

    "descriptive_task_difficulty": {
        "hard_persistence_accuracy":
            persistence_by_horizon,

        "move_rate":
            move_rate_by_horizon,

        "zone_support":
            zone_support_by_horizon,

        "interpretation":
            (
                "Persistence is strong, especially at +5s. "
                "The +10s target contains substantially more "
                "cross-zone movement. These are descriptive "
                "properties, not reasons to modify the "
                "frozen target definition."
            ),
    },

    "scientific_boundary": {
        "v2_d_confirm_used":
            False,

        "v2_reserve_13_role":
            (
                "Gate 0-2 semantic and target-design audit "
                "corpus only. These demos must never become "
                "V3 confirmation data."
            ),

        "mapping_changes_after_freeze":
            (
                "Require a new explicit mapping version."
            ),

        "target_contract_changes_after_freeze":
            (
                "Require a new explicit target-contract "
                "version. Silent mutation is forbidden."
            ),
    },

    "evidence_sha256":
        evidence_sha256,
}


GATE2_FREEZE.write_text(
    json.dumps(
        gate2_record,
        indent=2,
    )
    + "\n"
)


# ============================================================
# Baseline protocol v1
# ============================================================

baseline_protocol = {
    "version":
        "V3",

    "artifact":
        "Near-future tactical forecasting baseline protocol",

    "protocol_version":
        "v1",

    "status":
        "FROZEN",

    "frozen_date":
        "2026-09-15",

    "prediction_tasks": {
        "5s":
            "15-class bomb macro-zone forecast at t + 5s",

        "10s":
            "15-class bomb macro-zone forecast at t + 10s",

        "rule":
            (
                "The two horizons are evaluated separately. "
                "Do not combine them into one primary score."
            ),
    },

    "primary_metric": {
        "name":
            "multiclass_log_loss",

        "evaluation":
            "separately for +5s and +10s",

        "reason":
            (
                "The model must provide useful probability "
                "distributions rather than only hard labels."
            ),
    },

    "secondary_metrics": [
        "accuracy",
        "macro_f1",
        "per_zone_recall_with_support",
        "top2_accuracy",
        "multiclass_brier_score",
        "calibration_diagnostics",
        "equal_match_log_loss",
    ],

    "evaluation_invariants": {
        "split_unit":
            "match",

        "random_row_split_allowed":
            False,

        "same_match_in_train_and_test_allowed":
            False,

        "preprocessing_fit_on_test_data_allowed":
            False,

        "target_horizons_evaluated_separately":
            True,

        "report_row_weighted_metrics":
            True,

        "report_equal_match_metrics":
            True,

        "uncertainty_unit":
            "match",

        "bootstrap_unit":
            "match",
    },

    "probability_adapter_for_deterministic_baselines": {
        "purpose":
            (
                "Convert a deterministic baseline into a "
                "finite 15-class probability distribution "
                "without validation-set tuning."
            ),

        "training_only":
            True,

        "confidence_formula":
            (
                "q = (n_correct_train + 1) / "
                "(n_train + 2)"
            ),

        "predicted_class_probability":
            "q",

        "each_other_class_probability":
            "(1 - q) / 14",

        "note":
            (
                "This is Laplace-smoothed calibration based "
                "only on the training partition for the "
                "corresponding horizon."
            ),
    },

    "baselines": {
        "B0_persistence": {
            "hard_prediction":
                (
                    "future_macro_zone = "
                    "current_macro_zone"
                ),

            "probability_output":
                (
                    "Use the frozen deterministic-baseline "
                    "probability adapter estimated from the "
                    "training partition separately for each "
                    "horizon."
                ),

            "current_design_corpus_hard_accuracy": {
                "5s":
                    persistence_by_horizon[
                        "5"
                    ],

                "10s":
                    persistence_by_horizon[
                        "10"
                    ],
            },

            "role":
                (
                    "Minimum temporal baseline. Any learned "
                    "forecasting method must be interpreted "
                    "relative to this strong persistence "
                    "behavior."
                ),
        },

        "B1_constant_motion": {
            "state":
                (
                    "Use bomb XYZ at t and approximately "
                    "one second earlier."
                ),

            "velocity":
                (
                    "Estimate recent 3D bomb displacement "
                    "per second."
                ),

            "projection":
                (
                    "Project the bomb position forward by "
                    "the requested horizon."
                ),

            "projected_xyz_to_zone":
                (
                    "Use an inference-only XYZ nearest-"
                    "neighbor semantic resolver fitted on "
                    "training-match player samples only, "
                    "then apply the frozen 15-zone mapping."
                ),

            "ground_truth_uses_xyz_resolver":
                False,

            "probability_output":
                (
                    "Apply the deterministic-baseline "
                    "probability adapter using B1 training "
                    "accuracy for the corresponding horizon."
                ),

            "leakage_rule":
                (
                    "No target-match semantic samples may "
                    "be used to fit the XYZ resolver."
                ),
        },

        "B2_zone_markov": {
            "definition":
                (
                    "Estimate P(target_macro_zone | "
                    "current_macro_zone, horizon) from the "
                    "training partition."
                ),

            "smoothing":
                (
                    "Add-one Dirichlet/Laplace smoothing "
                    "over all 15 target classes."
                ),

            "fit_separately_by_horizon":
                True,

            "test_rows_used_for_transition_estimation":
                False,

            "probability_output":
                "native 15-class transition distribution",
        },

        "B3_tabular_map_aware": {
            "status":
                "FUTURE_STAGE",

            "rule":
                (
                    "Feature set must use only information "
                    "available at or before current time t."
                ),

            "feature_selection_using_confirmation_data":
                False,
        },
    },

    "rare_class_reporting": {
        "merge_classes_due_only_to_low_support":
            False,

        "always_report_per_class_support":
            True,

        "macro_f1_interpretation":
            (
                "Treat macro F1 cautiously when one or more "
                "zones have very small held-out support."
            ),

        "current_design_corpus_rare_classes":
            zone_support_by_horizon,
    },

    "data_boundary": {
        "v2_d_confirm":
            "SEALED_AND_FORBIDDEN",

        "13_v2_reserves":
            (
                "Design/integrity evidence only. Do not use "
                "as V3 confirmation."
            ),

        "D_V3_DEV":
            (
                "Acquire and freeze a new chronological "
                "development corpus before baseline fitting "
                "and model selection."
            ),

        "D_V3_CONFIRM":
            (
                "Acquire and freeze separately only after "
                "V3 development decisions are complete."
            ),
    },

    "next_action":
        "FREEZE_D_V3_DEV_MANIFEST",
}


BASELINE_PROTOCOL.write_text(
    json.dumps(
        baseline_protocol,
        indent=2,
    )
    + "\n"
)


# ============================================================
# Report
# ============================================================

print("=" * 104)
print("V3 GATE 2 FREEZE + BASELINE PROTOCOL V1")
print("=" * 104)

print()
print("GATE 2 TARGET INTEGRITY")
print("-" * 104)

print(
    "Candidate horizon rows:",
    f"{obs.height:,}",
)

print(
    "Valid horizon rows:",
    f"{valid.height:,}",
)

print(
    "Coverage:",
    f"{valid.height / obs.height:.4%}",
)

print(
    "Exclusion reasons:",
    exclusion_reasons,
)

print(
    "Semantic/snapshot integrity failures:",
    0,
)


print()
print("TASK CHARACTERISTICS")
print("-" * 104)

print(
    "+5s persistence:",
    f"{persistence_by_horizon['5']:.4%}",
)

print(
    "+5s move rate:",
    f"{move_rate_by_horizon['5']:.4%}",
)

print(
    "+10s persistence:",
    f"{persistence_by_horizon['10']:.4%}",
)

print(
    "+10s move rate:",
    f"{move_rate_by_horizon['10']:.4%}",
)

print(
    "15/15 zones represented at both horizons:",
    True,
)


print()
print("BASELINE LADDER")
print("-" * 104)

print(
    "B0 = persistence"
)

print(
    "B1 = constant motion"
)

print(
    "B2 = zone-transition / Markov"
)

print(
    "B3 = tabular map-aware model"
)


print()
print("SCORING")
print("-" * 104)

print(
    "Primary metric: multiclass log loss"
)

print(
    "Horizons scored separately: +5s, +10s"
)

print(
    "Split unit: match"
)

print(
    "Random row split allowed:",
    False,
)

print(
    "Equal-match robustness required:",
    True,
)


print()
print("DATA BOUNDARY")
print("-" * 104)

print(
    "V2 D_CONFIRM:",
    "SEALED"
)

print(
    "13 reserves:",
    "design/integrity only"
)

print(
    "Next corpus:",
    "new chronological D_V3_DEV"
)


print()
print("=" * 104)

print(
    "✅ V3 GATE 2 CLOSED"
)

print(
    "✅ BASELINE PROTOCOL V1 FROZEN"
)

print(
    "NEXT_ACTION=FREEZE_D_V3_DEV_MANIFEST"
)

print("=" * 104)

print()
print("Artifacts:")
print(" ", GATE2_FREEZE)
print(" ", BASELINE_PROTOCOL)
