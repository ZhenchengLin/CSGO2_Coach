"""Close V4-A development without fitting or scoring models."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


FILES = {
    "feature_contract": Path(
        "docs/v4_team_context_feature_contract_v1_frozen.json"
    ),
    "experiment_protocol": Path(
        "docs/v4_a_model_experiment_protocol_v1_frozen.json"
    ),
    "experiment_results": Path(
        "docs/v4_a_development_oof_results_v1.json"
    ),
    "gate5a": Path(
        "docs/v4_a_development_error_analysis_v1.json"
    ),
    "gate5b": Path(
        "docs/v4_a_movement_destination_analysis_v1.json"
    ),
    "oof_predictions": Path(
        "data/interim/v4_a_control_vs_team_oof_v1.parquet"
    ),
    "gate4b_auditor": Path(
        "scripts/audit_v4_a_development_oof_v1.py"
    ),
}

OUTPUT = Path(
    "docs/v4_a_development_closure_v1.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def close(actual, expected, label, tolerance=1e-7):
    require(
        abs(float(actual) - float(expected)) <= tolerance,
        f"{label}: {actual} != {expected}",
    )


print("\n=== V4-A DEVELOPMENT CLOSURE PREFLIGHT ===")

free_gib = shutil.disk_usage(".").free / 1024**3

require(
    free_gib >= 12,
    f"Free disk below 12 GiB: {free_gib:.2f}",
)

require(
    not OUTPUT.exists(),
    f"Closure already exists; refusing to overwrite: {OUTPUT}",
)

for name, path in FILES.items():
    require(
        path.is_file(),
        f"Missing {name}: {path}",
    )

feature = read_json(FILES["feature_contract"])
protocol = read_json(FILES["experiment_protocol"])
experiment = read_json(FILES["experiment_results"])
gate5a = read_json(FILES["gate5a"])
gate5b = read_json(FILES["gate5b"])

hashes = {
    name: sha256(path)
    for name, path in FILES.items()
}

require(
    feature["status"] == "FROZEN_FOR_V4_A_DEVELOPMENT",
    "Team Context contract is not frozen.",
)

require(
    protocol["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "Model experiment protocol is not frozen.",
)

require(
    experiment["status"] == "DEVELOPMENT_OOF_COMPLETE",
    "Development OOF experiment is incomplete.",
)

require(
    gate5a["status"] == "EXPLORATORY_DEVELOPMENT_ANALYSIS"
    and gate5b["status"] == "EXPLORATORY_DEVELOPMENT_ANALYSIS",
    "Development analysis is incomplete.",
)

require(
    experiment["experiment_protocol_sha256"]
    == hashes["experiment_protocol"],
    "Experiment/protocol identity mismatch.",
)

require(
    experiment["oof_predictions_sha256"]
    == hashes["oof_predictions"],
    "OOF artifact identity mismatch.",
)

require(
    gate5a["source_experiment_sha256"]
    == hashes["experiment_results"]
    and gate5a["source_oof_sha256"]
    == hashes["oof_predictions"],
    "Gate 5A provenance mismatch.",
)

require(
    gate5b["source_experiment_sha256"]
    == hashes["experiment_results"]
    and gate5b["source_gate5a_sha256"]
    == hashes["gate5a"]
    and gate5b["source_oof_sha256"]
    == hashes["oof_predictions"]
    and gate5b["source_model_protocol_sha256"]
    == hashes["experiment_protocol"],
    "Gate 5B provenance mismatch.",
)

require(
    protocol["provenance"]["v4_feature_protocol_sha256"]
    == hashes["feature_contract"],
    "Frozen feature contract identity mismatch.",
)

require(
    protocol["control"]["dimensions"] == 24
    and protocol["candidate"]["dimensions"] == 56,
    "Unexpected model feature dimensions.",
)

require(
    experiment["development_matches"] == 53
    and experiment["development_target_rows"] == 29065,
    "Unexpected development corpus size.",
)

require(
    gate5a["analysis_scope"]["target_rows"] == 29065
    and gate5b["analysis_scope"]["target_rows"] == 29065,
    "Development analysis row counts disagree.",
)

print("Artifact identities and provenance: PASS")


# ------------------------------------------------------------
# Reconcile the three recorded analyses before closure.
# ------------------------------------------------------------

summary = {}

for horizon, expected_rows in ((5, 14869), (10, 14196)):
    key = str(horizon)

    original = experiment["results_by_horizon"][key]
    analysis5a = gate5a["by_horizon"][key]
    analysis5b = gate5b["by_horizon"][key]

    control = original["control"]
    candidate = original["candidate"]
    paired = original["paired_comparison"]

    require(
        control["rows"] == candidate["rows"] == expected_rows,
        f"+{horizon}s OOF row mismatch.",
    )

    require(
        analysis5a["overall"]["rows"]
        == analysis5b["movement_event_all_rows"]["rows"]
        == expected_rows,
        f"+{horizon}s analysis row mismatch.",
    )

    close(
        candidate["log_loss"] - control["log_loss"],
        paired["row_weighted_mean"],
        f"+{horizon}s paired Log Loss",
    )

    close(
        analysis5a["overall"][
            "candidate_minus_control_log_loss"
        ],
        paired["row_weighted_mean"],
        f"+{horizon}s Gate 5A overall delta",
    )

    close(
        analysis5b["original_overall_multiclass_delta"],
        paired["row_weighted_mean"],
        f"+{horizon}s Gate 5B overall delta",
    )

    movement = analysis5b["movement_event_all_rows"]
    decomposition = analysis5b["moving_rows_decomposition"]

    close(
        decomposition["movement_event_delta"]
        + decomposition["conditional_destination_delta"],
        analysis5a["moving"][
            "candidate_minus_control_log_loss"
        ],
        f"+{horizon}s Moving decomposition",
    )

    close(
        analysis5b["staying_rows"][
            "candidate_minus_control_binary_log_loss"
        ],
        analysis5a["staying"][
            "candidate_minus_control_log_loss"
        ],
        f"+{horizon}s Staying delta",
    )

    require(
        analysis5a["moving"]["rows"]
        + analysis5a["staying"]["rows"]
        == expected_rows,
        f"+{horizon}s Staying/Moving partition mismatch.",
    )

    summary[key] = {
        "rows": expected_rows,
        "control_log_loss": control["log_loss"],
        "candidate_log_loss": candidate["log_loss"],
        "candidate_minus_control_log_loss": (
            paired["row_weighted_mean"]
        ),
        "equal_match_mean_delta": (
            paired["equal_match_mean"]
        ),
        "match_bootstrap_95ci": (
            paired["equal_match_bootstrap_95ci"]
        ),
        "matches_with_lower_candidate_ll": (
            analysis5a["match_level"][
                "candidate_lower_log_loss_matches"
            ]
        ),
        "matches_with_higher_candidate_ll": (
            analysis5a["match_level"][
                "candidate_higher_log_loss_matches"
            ]
        ),
        "staying_rows": analysis5a["staying"]["rows"],
        "moving_rows": analysis5a["moving"]["rows"],
        "staying_multiclass_delta": (
            analysis5a["staying"][
                "candidate_minus_control_log_loss"
            ]
        ),
        "moving_multiclass_delta": (
            analysis5a["moving"][
                "candidate_minus_control_log_loss"
            ]
        ),
        "all_rows_movement_binary_ll_delta": (
            movement[
                "candidate_minus_control_binary_log_loss"
            ]
        ),
        "all_rows_movement_binary_brier_delta": (
            movement[
                "candidate_minus_control_binary_brier"
            ]
        ),
        "moving_rows_event_ll_delta": (
            decomposition["movement_event_delta"]
        ),
        "moving_rows_destination_ll_delta": (
            decomposition["conditional_destination_delta"]
        ),
    }

    print(
        f"+{horizon}s: "
        f"Control LL={control['log_loss']:.6f}, "
        f"Candidate LL={candidate['log_loss']:.6f}, "
        f"Delta={paired['row_weighted_mean']:+.6f} — PASS"
    )


# ------------------------------------------------------------
# Record what is finished and what remains unvalidated.
# ------------------------------------------------------------

closure = {
    "version": "V4_A_DEVELOPMENT_CLOSURE_V1",
    "status": "DEVELOPMENT_CLOSED_NOT_INDEPENDENTLY_CONFIRMED",
    "research_question": protocol["research_question"],

    "scope": {
        "map": "de_mirage",
        "development_matches": 53,
        "development_target_rows": 29065,
        "horizons_sec": [5, 10],
        "target_classes": 15,
        "control_dimensions": 24,
        "candidate_dimensions": 56,
        "cv": "5-fold grouped by match",
        "new_model_fits": 20,
    },

    "development_results": summary,

    "development_findings": [
        (
            "The 56D candidate had lower development OOF "
            "multiclass Log Loss than the 24D control "
            "at both evaluated horizons."
        ),
        (
            "The descriptive Log Loss difference was "
            "larger on actual Moving rows than Staying rows."
        ),
        (
            "Movement-event binary Log Loss and binary "
            "Brier decreased on the development observations."
        ),
        (
            "On actual Moving rows, the multiclass Log Loss "
            "difference decomposed into movement-event and "
            "conditional-destination components."
        ),
    ],

    "limitations": [
        (
            "All reported V4-A results are from historical "
            "development data, not fresh independent confirmation."
        ),
        (
            "Repeated observations within matches are dependent. "
            "The reported bootstrap resamples matches."
        ),
        (
            "Gate 5A and Gate 5B are post-experiment "
            "exploratory subgroup analyses."
        ),
        (
            "Moving means a different macro-zone at the target "
            "time; it does not measure arbitrary physical motion."
        ),
        (
            "Per-zone comparisons with very low support "
            "do not establish stable zone-specific effects."
        ),
        (
            "The analyses do not establish player-level causal "
            "effects or tactical decision-making ability."
        ),
    ],

    "research_boundaries": {
        "v2_confirmation_used": False,
        "v3_confirmation_used": False,
        "fresh_v4_confirmation_used": False,
        "v3_frozen_artifacts_modified": False,
        "v4_a_features_and_model_experiment_protocol_frozen": True,
        "full_development_models_fitted_for_confirmation": False,
        "full_development_model_artifacts_frozen": False,
        "v4_confirmation_corpus_acquired": False,
        "v4_confirmation_protocol_frozen": False,
        "v4_independent_confirmation_complete": False,
        "tactical_decision_layer_in_scope": False,
    },

    "post_closure_rules": [
        (
            "Do not silently modify the frozen V4-A V1 "
            "features, training configuration, or evaluation."
        ),
        (
            "Do not treat the previously exposed V2 or V3 "
            "confirmation corpora as fresh V4 confirmation."
        ),
        (
            "A future V4-A confirmation requires separate "
            "prospectively fixed eligibility, acquisition, "
            "fit, scoring, and exclusion protocols."
        ),
        (
            "Before examining new confirmation outcomes, "
            "freeze the final development-fitted model "
            "artifacts and their preprocessing provenance."
        ),
        (
            "Any exploratory feature or model changes "
            "belong to a separately versioned experiment."
        ),
    ],

    "source_artifacts": {
        name: {
            "path": str(FILES[name]),
            "sha256": digest,
        }
        for name, digest in hashes.items()
    },

    "next_action": (
        "Review the frozen V4-A V1 candidate for possible "
        "independent confirmation. If proceeding, separately "
        "freeze full-development fitting, eligible fresh "
        "confirmation acquisition, and scoring protocols "
        "before accessing confirmation outcomes."
    ),
}

with OUTPUT.open("x") as file:
    json.dump(
        closure,
        file,
        indent=2,
        ensure_ascii=False,
    )
    file.write("\n")

print("\n=== V4-A DEVELOPMENT CLOSURE SUMMARY ===")
print("Development matches: 53")
print("Development target rows: 29,065")
print("Feature dimensions: 24D Control / 56D Candidate")
print("Development OOF and Gate 5A/5B reconciliation: PASS")
print("Development status: CLOSED")
print("Independent V4 confirmation: NOT PERFORMED")
print("Closure:", OUTPUT)
print("\nV4_A_DEVELOPMENT_CLOSURE_RECORDED")
