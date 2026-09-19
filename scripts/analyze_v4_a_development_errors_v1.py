"""V4 Gate 5A: exploratory development error analysis.

Analyze existing, audited OOF predictions only.

No model fitting, demo parsing, confirmation access,
or changes to frozen V3/V4 artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


OOF = Path(
    "data/interim/v4_a_control_vs_team_oof_v1.parquet"
)

RESULTS = Path(
    "docs/v4_a_development_oof_results_v1.json"
)

PROTOCOL = Path(
    "docs/v4_a_model_experiment_protocol_v1_frozen.json"
)

OUTPUT = Path(
    "docs/v4_a_development_error_analysis_v1.json"
)

HORIZONS = (5, 10)


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


def summarize(frame):
    """Descriptive paired metrics on exactly the same rows."""

    require(
        frame.height > 0,
        "Cannot summarize an empty subgroup.",
    )

    control_correct = (
        frame["control_predicted_class_index"]
        == frame["target_class_index"]
    )

    candidate_correct = (
        frame["candidate_predicted_class_index"]
        == frame["target_class_index"]
    )

    control_ll = float(
        frame["control_log_loss"].mean()
    )

    candidate_ll = float(
        frame["candidate_log_loss"].mean()
    )

    return {
        "rows": frame.height,

        "matches": frame["demo_filename"].n_unique(),

        "control_log_loss": control_ll,

        "candidate_log_loss": candidate_ll,

        "candidate_minus_control_log_loss": (
            candidate_ll - control_ll
        ),

        "control_accuracy": float(
            control_correct.mean()
        ),

        "candidate_accuracy": float(
            candidate_correct.mean()
        ),

        "control_brier": float(
            frame["control_multiclass_brier"].mean()
        ),

        "candidate_brier": float(
            frame["candidate_multiclass_brier"].mean()
        ),
    }


print("\n=== V4 GATE 5A: DEVELOPMENT ERROR ANALYSIS ===")


# ============================================================
# 1. Verify previously audited experiment inputs
# ============================================================

for path in (
    OOF,
    RESULTS,
    PROTOCOL,
):
    require(
        path.is_file(),
        f"Missing required input: {path}",
    )

require(
    not OUTPUT.exists(),
    f"Output already exists; refusing to overwrite: {OUTPUT}",
)

results = load_json(RESULTS)
protocol = load_json(PROTOCOL)

require(
    results["status"] == "DEVELOPMENT_OOF_COMPLETE",
    "V4-A OOF experiment is not complete.",
)

require(
    protocol["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "Unexpected model experiment protocol.",
)

require(
    results["experiment_protocol_sha256"]
    == sha256(PROTOCOL),
    "Experiment protocol SHA256 mismatch.",
)

require(
    results["oof_predictions_sha256"] == sha256(OOF),
    "OOF artifact SHA256 mismatch.",
)

classes = protocol["scope"]["target_classes"]

require(
    len(classes) == 15,
    "Unexpected target class representation.",
)

data = pl.read_parquet(OOF)

require(
    data.height == 29065,
    "Unexpected OOF row count.",
)

KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]

require(
    data.select(KEY).unique().height == data.height,
    "Duplicate OOF observation-horizon keys.",
)

require(
    data["demo_filename"].n_unique() == 53,
    "Unexpected development match coverage.",
)

require(
    data["stay_same_zone"].null_count() == 0,
    "Missing Staying/Moving labels.",
)

require(
    data["stay_same_zone"].dtype == pl.Boolean,
    "Staying/Moving indicator is not Boolean.",
)

require(
    (
        data["current_macro_zone"]
        == data["target_macro_zone"]
    ).to_list()
    == data["stay_same_zone"].to_list(),
    "Staying/Moving label differs from frozen zone labels.",
)

data = data.with_columns(
    (
        pl.col("candidate_log_loss")
        - pl.col("control_log_loss")
    ).alias("_delta_ll")
)

print("Frozen experiment and OOF identity: PASS")
print("Staying/Moving label consistency: PASS")


# ============================================================
# 2. Analyze each horizon independently
# ============================================================

by_horizon = {}

for horizon, expected_rows in (
    (5, 14869),
    (10, 14196),
):

    frame = data.filter(
        pl.col("horizon_sec") == horizon
    )

    require(
        frame.height == expected_rows,
        f"Unexpected +{horizon}s observation count.",
    )

    overall = summarize(frame)

    recorded = results["results_by_horizon"][str(horizon)]

    require(
        np.isclose(
            overall["control_log_loss"],
            recorded["control"]["log_loss"],
            atol=1e-8,
            rtol=0,
        ),
        f"+{horizon}s Control LL disagrees with OOF report.",
    )

    require(
        np.isclose(
            overall["candidate_log_loss"],
            recorded["candidate"]["log_loss"],
            atol=1e-8,
            rtol=0,
        ),
        f"+{horizon}s Candidate LL disagrees with OOF report.",
    )


    # --------------------------------------------------------
    # Staying versus Moving
    # --------------------------------------------------------

    staying = frame.filter(
        pl.col("stay_same_zone")
    )

    moving = frame.filter(
        ~pl.col("stay_same_zone")
    )

    require(
        staying.height + moving.height == frame.height,
        "Staying/Moving partition is incomplete.",
    )

    staying_result = summarize(staying)
    moving_result = summarize(moving)


    # --------------------------------------------------------
    # Per-target-zone descriptive results
    # --------------------------------------------------------

    per_target_zone = []

    for class_index, zone in enumerate(classes):

        part = frame.filter(
            pl.col("target_class_index") == class_index
        )

        require(
            part.height > 0,
            f"Missing target class: +{horizon}s {zone}",
        )

        metrics = summarize(part)

        metrics["zone"] = zone
        metrics["target_class_index"] = class_index

        metrics["interpretation_warning"] = (
            "Low support; descriptive only."
            if part.height < 30
            else None
        )

        per_target_zone.append(metrics)

    require(
        sum(row["rows"] for row in per_target_zone)
        == frame.height,
        "Target-zone partition is incomplete.",
    )


    # --------------------------------------------------------
    # Per-match paired log-loss differences
    # --------------------------------------------------------

    per_match = (
        frame
        .group_by("demo_filename")
        .agg([
            pl.len().alias("rows"),

            pl.col("control_log_loss")
            .mean()
            .alias("control_ll"),

            pl.col("candidate_log_loss")
            .mean()
            .alias("candidate_ll"),

            pl.col("_delta_ll")
            .mean()
            .alias("delta_ll"),
        ])
        .sort("demo_filename")
    )

    require(
        per_match.height == 53,
        "Expected 53 match-level results.",
    )

    improved_matches = per_match.filter(
        pl.col("delta_ll") < 0
    ).height

    worsened_matches = per_match.filter(
        pl.col("delta_ll") > 0
    ).height

    tied_matches = per_match.filter(
        pl.col("delta_ll") == 0
    ).height

    require(
        improved_matches + worsened_matches + tied_matches
        == 53,
        "Match-level direction count is inconsistent.",
    )

    equal_match_delta = float(
        per_match["delta_ll"].mean()
    )

    require(
        np.isclose(
            equal_match_delta,
            recorded["paired_comparison"]["equal_match_mean"],
            atol=1e-8,
            rtol=0,
        ),
        f"+{horizon}s equal-match delta mismatch.",
    )


    # --------------------------------------------------------
    # Save the exploratory analysis for this horizon
    # --------------------------------------------------------

    by_horizon[str(horizon)] = {
        "overall": overall,

        "staying": staying_result,

        "moving": moving_result,

        "per_target_zone": per_target_zone,

        "match_level": {
            "matches": 53,

            "candidate_lower_log_loss_matches": improved_matches,

            "candidate_higher_log_loss_matches": worsened_matches,

            "equal_log_loss_matches": tied_matches,

            "equal_match_mean_delta": equal_match_delta,

            "per_match": per_match.to_dicts(),
        },
    }


    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print(f"\n=== +{horizon}s ERROR ANALYSIS ===")

    print(
        f"Overall: {frame.height:,} rows"
        f" | Delta LL:"
        f" {overall['candidate_minus_control_log_loss']:+.6f}"
    )

    for label, item in (
        ("Staying", staying_result),
        ("Moving", moving_result),
    ):

        print(
            f"{label}:"
            f" rows={item['rows']:,}"
            f" | Control LL={item['control_log_loss']:.6f}"
            f" | Candidate LL={item['candidate_log_loss']:.6f}"
            f" | Delta={item['candidate_minus_control_log_loss']:+.6f}"
        )

    print(
        "Matches with lower Candidate LL:",
        improved_matches,
        "/ 53",
    )

    print(
        "Matches with higher Candidate LL:",
        worsened_matches,
        "/ 53",
    )

    print("\nPer-target-zone Log Loss difference:")

    for row in per_target_zone:
        print(
            f"  {row['zone']:<18}"
            f" n={row['rows']:>5}"
            f" | Delta LL="
            f"{row['candidate_minus_control_log_loss']:+.6f}"
        )


# ============================================================
# 3. Write a new, versioned exploratory analysis report
# ============================================================

report = {
    "version": "V4_A_DEVELOPMENT_ERROR_ANALYSIS_V1",

    "status": "EXPLORATORY_DEVELOPMENT_ANALYSIS",

    "research_questions": [
        "Is the log-loss change concentrated in Staying "
        "or Moving observations?",

        "How does the paired log-loss difference vary "
        "across the 15 target macro-zones?",

        "How are match-level paired differences distributed "
        "across the 53 development matches?",
    ],

    "source_experiment": str(RESULTS),

    "source_experiment_sha256": sha256(RESULTS),

    "source_oof_sha256": sha256(OOF),

    "model_experiment_protocol_sha256": sha256(PROTOCOL),

    "analysis_scope": {
        "development_matches": 53,

        "target_rows": 29065,

        "horizons_sec": [5, 10],

        "new_models_trained": False,

        "new_model_predictions_generated": False,

        "confirmation_data_used": False,
    },

    "by_horizon": by_horizon,

    "limitations": [
        "These are post-experiment exploratory subgroup analyses.",

        "Staying and Moving are determined using the future "
        "target label. They are evaluation groups, not "
        "features available at prediction time.",

        "Per-zone subgroups may have low support.",

        "Multiple observations from one match are dependent.",

        "The two prediction horizons are not independent "
        "experimental samples.",

        "Descriptive subgroup differences do not establish "
        "why the model predictions changed.",

        "This development analysis does not replace "
        "fresh independent V4 confirmation.",
    ],
}

with OUTPUT.open("x") as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )

    file.write("\n")


print("\n=== V4 GATE 5A FINAL SUMMARY ===")

print("Development matches: 53")
print("Development target rows: 29,065")
print("Horizon analyses: +5s and +10s")
print("Staying/Moving analysis: COMPLETE")
print("Per-target-zone analysis: COMPLETE")
print("Match-level paired analysis: COMPLETE")
print("Report:", OUTPUT)

print("\nGATE_5A_DEVELOPMENT_ERROR_ANALYSIS_COMPLETE")

print("No models were trained.")
print("No confirmation data was accessed.")
print("No frozen research artifacts were modified.")
