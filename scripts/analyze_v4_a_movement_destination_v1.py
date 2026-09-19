"""V4 Gate 5B: movement probability and destination decomposition.

Exploratory analysis of existing V4-A development OOF predictions.

No model fitting, demo parsing, confirmation access,
or modification of frozen research artifacts.
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

EXPERIMENT = Path(
    "docs/v4_a_development_oof_results_v1.json"
)

ERROR_ANALYSIS = Path(
    "docs/v4_a_development_error_analysis_v1.json"
)

PROTOCOL = Path(
    "docs/v4_a_model_experiment_protocol_v1_frozen.json"
)

OUTPUT = Path(
    "docs/v4_a_movement_destination_analysis_v1.json"
)

KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


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


def mean(values):
    require(len(values) > 0, "Cannot summarize an empty group.")
    return float(np.mean(values))


def compare(actual, expected, label, atol=1e-7):
    require(
        np.isclose(
            actual,
            expected,
            atol=atol,
            rtol=0.0,
        ),
        f"{label}: expected {expected}, got {actual}",
    )


print("\n=== V4 GATE 5B: MOVEMENT AND DESTINATION ===")


# ============================================================
# 1. Validate existing experiment and analysis artifacts.
# ============================================================

for path in (
    OOF,
    EXPERIMENT,
    ERROR_ANALYSIS,
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

experiment = read_json(EXPERIMENT)
gate5a = read_json(ERROR_ANALYSIS)
protocol = read_json(PROTOCOL)

require(
    experiment["status"] == "DEVELOPMENT_OOF_COMPLETE",
    "V4-A development OOF experiment is incomplete.",
)

require(
    gate5a["status"] == "EXPLORATORY_DEVELOPMENT_ANALYSIS",
    "Gate 5A report is incomplete.",
)

require(
    protocol["status"]
    == "FROZEN_BEFORE_FIRST_V4_A_MODEL_FIT",
    "Unexpected frozen model protocol status.",
)

require(
    experiment["experiment_protocol_sha256"]
    == sha256(PROTOCOL),
    "Experiment protocol SHA256 mismatch.",
)

require(
    experiment["oof_predictions_sha256"]
    == sha256(OOF),
    "OOF artifact SHA256 mismatch.",
)

require(
    gate5a["source_oof_sha256"] == sha256(OOF),
    "Gate 5A used a different OOF artifact.",
)

require(
    gate5a["source_experiment_sha256"]
    == sha256(EXPERIMENT),
    "Gate 5A used a different experiment report.",
)

classes = protocol["scope"]["target_classes"]

require(
    len(classes) == 15
    and len(set(classes)) == 15,
    "Unexpected target class representation.",
)

zone_to_index = {
    zone: index
    for index, zone in enumerate(classes)
}

data = pl.read_parquet(OOF)

require(
    data.height == 29065,
    "Unexpected OOF target row count.",
)

require(
    data.select(KEY).unique().height == data.height,
    "Duplicate OOF target keys.",
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
    "Staying/Moving labels are not Boolean.",
)

print("Frozen OOF and Gate 5A provenance: PASS")


# ============================================================
# 2. Process +5s and +10s independently.
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
        frame.height == expected_rows
        and frame["demo_filename"].n_unique() == 53,
        f"Unexpected +{horizon}s OOF coverage.",
    )

    n = frame.height

    current_zones = frame["current_macro_zone"].to_list()

    require(
        all(zone in zone_to_index for zone in current_zones),
        f"Unexpected current macro-zone at +{horizon}s.",
    )

    current_indices = np.array(
        [zone_to_index[zone] for zone in current_zones],
        dtype=np.int64,
    )

    true_indices = (
        frame["target_class_index"]
        .to_numpy()
        .astype(np.int64)
    )

    require(
        ((true_indices >= 0) & (true_indices < 15)).all(),
        f"Invalid target indices at +{horizon}s.",
    )

    actual_moving = true_indices != current_indices

    require(
        np.array_equal(
            actual_moving,
            ~frame["stay_same_zone"].to_numpy(),
        ),
        f"Moving label inconsistency at +{horizon}s.",
    )

    moving_mask = actual_moving
    staying_mask = ~moving_mask

    require(
        moving_mask.any() and staying_mask.any(),
        f"Missing Staying or Moving observations at +{horizon}s.",
    )

    historical = gate5a["by_horizon"][str(horizon)]

    require(
        int(moving_mask.sum()) == historical["moving"]["rows"],
        f"Moving count differs from Gate 5A at +{horizon}s.",
    )

    require(
        int(staying_mask.sum()) == historical["staying"]["rows"],
        f"Staying count differs from Gate 5A at +{horizon}s.",
    )


    # --------------------------------------------------------
    # 3. Calculate both models' movement probabilities.
    # --------------------------------------------------------

    calculated = {}

    for model_name in ("control", "candidate"):

        probability_columns = [
            f"{model_name}_p_{zone}"
            for zone in classes
        ]

        require(
            all(column in frame.columns
                for column in probability_columns),
            f"Missing {model_name} class probabilities.",
        )

        probabilities = (
            frame.select(probability_columns)
            .to_numpy()
            .astype(np.float64)
        )

        require(
            probabilities.shape == (n, 15),
            f"Unexpected {model_name} probability shape.",
        )

        require(
            np.isfinite(probabilities).all()
            and (probabilities >= 0).all()
            and (probabilities <= 1).all(),
            f"Invalid {model_name} class probabilities.",
        )

        require(
            np.allclose(
                probabilities.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-6,
            ),
            f"{model_name} probabilities do not sum to one.",
        )

        p_current = probabilities[
            np.arange(n),
            current_indices,
        ]

        p_true = probabilities[
            np.arange(n),
            true_indices,
        ]

        require(
            (p_current > 0).all()
            and (p_true > 0).all(),
            f"{model_name} has zero current/target probability.",
        )

        p_moving = 1.0 - p_current

        require(
            ((p_moving >= 0) & (p_moving <= 1)).all(),
            f"Invalid {model_name} movement probability.",
        )

        require(
            (p_moving[moving_mask] > 0).all(),
            f"{model_name} assigns zero probability to "
            "an observed movement event.",
        )


        # Binary movement-event Log Loss.
        #
        # Moving row:  -log(P(Moving)).
        # Staying row: -log(P(Staying)).
        #
        # Calculate the staying term directly from p_current
        # to avoid unnecessary floating-point cancellation.

        binary_ll = np.empty(n, dtype=np.float64)

        binary_ll[moving_mask] = -np.log(
            p_moving[moving_mask]
        )

        binary_ll[staying_mask] = -np.log(
            p_current[staying_mask]
        )

        binary_brier = (
            p_moving - actual_moving.astype(np.float64)
        ) ** 2


        # Conditional destination probability on actual
        # Moving observations only:
        #
        # P(target zone | Moving, X)
        #     = P(target zone | X) / P(Moving | X).

        p_destination = (
            p_true[moving_mask]
            / p_moving[moving_mask]
        )

        require(
            np.isfinite(p_destination).all()
            and (p_destination > 0).all()
            and (p_destination <= 1.00001).all(),
            f"Invalid {model_name} conditional destination "
            f"probability at +{horizon}s.",
        )

        destination_ll = -np.log(p_destination)


        # Verify the probability decomposition against
        # the already audited original multiclass Log Loss.

        stored_ll = (
            frame[f"{model_name}_log_loss"]
            .to_numpy()
            .astype(np.float64)
        )

        require(
            np.allclose(
                binary_ll[moving_mask] + destination_ll,
                stored_ll[moving_mask],
                rtol=0.0,
                atol=1e-6,
            ),
            f"{model_name} movement/destination decomposition "
            f"does not reconstruct multiclass LL.",
        )

        require(
            np.allclose(
                binary_ll[staying_mask],
                stored_ll[staying_mask],
                rtol=0.0,
                atol=1e-6,
            ),
            f"{model_name} staying binary LL does not "
            f"reconstruct multiclass LL.",
        )

        calculated[model_name] = {
            "p_moving": p_moving,
            "binary_ll": binary_ll,
            "binary_brier": binary_brier,
            "destination_ll": destination_ll,
            "stored_multiclass_ll": stored_ll,
        }

        print(
            f"+{horizon}s {model_name}: "
            "probability decomposition PASS"
        )


    # --------------------------------------------------------
    # 4. Overall movement-event prediction.
    # --------------------------------------------------------

    control = calculated["control"]
    candidate = calculated["candidate"]

    movement_event = {
        "rows": n,
        "moving_rows": int(moving_mask.sum()),
        "staying_rows": int(staying_mask.sum()),
        "moving_prevalence": float(actual_moving.mean()),

        "control_binary_log_loss": mean(
            control["binary_ll"]
        ),

        "candidate_binary_log_loss": mean(
            candidate["binary_ll"]
        ),

        "candidate_minus_control_binary_log_loss": mean(
            candidate["binary_ll"] - control["binary_ll"]
        ),

        "control_binary_brier": mean(
            control["binary_brier"]
        ),

        "candidate_binary_brier": mean(
            candidate["binary_brier"]
        ),

        "candidate_minus_control_binary_brier": mean(
            candidate["binary_brier"]
            - control["binary_brier"]
        ),
    }


    # --------------------------------------------------------
    # 5. Decompose the actual Moving observations.
    # --------------------------------------------------------

    movement_event_delta_on_moving = mean(
        candidate["binary_ll"][moving_mask]
        - control["binary_ll"][moving_mask]
    )

    destination_delta_on_moving = mean(
        candidate["destination_ll"]
        - control["destination_ll"]
    )

    multiclass_delta_on_moving = mean(
        candidate["stored_multiclass_ll"][moving_mask]
        - control["stored_multiclass_ll"][moving_mask]
    )

    compare(
        movement_event_delta_on_moving
        + destination_delta_on_moving,
        multiclass_delta_on_moving,
        f"+{horizon}s Moving Log Loss decomposition",
        atol=1e-6,
    )

    compare(
        multiclass_delta_on_moving,
        historical["moving"][
            "candidate_minus_control_log_loss"
        ],
        f"+{horizon}s Gate 5A Moving Log Loss",
        atol=1e-6,
    )

    decomposition = {
        "moving_rows": int(moving_mask.sum()),

        "control_movement_event_ll": mean(
            control["binary_ll"][moving_mask]
        ),

        "candidate_movement_event_ll": mean(
            candidate["binary_ll"][moving_mask]
        ),

        "movement_event_delta": (
            movement_event_delta_on_moving
        ),

        "control_conditional_destination_ll": mean(
            control["destination_ll"]
        ),

        "candidate_conditional_destination_ll": mean(
            candidate["destination_ll"]
        ),

        "conditional_destination_delta": (
            destination_delta_on_moving
        ),

        "original_moving_multiclass_delta": (
            multiclass_delta_on_moving
        ),

        "decomposition_reconstruction": "PASS",
    }


    # --------------------------------------------------------
    # 6. Staying rows: measure false-alarm cost.
    # --------------------------------------------------------

    staying_event_delta = mean(
        candidate["binary_ll"][staying_mask]
        - control["binary_ll"][staying_mask]
    )

    compare(
        staying_event_delta,
        historical["staying"][
            "candidate_minus_control_log_loss"
        ],
        f"+{horizon}s Gate 5A Staying Log Loss",
        atol=1e-6,
    )

    staying = {
        "rows": int(staying_mask.sum()),

        "control_binary_log_loss": mean(
            control["binary_ll"][staying_mask]
        ),

        "candidate_binary_log_loss": mean(
            candidate["binary_ll"][staying_mask]
        ),

        "candidate_minus_control_binary_log_loss": (
            staying_event_delta
        ),
    }


    # --------------------------------------------------------
    # 7. Verify the original full multiclass difference.
    # --------------------------------------------------------

    reconstructed_overall_delta = (
        actual_moving.mean()
        * (
            movement_event_delta_on_moving
            + destination_delta_on_moving
        )
        + staying_mask.mean() * staying_event_delta
    )

    original_overall_delta = mean(
        candidate["stored_multiclass_ll"]
        - control["stored_multiclass_ll"]
    )

    compare(
        reconstructed_overall_delta,
        original_overall_delta,
        f"+{horizon}s overall LL reconstruction",
        atol=1e-6,
    )

    compare(
        original_overall_delta,
        experiment["results_by_horizon"][str(horizon)][
            "paired_comparison"
        ]["row_weighted_mean"],
        f"+{horizon}s original experiment delta",
        atol=1e-6,
    )


    # --------------------------------------------------------
    # 8. Save horizon-specific results.
    # --------------------------------------------------------

    by_horizon[str(horizon)] = {
        "movement_event_all_rows": movement_event,

        "moving_rows_decomposition": decomposition,

        "staying_rows": staying,

        "original_overall_multiclass_delta": (
            original_overall_delta
        ),

        "reconstructed_overall_multiclass_delta": (
            reconstructed_overall_delta
        ),
    }


    print(f"\n=== +{horizon}s GATE 5B RESULTS ===")

    print(
        "Moving prevalence:",
        f"{movement_event['moving_prevalence']:.2%}",
    )

    print(
        "Overall movement-event binary LL:"
        f"\n  Control:   "
        f"{movement_event['control_binary_log_loss']:.6f}"
        f"\n  Candidate: "
        f"{movement_event['candidate_binary_log_loss']:.6f}"
        f"\n  Delta:     "
        f"{movement_event['candidate_minus_control_binary_log_loss']:+.6f}"
    )

    print(
        "Overall movement-event binary Brier:"
        f"\n  Control:   "
        f"{movement_event['control_binary_brier']:.6f}"
        f"\n  Candidate: "
        f"{movement_event['candidate_binary_brier']:.6f}"
        f"\n  Delta:     "
        f"{movement_event['candidate_minus_control_binary_brier']:+.6f}"
    )

    print(
        "Actual Moving rows — LL decomposition:"
        f"\n  Movement-event delta: "
        f"{movement_event_delta_on_moving:+.6f}"
        f"\n  Conditional-destination delta: "
        f"{destination_delta_on_moving:+.6f}"
        f"\n  Sum: "
        f"{multiclass_delta_on_moving:+.6f}"
    )

    print(
        "Actual Staying rows — binary LL delta:",
        f"{staying_event_delta:+.6f}",
    )

    print(
        "Overall multiclass LL reconstruction: PASS"
    )


# ============================================================
# 9. Record a new exploratory analysis.
# ============================================================

report = {
    "version": "V4_A_MOVEMENT_DESTINATION_ANALYSIS_V1",

    "status": "EXPLORATORY_DEVELOPMENT_ANALYSIS",

    "research_question": (
        "Does the Team Context-associated Log Loss change "
        "arise from movement-event probability estimation, "
        "conditional destination probabilities, or both?"
    ),

    "definitions": {
        "moving": (
            "Future target macro-zone differs from the "
            "current macro-zone."
        ),

        "movement_probability": (
            "1 - P(future zone = current zone | X_t)"
        ),

        "conditional_destination_probability": (
            "P(future target zone | X_t) / "
            "P(moving | X_t), evaluated on actual Moving rows."
        ),

        "moving_log_loss_identity": (
            "Multiclass LL = movement-event binary LL "
            "+ conditional-destination LL."
        ),
    },

    "source_experiment_sha256": sha256(EXPERIMENT),

    "source_gate5a_sha256": sha256(ERROR_ANALYSIS),

    "source_oof_sha256": sha256(OOF),

    "source_model_protocol_sha256": sha256(PROTOCOL),

    "analysis_scope": {
        "development_matches": 53,
        "target_rows": 29065,
        "horizons_sec": [5, 10],
        "new_models_trained": False,
        "confirmation_data_used": False,
    },

    "by_horizon": by_horizon,

    "limitations": [
        "This is a post-experiment development analysis.",

        "Moving and Staying are defined using future target "
        "labels only for evaluation.",

        "Conditional-destination scores are calculated only "
        "on observations where the bomb actually changes "
        "macro-zone.",

        "The movement-event probability measures a "
        "different future macro-zone, not arbitrary physical "
        "movement or a tactical decision.",

        "Observed probability changes do not independently "
        "establish causal effects of any individual player "
        "or tactical behavior.",

        "Development results do not replace fresh "
        "independent V4 confirmation.",
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


print("\n=== V4 GATE 5B FINAL SUMMARY ===")

print("Development matches: 53")
print("Development target rows: 29,065")
print("Movement-event binary evaluation: COMPLETE")
print("Conditional destination decomposition: COMPLETE")
print("Staying false-alarm analysis: COMPLETE")
print("Multiclass Log Loss reconstruction: PASS")
print("Report:", OUTPUT)

print("\nGATE_5B_MOVEMENT_DESTINATION_ANALYSIS_COMPLETE")
print("No models were trained.")
print("No confirmation data was accessed.")
print("No frozen research artifacts were modified.")
