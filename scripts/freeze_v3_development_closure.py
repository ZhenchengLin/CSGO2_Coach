from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


B0_RESULTS = Path(
    "docs/v3_b0_persistence_results.json"
)

B1_RESULTS = Path(
    "docs/v3_b1_constant_motion_results.json"
)

B2_RESULTS = Path(
    "docs/v3_b2_markov_results.json"
)

B3_RESULTS = Path(
    "docs/v3_b3_tabular_map_aware_results.json"
)

TARGETS = Path(
    "data/processed/v3_dev_targets_v1.parquet"
)

B0_OOF = Path(
    "data/interim/v3_b0_persistence_oof_predictions.parquet"
)

B1_OOF = Path(
    "data/interim/v3_b1_constant_motion_oof_predictions.parquet"
)

B2_OOF = Path(
    "data/interim/v3_b2_markov_oof_predictions.parquet"
)

B3_OOF = Path(
    "data/interim/v3_b3_tabular_map_aware_oof_predictions.parquet"
)

OUTPUT = Path(
    "docs/v3_development_closure_v1.json"
)


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text()
    )


for p in [
    B0_RESULTS,
    B1_RESULTS,
    B2_RESULTS,
    B3_RESULTS,
    TARGETS,
    B0_OOF,
    B1_OOF,
    B2_OOF,
    B3_OOF,
]:
    require(
        p.exists(),
        f"Missing: {p}",
    )


b0 = read_json(B0_RESULTS)
b1 = read_json(B1_RESULTS)
b2 = read_json(B2_RESULTS)
b3 = read_json(B3_RESULTS)


targets = pl.read_parquet(
    TARGETS
)

b0_pred = pl.read_parquet(
    B0_OOF
)

b1_pred = pl.read_parquet(
    B1_OOF
)

b2_pred = pl.read_parquet(
    B2_OOF
)

b3_pred = pl.read_parquet(
    B3_OOF
)


KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


paired = (
    targets
    .select([
        *KEY,
        "dev_source_role",
        "target_class_index",
        "target_macro_zone",
    ])
    .join(
        b0_pred.select([
            *KEY,
            pl.col("log_loss").alias("b0_ll"),
        ]),
        on=KEY,
        how="left",
    )
    .join(
        b1_pred.select([
            *KEY,
            pl.col("log_loss").alias("b1_ll"),
        ]),
        on=KEY,
        how="left",
    )
    .join(
        b2_pred.select([
            *KEY,
            pl.col("log_loss").alias("b2_ll"),
        ]),
        on=KEY,
        how="left",
    )
    .join(
        b3_pred.select([
            *KEY,
            pl.col("log_loss").alias("b3_ll"),
            pl.col("predicted_class_index").alias("b3_pred"),
        ]),
        on=KEY,
        how="left",
    )
)


require(
    paired["b0_ll"].null_count() == 0,
    "Missing B0 pairing.",
)

require(
    paired["b1_ll"].null_count() == 0,
    "Missing B1 pairing.",
)

require(
    paired["b2_ll"].null_count() == 0,
    "Missing B2 pairing.",
)

require(
    paired["b3_ll"].null_count() == 0,
    "Missing B3 pairing.",
)


horizon_summary = {}


for horizon in [5, 10]:

    p = paired.filter(
        pl.col("horizon_sec") == horizon
    )

    per_match = (
        p
        .group_by([
            "demo_filename",
            "dev_source_role",
        ])
        .agg(
            (
                pl.col("b3_ll")
                - pl.col("b2_ll")
            )
            .mean()
            .alias("delta")
        )
    )

    delta = (
        per_match["delta"]
        .to_numpy()
    )

    fresh = per_match.filter(
        pl.col("dev_source_role")
        == "V3_FRESH"
    )

    fresh_delta = (
        fresh["delta"]
        .to_numpy()
    )

    h = str(horizon)

    horizon_summary[h] = {
        "b0_log_loss":
            float(
                p["b0_ll"].mean()
            ),

        "b1_log_loss":
            float(
                p["b1_ll"].mean()
            ),

        "b2_log_loss":
            float(
                p["b2_ll"].mean()
            ),

        "b3_log_loss":
            float(
                p["b3_ll"].mean()
            ),

        "b3_minus_b2_row_delta":
            b3["results_by_horizon"][h][
                "delta_ll_b3_minus_b2"
            ],

        "b3_minus_b2_equal_match_delta":
            b3["results_by_horizon"][h][
                "equal_match_delta_ll_b3_minus_b2"
            ],

        "b3_minus_b2_match_bootstrap_95ci":
            b3["results_by_horizon"][h][
                "b3_minus_b2_match_bootstrap_95ci"
            ],

        "b3_lower_ll_matches":
            int((delta < 0).sum()),

        "total_matches":
            int(len(delta)),

        "v3_fresh_lower_ll_matches":
            int((fresh_delta < 0).sum()),

        "v3_fresh_total_matches":
            int(len(fresh_delta)),

        "v3_fresh_mean_delta":
            float(fresh_delta.mean()),

        "b3_accuracy":
            b3["results_by_horizon"][h]["accuracy"],

        "b3_macro_f1":
            b3["results_by_horizon"][h]["macro_f1"],

        "b3_top2_accuracy":
            b3["results_by_horizon"][h]["top2_accuracy"],
    }


require(
    horizon_summary["5"]["b3_lower_ll_matches"] == 53,
    "B3 does not beat B2 on all +5s development matches.",
)

require(
    horizon_summary["10"]["b3_lower_ll_matches"] == 53,
    "B3 does not beat B2 on all +10s development matches.",
)

require(
    horizon_summary["5"]["v3_fresh_lower_ll_matches"] == 20,
    "Expected 20/20 fresh +5s matches.",
)

require(
    horizon_summary["10"]["v3_fresh_lower_ll_matches"] == 20,
    "Expected 20/20 fresh +10s matches.",
)


record = {
    "version":
        "V3_DEVELOPMENT_CLOSURE_V1",

    "status":
        "DEVELOPMENT_CLOSED",

    "development_corpus": {
        "matches":
            53,

        "roles": {
            "V0_DEVELOPMENT":
                20,

            "V2_RESERVE":
                13,

            "V3_FRESH":
                20,
        },

        "important_note":
            (
                "All 53 matches are development data. "
                "V3_FRESH is not confirmation evidence."
            ),
    },

    "selected_confirmation_candidate": {
        "name":
            "B3_TABULAR_MAP_AWARE_V1",

        "selection_metric":
            "development OOF multiclass log loss",

        "feature_protocol":
            "docs/v3_b3_feature_protocol_v1.json",

        "model_protocol":
            "docs/v3_b3_model_protocol_v1.json",

        "feature_or_hyperparameter_changes_after_selection":
            False,
    },

    "development_results":
        horizon_summary,

    "remaining_development_limitations": [
        (
            "Macro F1 remains below B0 despite strong "
            "log-loss, accuracy, and top-2 improvements."
        ),
        (
            "Rare classes such as MID_WINDOW, MARKET, "
            "and CT_SPAWN have extremely low support."
        ),
        (
            "A_RAMP and CONNECTOR remain meaningful "
            "route-specific hard-classification weaknesses."
        ),
        (
            "Development evidence must not be described "
            "as independent confirmation."
        ),
    ],

    "post_closure_rules": [
        (
            "Do not silently modify B3 V1 features, "
            "preprocessing, or hyperparameters."
        ),
        (
            "Do not create additional B3 variants before "
            "the planned untouched V3 confirmation evaluation."
        ),
        (
            "Any later exploratory model change must be "
            "declared as a new experiment/version and must "
            "not retroactively replace B3 V1."
        ),
        (
            "V2 D_CONFIRM remains permanently forbidden "
            "for V3."
        ),
        (
            "Future D_V3_CONFIRM must be later than all "
            "development matches and untouched until "
            "confirmation protocols and final fitted "
            "development artifacts are frozen."
        ),
    ],

    "artifact_sha256": {
        "b0_results":
            sha256(B0_RESULTS),

        "b1_results":
            sha256(B1_RESULTS),

        "b2_results":
            sha256(B2_RESULTS),

        "b3_results":
            sha256(B3_RESULTS),

        "targets":
            sha256(TARGETS),

        "b0_oof":
            sha256(B0_OOF),

        "b1_oof":
            sha256(B1_OOF),

        "b2_oof":
            sha256(B2_OOF),

        "b3_oof":
            sha256(B3_OOF),
    },

    "v2_d_confirm_used":
        False,

    "v3_confirmation_used":
        False,

    "next_action":
        (
            "Freeze full-development fitted B0-B3 artifacts "
            "and the untouched D_V3_CONFIRM protocol before "
            "accessing confirmation outcomes."
        ),
}


if OUTPUT.exists():

    existing = read_json(
        OUTPUT
    )

    require(
        existing == record,
        "Existing development closure differs.",
    )

else:

    OUTPUT.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n"
    )


print("V3 DEVELOPMENT CLOSURE")
print()

for h in ["5", "10"]:

    x = horizon_summary[h]

    print(f"+{h}s")
    print(
        "  B0 LL:",
        f"{x['b0_log_loss']:.6f}",
    )
    print(
        "  B1 LL:",
        f"{x['b1_log_loss']:.6f}",
    )
    print(
        "  B2 LL:",
        f"{x['b2_log_loss']:.6f}",
    )
    print(
        "  B3 LL:",
        f"{x['b3_log_loss']:.6f}",
    )
    print(
        "  B3 < B2 matches:",
        f"{x['b3_lower_ll_matches']}/53",
    )
    print(
        "  fresh wins:",
        f"{x['v3_fresh_lower_ll_matches']}/20",
    )
    print()

print("selected candidate: B3_TABULAR_MAP_AWARE_V1")
print("development status: CLOSED")
print("confirmation used: NO")
print()
print("V3_DEVELOPMENT_CLOSED")
