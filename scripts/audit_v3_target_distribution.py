from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl


OBS_PATH = Path(
    "data/interim/v3_target_integrity_observations.csv"
)

MAPPING_PATH = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)

ZONE_SUPPORT_OUTPUT = Path(
    "data/interim/v3_target_zone_support.csv"
)

CURRENT_SUPPORT_OUTPUT = Path(
    "data/interim/v3_current_zone_support.csv"
)

TRANSITION_OUTPUT = Path(
    "data/interim/v3_target_transition_counts.csv"
)

SUMMARY_OUTPUT = Path(
    "docs/v3_target_distribution_audit.json"
)


HORIZONS = [
    5,
    10,
]


KEY_COLUMNS = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Load frozen semantic representation
# ============================================================

require(
    OBS_PATH.exists(),
    f"Missing observations: {OBS_PATH}",
)

require(
    MAPPING_PATH.exists(),
    f"Missing frozen mapping: {MAPPING_PATH}",
)


mapping = json.loads(
    MAPPING_PATH.read_text()
)

require(
    mapping["status"] == "FROZEN",
    "Macro-zone mapping is not frozen.",
)

macro_zones = sorted(
    mapping["zones"].keys()
)

require(
    len(macro_zones) == 15,
    f"Expected 15 zones, found {len(macro_zones)}",
)


# ============================================================
# Load Gate 2D observations
# ============================================================

obs = pl.read_csv(
    OBS_PATH,
    infer_schema_length=None,
)


required_columns = {
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
    "current_source",
    "current_macro_zone",
    "horizon_sec",
    "target_tick",
    "target_source",
    "target_macro_zone",
    "stay_same_zone",
    "valid",
    "exclusion_reason",
}


missing_columns = (
    required_columns
    - set(obs.columns)
)

require(
    not missing_columns,
    (
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    ),
)


require(
    set(
        obs[
            "horizon_sec"
        ].unique().to_list()
    )
    == set(HORIZONS),
    "Unexpected horizons in Gate 2D artifact.",
)


# ============================================================
# Unique candidate-key invariant
# ============================================================

duplicate_keys = (
    obs
    .group_by(
        KEY_COLUMNS
        + [
            "horizon_sec",
        ]
    )
    .len()
    .filter(
        pl.col("len") != 1
    )
)

require(
    duplicate_keys.height == 0,
    "Duplicate candidate horizon keys detected.",
)


# ============================================================
# Candidate-current-point structure
# ============================================================

def key_from_row(row):
    return (
        str(
            row["demo_filename"]
        ),
        int(
            row["round_num"]
        ),
        int(
            row["current_nominal_tick"]
        ),
    )


rows = obs.to_dicts()


candidate_keys = {}

valid_keys = {}

for horizon in HORIZONS:

    candidate_keys[
        horizon
    ] = {
        key_from_row(row)
        for row in rows
        if int(
            row[
                "horizon_sec"
            ]
        ) == horizon
    }

    valid_keys[
        horizon
    ] = {
        key_from_row(row)
        for row in rows
        if (
            int(
                row[
                    "horizon_sec"
                ]
            )
            == horizon
            and row[
                "valid"
            ]
            is True
        )
    }


require(
    candidate_keys[5]
    == candidate_keys[10],
    (
        "The +5s and +10s candidate current-point "
        "sets are not identical."
    ),
)


require(
    valid_keys[10]
    <= valid_keys[5],
    (
        "Found a current point valid at +10s "
        "but invalid at +5s."
    ),
)


n_current_points = len(
    candidate_keys[5]
)

n_both_valid = len(
    valid_keys[10]
)

n_only_5_valid = len(
    valid_keys[5]
    - valid_keys[10]
)

n_neither_valid = len(
    candidate_keys[5]
    - valid_keys[5]
)


# ============================================================
# Exclusion integrity
# ============================================================

invalid = obs.filter(
    pl.col("valid")
    == False
)

exclusion_reasons = set(
    invalid[
        "exclusion_reason"
    ]
    .drop_nulls()
    .unique()
    .to_list()
)


require(
    exclusion_reasons
    <= {
        "TARGET_AFTER_ROUND_END",
    },
    (
        "Gate 2E found a non-boundary integrity "
        "failure: "
        f"{sorted(exclusion_reasons)}"
    ),
)


# ============================================================
# Valid-row invariants
# ============================================================

valid = obs.filter(
    pl.col("valid")
    == True
)


for column in [
    "current_tick",
    "current_source",
    "current_macro_zone",
    "target_tick",
    "target_source",
    "target_macro_zone",
    "stay_same_zone",
]:

    null_count = (
        valid[
            column
        ].null_count()
    )

    require(
        null_count == 0,
        (
            f"Valid rows contain nulls in "
            f"{column}: {null_count}"
        ),
    )


observed_current_zones = set(
    valid[
        "current_macro_zone"
    ]
    .unique()
    .to_list()
)

observed_target_zones = set(
    valid[
        "target_macro_zone"
    ]
    .unique()
    .to_list()
)


require(
    observed_current_zones
    <= set(macro_zones),
    "Unknown current macro-zone detected.",
)

require(
    observed_target_zones
    <= set(macro_zones),
    "Unknown target macro-zone detected.",
)


# ============================================================
# Target support
# ============================================================

zone_support = (
    valid
    .group_by([
        "horizon_sec",
        "target_macro_zone",
    ])
    .len()
    .rename({
        "len":
            "n",
    })
    .with_columns(
        (
            pl.col("n")
            /
            pl.col("n")
            .sum()
            .over(
                "horizon_sec"
            )
        ).alias(
            "share"
        )
    )
    .with_columns([
        (
            pl.col("n")
            < 20
        ).alias(
            "below_20"
        ),

        (
            pl.col("n")
            < 50
        ).alias(
            "below_50"
        ),

        (
            pl.col("n")
            < 100
        ).alias(
            "below_100"
        ),
    ])
    .sort([
        "horizon_sec",
        "n",
    ], descending=[
        False,
        True,
    ])
)


ZONE_SUPPORT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

zone_support.write_csv(
    ZONE_SUPPORT_OUTPUT
)


# ============================================================
# Current-zone support
# ============================================================

current_support = (
    valid
    .group_by([
        "horizon_sec",
        "current_macro_zone",
    ])
    .len()
    .rename({
        "len":
            "n",
    })
    .with_columns(
        (
            pl.col("n")
            /
            pl.col("n")
            .sum()
            .over(
                "horizon_sec"
            )
        ).alias(
            "share"
        )
    )
    .sort([
        "horizon_sec",
        "n",
    ], descending=[
        False,
        True,
    ])
)


current_support.write_csv(
    CURRENT_SUPPORT_OUTPUT
)


# ============================================================
# Current -> future transition distribution
# ============================================================

transitions = (
    valid
    .group_by([
        "horizon_sec",
        "current_macro_zone",
        "target_macro_zone",
    ])
    .len()
    .rename({
        "len":
            "n",
    })
    .with_columns(
        (
            pl.col("n")
            /
            pl.col("n")
            .sum()
            .over([
                "horizon_sec",
                "current_macro_zone",
            ])
        ).alias(
            "conditional_share"
        )
    )
    .with_columns(
        (
            pl.col(
                "current_macro_zone"
            )
            ==
            pl.col(
                "target_macro_zone"
            )
        ).alias(
            "is_persistence"
        )
    )
    .sort([
        "horizon_sec",
        "n",
    ], descending=[
        False,
        True,
    ])
)


transitions.write_csv(
    TRANSITION_OUTPUT
)


# ============================================================
# Source distributions
# ============================================================

target_sources = (
    valid
    .group_by([
        "horizon_sec",
        "target_source",
    ])
    .len()
    .rename({
        "len":
            "n",
    })
    .with_columns(
        (
            pl.col("n")
            /
            pl.col("n")
            .sum()
            .over(
                "horizon_sec"
            )
        ).alias(
            "share"
        )
    )
    .sort([
        "horizon_sec",
        "n",
    ], descending=[
        False,
        True,
    ])
)


# ============================================================
# Stay / move
# ============================================================

stay_move = (
    valid
    .group_by([
        "horizon_sec",
        "stay_same_zone",
    ])
    .len()
    .rename({
        "len":
            "n",
    })
    .with_columns(
        (
            pl.col("n")
            /
            pl.col("n")
            .sum()
            .over(
                "horizon_sec"
            )
        ).alias(
            "share"
        )
    )
    .sort([
        "horizon_sec",
        "stay_same_zone",
    ])
)


# ============================================================
# Target entropy
#
# Descriptive only.
# No pass/fail threshold is defined from entropy.
# ============================================================

entropy_summary = {}


for horizon in HORIZONS:

    horizon_counts = (
        zone_support
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )[
            "n"
        ]
        .to_list()
    )

    total = sum(
        horizon_counts
    )

    probabilities = [
        count / total
        for count
        in horizon_counts
    ]

    entropy_bits = -sum(
        probability
        * math.log2(
            probability
        )
        for probability
        in probabilities
        if probability > 0
    )

    normalized_entropy = (
        entropy_bits
        /
        math.log2(
            len(
                macro_zones
            )
        )
    )

    entropy_summary[
        str(
            horizon
        )
    ] = {
        "entropy_bits":
            entropy_bits,

        "normalized_entropy":
            normalized_entropy,
    }


# ============================================================
# Full 15-zone representation check
# ============================================================

zone_representation = {}


for horizon in HORIZONS:

    observed = set(
        zone_support
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )[
            "target_macro_zone"
        ]
        .to_list()
    )

    missing = sorted(
        set(macro_zones)
        - observed
    )

    zone_representation[
        str(
            horizon
        )
    ] = {
        "n_observed":
            len(observed),

        "missing_zones":
            missing,
    }


# ============================================================
# Print report
# ============================================================

print("=" * 104)
print("V3 GATE 2E — TARGET DISTRIBUTION AUDIT")
print("=" * 104)


print()
print("CURRENT-POINT COMPLETENESS")
print("-" * 104)

print(
    "Unique current decision points:",
    f"{n_current_points:,}",
)

print(
    "Both +5s and +10s valid:",
    f"{n_both_valid:,}",
    f"({n_both_valid / n_current_points:.2%})",
)

print(
    "+5s valid, +10s boundary-excluded:",
    f"{n_only_5_valid:,}",
    f"({n_only_5_valid / n_current_points:.2%})",
)

print(
    "Neither horizon valid:",
    f"{n_neither_valid:,}",
    f"({n_neither_valid / n_current_points:.2%})",
)


print()
print("EXCLUSION INTEGRITY")
print("-" * 104)

print(
    "Observed exclusion reasons:",
    sorted(
        exclusion_reasons
    ),
)

print(
    "Semantic/snapshot integrity failures:",
    0,
)


print()
print("PERSISTENCE BASELINE — HARD ACCURACY")
print("-" * 104)

for horizon in HORIZONS:

    row = (
        stay_move
        .filter(
            (
                pl.col(
                    "horizon_sec"
                )
                == horizon
            )
            &
            (
                pl.col(
                    "stay_same_zone"
                )
                == True
            )
        )
        .row(
            0,
            named=True,
        )
    )

    print(
        f"+{horizon}s:",
        f"{row['share']:.4%}",
        (
            "(predict future zone = "
            "current zone)"
        ),
    )


print()
print("MOVE RATE")
print("-" * 104)

for horizon in HORIZONS:

    row = (
        stay_move
        .filter(
            (
                pl.col(
                    "horizon_sec"
                )
                == horizon
            )
            &
            (
                pl.col(
                    "stay_same_zone"
                )
                == False
            )
        )
        .row(
            0,
            named=True,
        )
    )

    print(
        f"+{horizon}s:",
        f"{row['share']:.4%}",
    )


print()
print("TARGET SOURCE DISTRIBUTION")
print("-" * 104)

for horizon in HORIZONS:

    print()
    print(
        f"+{horizon}s"
    )

    rows_h = (
        target_sources
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )
        .to_dicts()
    )

    for row in rows_h:

        print(
            f"  {row['target_source']:<24}"
            f"{row['n']:>6}  "
            f"{row['share']:>8.3%}"
        )


print()
print("TARGET ZONE SUPPORT — ALL 15 ZONES")
print("-" * 104)

for horizon in HORIZONS:

    print()
    print(
        f"+{horizon}s"
    )

    rows_h = (
        zone_support
        .filter(
            pl.col(
                "horizon_sec"
            )
            == horizon
        )
        .sort(
            "n",
            descending=True,
        )
        .to_dicts()
    )

    for row in rows_h:

        markers = []

        if row[
            "below_20"
        ]:
            markers.append(
                "<20"
            )

        elif row[
            "below_50"
        ]:
            markers.append(
                "<50"
            )

        elif row[
            "below_100"
        ]:
            markers.append(
                "<100"
            )

        marker_text = (
            " [" + ",".join(markers) + "]"
            if markers
            else ""
        )

        print(
            f"  {row['target_macro_zone']:<20}"
            f"{row['n']:>6}  "
            f"{row['share']:>8.3%}"
            f"{marker_text}"
        )


print()
print("TARGET DISTRIBUTION ENTROPY")
print("-" * 104)

for horizon in HORIZONS:

    stats = entropy_summary[
        str(
            horizon
        )
    ]

    print(
        f"+{horizon}s:",
        f"H={stats['entropy_bits']:.4f} bits",
        (
            "normalized="
            f"{stats['normalized_entropy']:.4f}"
        ),
    )


print()
print("ZONE REPRESENTATION")
print("-" * 104)

for horizon in HORIZONS:

    info = zone_representation[
        str(
            horizon
        )
    ]

    print(
        f"+{horizon}s:",
        f"{info['n_observed']}/15 zones represented",
    )

    print(
        "  missing:",
        info[
            "missing_zones"
        ],
    )


print()
print("TOP NON-PERSISTENCE TRANSITIONS")
print("-" * 104)

for horizon in HORIZONS:

    print()
    print(
        f"+{horizon}s"
    )

    rows_h = (
        transitions
        .filter(
            (
                pl.col(
                    "horizon_sec"
                )
                == horizon
            )
            &
            (
                pl.col(
                    "is_persistence"
                )
                == False
            )
        )
        .sort(
            "n",
            descending=True,
        )
        .head(
            15
        )
        .to_dicts()
    )

    for row in rows_h:

        print(
            "  "
            f"{row['current_macro_zone']:<18}"
            " -> "
            f"{row['target_macro_zone']:<18}"
            f"{row['n']:>5}"
        )


# ============================================================
# Machine-readable record
# ============================================================

summary = {
    "version":
        "V3",

    "gate":
        "2E",

    "status":
        "TARGET_DISTRIBUTION_AUDIT_COMPLETE",

    "training_performed":
        False,

    "v2_d_confirm_used":
        False,

    "current_decision_points":
        n_current_points,

    "both_horizons_valid":
        n_both_valid,

    "only_5s_valid":
        n_only_5_valid,

    "neither_horizon_valid":
        n_neither_valid,

    "exclusion_reasons":
        sorted(
            exclusion_reasons
        ),

    "semantic_or_snapshot_integrity_failures":
        0,

    "zone_representation":
        zone_representation,

    "entropy":
        entropy_summary,

    "notes": [
        (
            "Class-support thresholds shown in the "
            "terminal are descriptive diagnostics only."
        ),
        (
            "No macro-zone mapping or target contract "
            "was changed using these distributions."
        ),
        (
            "Hard persistence accuracy is descriptive. "
            "A probabilistic persistence baseline must "
            "be defined separately for log-loss scoring."
        ),
    ],
}


SUMMARY_OUTPUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 104)

print(
    "✅ V3 GATE 2E TARGET DISTRIBUTION AUDIT COMPLETE"
)

print(
    "NEXT_ACTION=DECIDE_GATE_2_FREEZE_AND_BASELINE_PROTOCOL"
)

print("=" * 104)

print()
print("Artifacts:")
print(" ", ZONE_SUPPORT_OUTPUT)
print(" ", CURRENT_SUPPORT_OUTPUT)
print(" ", TRANSITION_OUTPUT)
print(" ", SUMMARY_OUTPUT)
