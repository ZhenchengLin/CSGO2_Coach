from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CONFUSIONS = Path(
    "data/interim/v3_xyz_place_confusions_symmetric.csv"
)

BOUNDARIES = Path(
    "data/interim/v3_place_boundary_adjacency.csv"
)

TRANSITIONS = Path(
    "data/interim/v3_place_transition_edges.csv"
)

OUTPUT = Path(
    "data/interim/v3_spatial_semantic_crosscheck.csv"
)

SUMMARY_OUTPUT = Path(
    "docs/v3_gate1_spatial_semantic_evidence.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


for path in [
    CONFUSIONS,
    BOUNDARIES,
    TRANSITIONS,
]:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )


# ============================================================
# Load
# ============================================================

confusions = pd.read_csv(
    CONFUSIONS
)

boundaries = pd.read_csv(
    BOUNDARIES
)

transitions = pd.read_csv(
    TRANSITIONS
)


print("=" * 100)
print("V3 GATE 1E — SPATIAL / SEMANTIC EVIDENCE CROSS-CHECK")
print("=" * 100)

print()
print(
    "Confusion pairs:",
    len(confusions),
)

print(
    "Boundary-audit pairs:",
    len(boundaries),
)

print(
    "Trajectory edges:",
    len(transitions),
)


# ============================================================
# Canonical pair identity
# ============================================================

def canonical_pair(a, b):
    return tuple(
        sorted([
            str(a),
            str(b),
        ])
    )


confusions[
    "pair_key"
] = [
    "|".join(
        canonical_pair(
            row.place_a,
            row.place_b,
        )
    )
    for row in confusions.itertuples()
]


boundaries[
    "pair_key"
] = [
    "|".join(
        canonical_pair(
            row.place_a,
            row.place_b,
        )
    )
    for row in boundaries.itertuples()
]


transitions[
    "pair_key"
] = [
    "|".join(
        canonical_pair(
            row.place_a,
            row.place_b,
        )
    )
    for row in transitions.itertuples()
]


require(
    confusions[
        "pair_key"
    ].is_unique,
    "Duplicate confusion pair identities.",
)

require(
    transitions[
        "pair_key"
    ].is_unique,
    "Duplicate transition pair identities.",
)


# ============================================================
# Join independent evidence
# ============================================================

boundary_keep = boundaries[
    [
        "pair_key",
        "boundary_min_distance",
        "boundary_p01_distance",
        "boundary_p05_distance",
        "fraction_within_10",
        "fraction_within_25",
    ]
].copy()


transition_keep = transitions[
    [
        "pair_key",
        "n_transitions",
        "demos_present",
        "unique_players",
        "evidence",
        "step_distance_median",
        "step_distance_p95",
    ]
].copy()


result = (
    confusions
    .merge(
        boundary_keep,
        on="pair_key",
        how="left",
    )
    .merge(
        transition_keep,
        on="pair_key",
        how="left",
    )
)


result["trajectory_evidence"] = (
    result["evidence"]
    .fillna(
        "UNOBSERVED"
    )
)


# ============================================================
# Diagnostic flags
#
# IMPORTANT:
# These characterize evidence.
# They do not define macro-zone merging.
# ============================================================

result[
    "has_trajectory_edge"
] = (
    result[
        "trajectory_evidence"
    ]
    != "UNOBSERVED"
)


result[
    "strong_trajectory_edge"
] = (
    result[
        "trajectory_evidence"
    ]
    .isin([
        "ROBUST",
        "SUPPORTED",
    ])
)


result[
    "close_empirical_boundary"
] = (
    result[
        "boundary_p01_distance"
    ]
    .notna()
    &
    (
        result[
            "boundary_p01_distance"
        ]
        <= 10
    )
)


result[
    "dual_support"
] = (
    result[
        "strong_trajectory_edge"
    ]
    &
    result[
        "close_empirical_boundary"
    ]
)


# ============================================================
# Error mass
# ============================================================

total_errors = int(
    result[
        "n_errors"
    ].sum()
)


def error_mass(mask):
    return float(
        result.loc[
            mask,
            "n_errors",
        ].sum()
        / total_errors
    )


robust_mass = error_mass(
    result[
        "trajectory_evidence"
    ]
    == "ROBUST"
)


supported_mass = error_mass(
    result[
        "trajectory_evidence"
    ]
    == "SUPPORTED"
)


sparse_mass = error_mass(
    result[
        "trajectory_evidence"
    ]
    == "SPARSE"
)


unobserved_mass = error_mass(
    result[
        "trajectory_evidence"
    ]
    == "UNOBSERVED"
)


strong_mass = error_mass(
    result[
        "strong_trajectory_edge"
    ]
)


dual_mass = error_mass(
    result[
        "dual_support"
    ]
)


# ============================================================
# Top confusion diagnostics
# ============================================================

top25 = (
    result
    .sort_values(
        "n_errors",
        ascending=False,
    )
    .head(25)
    .copy()
)


top25_error_mass = float(
    top25[
        "n_errors"
    ].sum()
    / total_errors
)


top25_robust = int(
    (
        top25[
            "trajectory_evidence"
        ]
        == "ROBUST"
    ).sum()
)


top25_strong = int(
    top25[
        "strong_trajectory_edge"
    ].sum()
)


# ============================================================
# Save full evidence table
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result = result.sort_values(
    "n_errors",
    ascending=False,
)

result.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Summary record
# ============================================================

summary = {
    "version":
        "V3",

    "stage":
        "Gate 1 spatial-semantic diagnosis",

    "scientific_role":
        (
            "Cross-check independent classifier, "
            "spatial-boundary, and player-trajectory evidence."
        ),

    "no_model_training":
        True,

    "v2_d_confirm_used":
        False,

    "n_confusion_pairs":
        int(
            len(result)
        ),

    "total_resolver_errors":
        total_errors,

    "top25": {
        "error_mass":
            top25_error_mass,

        "robust_trajectory_edges":
            top25_robust,

        "strong_trajectory_edges":
            top25_strong,

        "n_pairs":
            int(
                len(top25)
            ),
    },

    "error_mass_by_trajectory_evidence": {
        "ROBUST":
            robust_mass,

        "SUPPORTED":
            supported_mass,

        "SPARSE":
            sparse_mass,

        "UNOBSERVED":
            unobserved_mass,

        "ROBUST_OR_SUPPORTED":
            strong_mass,
    },

    "dual_evidence_error_mass":
        dual_mass,

    "interpretation_guardrails": [
        (
            "Trajectory adjacency does not imply "
            "macro-zone equivalence."
        ),
        (
            "Classifier confusion does not define "
            "the semantic graph."
        ),
        (
            "Boundary proximity does not by itself "
            "justify merging tactical regions."
        ),
        (
            "Macro-zone design must preserve "
            "tactically meaningful route distinctions."
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


# ============================================================
# Report
# ============================================================

print()
print("=" * 100)
print("TOP 25 CONFUSION CROSS-CHECK")
print("=" * 100)

print(
    top25[
        [
            "place_a",
            "place_b",
            "n_errors",
            "fraction_all_errors",
            "boundary_p01_distance",
            "trajectory_evidence",
            "n_transitions",
            "demos_present",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("ERROR-MASS SUMMARY")
print("=" * 100)

print(
    "Total resolver errors:",
    f"{total_errors:,}",
)

print(
    "Top-25 error mass:",
    f"{top25_error_mass:.2%}",
)

print(
    "Top-25 ROBUST trajectory pairs:",
    f"{top25_robust}/25",
)

print()
print(
    "ROBUST trajectory error mass:",
    f"{robust_mass:.2%}",
)

print(
    "SUPPORTED trajectory error mass:",
    f"{supported_mass:.2%}",
)

print(
    "SPARSE trajectory error mass:",
    f"{sparse_mass:.2%}",
)

print(
    "UNOBSERVED trajectory error mass:",
    f"{unobserved_mass:.2%}",
)

print()

print(
    "ROBUST + SUPPORTED error mass:",
    f"{strong_mass:.2%}",
)

print(
    "Dual evidence "
    "(trajectory + p01 boundary <= 10) error mass:",
    f"{dual_mass:.2%}",
)


print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

print(
    "This step characterizes WHY the 23-place resolver "
    "disagrees."
)

print(
    "It does NOT decide which places should be merged."
)

print(
    "No model was trained or tuned."
)

print()
print(
    "NEXT_ACTION=DESIGN_TACTICAL_MACRO_ZONE_CANDIDATES"
)

print("=" * 100)

print()
print("Artifacts:")
print(" ", OUTPUT)
print(" ", SUMMARY_OUTPUT)
