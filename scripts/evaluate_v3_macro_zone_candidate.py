from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


MAPPING_PATH = Path(
    "docs/v3_macro_zone_candidate_v1.json"
)

PREDICTIONS_PATH = Path(
    "data/interim/v3_xyz_place_predictions.csv"
)

OUTPUT_PATH = Path(
    "data/interim/v3_macro_zone_candidate_v1_evaluation.csv"
)

CONFUSION_OUTPUT = Path(
    "data/interim/v3_macro_zone_candidate_v1_confusions.csv"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


mapping_record = json.loads(
    MAPPING_PATH.read_text()
)

predictions = pd.read_csv(
    PREDICTIONS_PATH
)


# ------------------------------------------------------------
# Invert mapping
# ------------------------------------------------------------

place_to_zone = {}

for zone, places in (
    mapping_record[
        "zones"
    ].items()
):

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate mapped place: {place}",
        )

        place_to_zone[
            place
        ] = zone


observed_places = set(
    predictions[
        "true_place"
    ].astype(str)
)

mapped_places = set(
    place_to_zone
)

require(
    observed_places
    == mapped_places,
    (
        "Candidate mapping does not exactly cover "
        "the 23-place semantic seed.\n"
        f"Missing: {sorted(observed_places - mapped_places)}\n"
        f"Extra: {sorted(mapped_places - observed_places)}"
    ),
)


# ------------------------------------------------------------
# Map true/predicted fine places
# ------------------------------------------------------------

predictions[
    "true_macro"
] = (
    predictions[
        "true_place"
    ].map(place_to_zone)
)

predictions[
    "pred_macro"
] = (
    predictions[
        "predicted_place"
    ].map(place_to_zone)
)

predictions[
    "macro_correct"
] = (
    predictions[
        "true_macro"
    ]
    ==
    predictions[
        "pred_macro"
    ]
)


# ------------------------------------------------------------
# Per-demo metrics
# ------------------------------------------------------------

rows = []

for demo, group in predictions.groupby(
    "demo_filename"
):

    truth = group[
        "true_macro"
    ]

    pred = group[
        "pred_macro"
    ]

    rows.append({
        "demo_filename":
            demo,

        "n":
            len(group),

        "macro_accuracy":
            accuracy_score(
                truth,
                pred,
            ),

        "macro_f1":
            f1_score(
                truth,
                pred,
                average="macro",
                zero_division=0,
            ),
    })


result = (
    pd.DataFrame(
        rows
    )
    .sort_values(
        "macro_accuracy"
    )
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ------------------------------------------------------------
# Global metrics
# ------------------------------------------------------------

truth = predictions[
    "true_macro"
]

pred = predictions[
    "pred_macro"
]

global_accuracy = accuracy_score(
    truth,
    pred,
)

global_f1 = f1_score(
    truth,
    pred,
    average="macro",
    zero_division=0,
)

fine_accuracy = float(
    predictions[
        "correct"
    ].mean()
)

absorbed_errors = int(
    (
        ~predictions[
            "correct"
        ]
        &
        predictions[
            "macro_correct"
        ]
    ).sum()
)

fine_errors = int(
    (
        ~predictions[
            "correct"
        ]
    ).sum()
)

remaining_macro_errors = int(
    (
        ~predictions[
            "macro_correct"
        ]
    ).sum()
)


# ------------------------------------------------------------
# Macro confusion
# ------------------------------------------------------------

errors = predictions[
    ~predictions[
        "macro_correct"
    ]
]

confusions = (
    errors
    .groupby(
        [
            "true_macro",
            "pred_macro",
        ]
    )
    .size()
    .reset_index(
        name="n_errors"
    )
    .sort_values(
        "n_errors",
        ascending=False,
    )
)

confusions[
    "fraction_macro_errors"
] = (
    confusions[
        "n_errors"
    ]
    / remaining_macro_errors
)

confusions.to_csv(
    CONFUSION_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("=" * 100)
print("V3 GATE 1F — MACRO-ZONE CANDIDATE V1")
print("=" * 100)

print()
print(
    "Fine-grained places:",
    len(place_to_zone),
)

print(
    "Candidate macro-zones:",
    len(
        mapping_record[
            "zones"
        ]
    ),
)

print(
    "Samples:",
    f"{len(predictions):,}",
)

print()
print(
    "23-place XYZ accuracy:",
    f"{fine_accuracy:.6%}",
)

print(
    "15-zone XYZ accuracy:",
    f"{global_accuracy:.6%}",
)

print(
    "15-zone Macro F1:",
    f"{global_f1:.6f}",
)

print()
print(
    "Fine-place errors:",
    f"{fine_errors:,}",
)

print(
    "Errors absorbed by tactical abstraction:",
    f"{absorbed_errors:,}",
)

print(
    "Absorbed fraction:",
    f"{absorbed_errors / fine_errors:.2%}",
)

print(
    "Remaining macro errors:",
    f"{remaining_macro_errors:,}",
)

print()
print(
    "Minimum fold macro accuracy:",
    f"{result['macro_accuracy'].min():.6%}",
)

print(
    "Mean fold macro accuracy:",
    f"{result['macro_accuracy'].mean():.6%}",
)


print()
print("=" * 100)
print("PER-DEMO")
print("=" * 100)

print(
    result.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TOP REMAINING MACRO CONFUSIONS")
print("=" * 100)

print(
    confusions.head(20)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

print(
    "This evaluates a predeclared tactical abstraction."
)

print(
    "It does not tune the XYZ resolver."
)

print(
    "It does not yet freeze the macro-zone mapping."
)

print(
    "NEXT_ACTION=AUDIT_MACRO_ZONE_TRANSITION_STRUCTURE"
)

print("=" * 100)

print()
print("Artifacts:")
print(" ", OUTPUT_PATH)
print(" ", CONFUSION_OUTPUT)
