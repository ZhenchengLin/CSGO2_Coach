"""
V4-A Confirmation: in-memory target-row and feature alignment.

This module checks arrays that a future Runner will prepare
from independently verified Confirmation artifacts.

It does not read a Manifest, inspect Demo data, load models,
predict, score, select matches, or authorize scoring.
"""

from __future__ import annotations

import numpy as np


TARGET_KEY = (
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
)

CONTROL_DIMENSIONS = 24
CANDIDATE_DIMENSIONS = 56
CLASS_COUNT = 15


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_rows(
    control_keys,
    candidate_keys,
    control_labels,
    candidate_labels,
    control_features,
    candidate_features,
):
    """
    Check one horizon's Control and Candidate inputs.

    Keys must be ordered sequences of:
        (demo_filename, round_num,
         current_nominal_tick, horizon_sec)

    Row identity is checked before any feature comparison.

    This function does not establish that the supplied rows
    came from a frozen Manifest or verified target artifact.
    Those are separate future Runner responsibilities.
    """

    left_keys = tuple(tuple(key) for key in control_keys)
    right_keys = tuple(tuple(key) for key in candidate_keys)

    require(
        bool(left_keys),
        "Target-row collection is empty.",
    )

    require(
        left_keys == right_keys,
        "Control and Candidate target-row order differs.",
    )

    require(
        len(set(left_keys)) == len(left_keys),
        "Duplicate target-row key.",
    )

    horizons = set()

    for key in left_keys:
        require(
            len(key) == len(TARGET_KEY),
            "Unexpected target-row key dimensions.",
        )

        filename, round_num, nominal_tick, horizon = key

        require(
            isinstance(filename, str) and bool(filename),
            "Invalid Demo filename in target-row key.",
        )

        require(
            isinstance(round_num, (int, np.integer))
            and not isinstance(round_num, (bool, np.bool_))
            and round_num > 0,
            "Invalid round number.",
        )

        require(
            isinstance(nominal_tick, (int, np.integer))
            and not isinstance(nominal_tick, (bool, np.bool_))
            and nominal_tick >= 0,
            "Invalid nominal tick.",
        )

        require(
            isinstance(horizon, (int, np.integer))
            and horizon in (5, 10),
            "Unexpected prediction horizon.",
        )

        horizons.add(int(horizon))

    require(
        len(horizons) == 1,
        "One validation call must contain exactly one horizon.",
    )

    left_y = np.asarray(control_labels)
    right_y = np.asarray(candidate_labels)

    require(
        left_y.shape == right_y.shape == (len(left_keys),),
        "Control and Candidate label shapes differ.",
    )

    require(
        left_y.dtype.kind in "iu"
        and right_y.dtype.kind in "iu",
        "Target labels must use integer class indices.",
    )

    require(
        np.array_equal(left_y, right_y),
        "Control and Candidate target labels differ.",
    )

    require(
        np.all((left_y >= 0) & (left_y < CLASS_COUNT)),
        "Target class index is outside 0..14.",
    )

    left_x = np.asarray(control_features)
    right_x = np.asarray(candidate_features)

    require(
        left_x.shape == (
            len(left_keys),
            CONTROL_DIMENSIONS,
        ),
        "Control matrix must have exactly 24 columns.",
    )

    require(
        right_x.shape == (
            len(left_keys),
            CANDIDATE_DIMENSIONS,
        ),
        "Candidate matrix must have exactly 56 columns.",
    )

    require(
        left_x.dtype.kind in "iuf"
        and right_x.dtype.kind in "iuf",
        "Feature matrices must contain numeric values.",
    )

    require(
        np.isfinite(left_x).all()
        and np.isfinite(right_x).all(),
        "Feature matrices contain nonfinite values.",
    )

    require(
        np.array_equal(
            left_x,
            right_x[:, :CONTROL_DIMENSIONS],
        ),
        "Candidate's first 24 features differ from Control.",
    )

    return {
        "status": "IN_MEMORY_ALIGNMENT_PASS",
        "rows": len(left_keys),
        "horizon_sec": horizons.pop(),
        "control_dimensions": CONTROL_DIMENSIONS,
        "candidate_dimensions": CANDIDATE_DIMENSIONS,
        "target_keys_identical": True,
        "target_labels_identical": True,
        "control_feature_prefix_identical": True,
        "final_manifest_verified": False,
        "confirmation_scoring_authorized": False,
    }
