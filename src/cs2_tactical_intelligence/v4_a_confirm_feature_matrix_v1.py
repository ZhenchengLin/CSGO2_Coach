"""V4-A frozen Control/Candidate feature construction.

Functions copied verbatim from the frozen four-model fit script.
No demo parsing, model loading, fitting, or prediction.
"""

import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


_fit = json.loads(
    Path(
        "docs/v4_a_confirm_fit_contract_v1_frozen.json"
    ).read_text()
)

_extraction = json.loads(
    Path(
        "docs/v4_a_confirm_feature_extraction_contract_v1_frozen.json"
    ).read_text()
)

class_order = _fit["development_population"]["target_classes"]

zone_to_index = {
    zone: index
    for index, zone in enumerate(class_order)
}

team_columns = (
    _extraction["team_context_32d"]["feature_order"]
)

control_columns = (
    _fit["model_variants"]["control"]["feature_order"]
)

candidate_columns = (
    _fit["model_variants"]["candidate"]["feature_order"]
)

require(
    len(class_order) == 15
    and len(control_columns) == 24
    and len(team_columns) == 32
    and len(candidate_columns) == 56,
    "Unexpected frozen feature dimensions.",
)

require(
    control_columns
    == _extraction["control_24d"]["feature_order"],
    "Control feature ordering mismatch.",
)

require(
    candidate_columns
    == control_columns + team_columns
    == _extraction["candidate_56d"]["feature_order"],
    "Candidate feature ordering mismatch.",
)




def build_control_features(frame):
    n = frame.height
    X = np.zeros((n, 24), dtype=np.float32)

    for index, zone in enumerate(frame["current_macro_zone"].to_list()):
        require(zone in zone_to_index, f"Unknown current zone: {zone}")
        X[index, zone_to_index[zone]] = 1.0

    for index, source in enumerate(frame["current_source"].to_list()):
        if source in ("CARRIED_INVENTORY", "CARRIED_PICKUP_FALLBACK"):
            X[index, 15] = 1.0
        elif source == "DROPPED":
            X[index, 16] = 1.0
        else:
            raise RuntimeError(f"STOP: Unexpected bomb source: {source}")

    xyz = np.column_stack([
        frame["current_bomb_X"].to_numpy(),
        frame["current_bomb_Y"].to_numpy(),
        frame["current_bomb_Z"].to_numpy(),
    ]).astype(np.float32, copy=False)

    velocity = np.column_stack([
        frame["velocity_X"].to_numpy(),
        frame["velocity_Y"].to_numpy(),
        frame["velocity_Z"].to_numpy(),
    ]).astype(np.float32, copy=False)

    speed = np.linalg.norm(
        velocity.astype(np.float64),
        axis=1,
    ).astype(np.float32)

    X[:, 17:20] = xyz
    X[:, 20:23] = velocity
    X[:, 23] = speed

    require(
        X.shape == (n, 24) and np.isfinite(X).all(),
        "Invalid Control features.",
    )

    return X


def build_candidate_features(frame, control):
    team_matrix = np.column_stack([
        frame[column].to_numpy()
        for column in team_columns
    ]).astype(np.float32, copy=False)

    X = np.concatenate([control, team_matrix], axis=1)

    require(
        X.shape == (frame.height, 56)
        and np.isfinite(X).all()
        and np.array_equal(X[:, :24], control),
        "Invalid Candidate features.",
    )

    return X
