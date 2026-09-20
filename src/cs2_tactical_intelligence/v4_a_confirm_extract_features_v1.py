"""V4-A raw-demo target, motion, and Team Context integration.

All current-time features originate from the same parsed raw demo.
Future snapshots are used only by the target extractor.

No frozen Development feature cache, model, or scoring access.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cs2_tactical_intelligence import (
    v4_a_confirm_target_rows_v1 as targets,
    v4_a_confirm_motion_v1 as motion,
    v4_a_confirm_team_context_v1 as team,
    v4_a_confirm_feature_matrix_v1 as features,
)


MOTION_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "current_tick",
]

TEAM_KEY = [
    "demo_filename",
    "round_num",
    "current_tick",
]

TARGET_KEY = [
    "demo_filename",
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
]


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def extract_features(inputs):
    """Independently construct one demo's target and feature rows."""

    extracted = targets.extract_target_rows(inputs)
    rows = extracted["rows"]

    require(
        len(rows) > 0,
        "Demo has no retained target rows.",
    )

    # One motion result per nominal current observation.
    # One Team Context vector per resolved current snapshot.
    motion_requests = {}
    current_xyz_by_snapshot = {}

    for row in rows:
        round_num = int(row["round_num"])
        nominal = int(row["current_nominal_tick"])
        current_tick = int(row["current_tick"])

        xyz = (
            float(row["current_bomb_X"]),
            float(row["current_bomb_Y"]),
            float(row["current_bomb_Z"]),
        )

        motion_key = (round_num, nominal, current_tick)
        snapshot_key = (round_num, current_tick)

        if motion_key in motion_requests:
            require(
                motion_requests[motion_key] == xyz,
                f"Inconsistent current XYZ for {motion_key}",
            )
        else:
            motion_requests[motion_key] = xyz

        if snapshot_key in current_xyz_by_snapshot:
            require(
                current_xyz_by_snapshot[snapshot_key] == xyz,
                f"Inconsistent XYZ for resolved snapshot {snapshot_key}",
            )
        else:
            current_xyz_by_snapshot[snapshot_key] = xyz

    # Determine historical ticks exclusively from current-time ticks
    # and the frozen one-second causal motion rule.
    prior_by_request = {}
    required_keys = set(current_xyz_by_snapshot)

    for key in motion_requests:
        round_num, nominal, current_tick = key

        prior_tick = motion.resolve_snapshot_tick(
            inputs.ticks_by_round.get(round_num, []),
            current_tick - motion.HISTORY_TICKS,
        )

        require(
            prior_tick is not None,
            f"Required causal history unavailable: {key}",
        )

        require(
            prior_tick < current_tick,
            f"Historical snapshot is not earlier than current: {key}",
        )

        prior_by_request[key] = prior_tick
        required_keys.add((round_num, prior_tick))

    # Materialize current and historical snapshots together.
    # No future target-time snapshot is requested here.
    snapshots = inputs.materialize_snapshots(required_keys)

    require(
        set(snapshots) == required_keys,
        "Current/historical snapshot coverage mismatch.",
    )

    team_records = []

    for round_num, current_tick in sorted(current_xyz_by_snapshot):
        snapshot = snapshots[(round_num, current_tick)]

        context = team.build_team_context(snapshot)

        record = {
            "demo_filename": inputs.demo_filename,
            "round_num": round_num,
            "current_tick": current_tick,
        }

        record.update(
            dict(zip(team.TEAM_COLUMNS, context["vector"]))
        )

        team_records.append(record)

    motion_records = []

    for key, xyz in sorted(motion_requests.items()):
        round_num, nominal, current_tick = key
        prior_tick = prior_by_request[key]

        calculated = motion.calculate_motion(
            current_tick=current_tick,
            current_xyz=xyz,
            available_ticks=inputs.ticks_by_round.get(round_num, []),
            snapshots={
                prior_tick: snapshots[(round_num, prior_tick)]
            },
            events=inputs.events_by_round.get(round_num, []),
        )

        require(
            calculated["status"] == "RESOLVED",
            f"Required causal motion unresolved: {key}: "
            f"{calculated.get('reason')}",
        )

        require(
            calculated["prior_tick"] == prior_tick,
            f"Historical tick selection mismatch: {key}",
        )

        motion_records.append({
            "demo_filename": inputs.demo_filename,
            "round_num": round_num,
            "current_nominal_tick": nominal,
            "current_tick": current_tick,
            **calculated,
        })

    target_frame = pl.DataFrame(
        rows,
        infer_schema_length=None,
    )

    motion_frame = pl.DataFrame(
        motion_records,
        infer_schema_length=None,
    )

    team_frame = pl.DataFrame(
        team_records,
        infer_schema_length=None,
    )

    require(
        target_frame.select(TARGET_KEY).unique().height
        == target_frame.height,
        "Duplicate target-row keys.",
    )

    require(
        motion_frame.select(MOTION_KEY).unique().height
        == motion_frame.height,
        "Duplicate motion keys.",
    )

    require(
        team_frame.select(TEAM_KEY).unique().height
        == team_frame.height,
        "Duplicate Team Context keys.",
    )

    joined = (
        target_frame
        .join(
            motion_frame.select(
                MOTION_KEY
                + ["velocity_X", "velocity_Y", "velocity_Z", "status"]
            ),
            on=MOTION_KEY,
            how="left",
            validate="m:1",
        )
        .join(
            team_frame,
            on=TEAM_KEY,
            how="left",
            validate="m:1",
        )
        .sort(TARGET_KEY)
    )

    require(
        joined.height == target_frame.height,
        "Feature joins changed the target-row population.",
    )

    require(
        joined["status"].null_count() == 0
        and joined.filter(
            pl.col("status") != "RESOLVED"
        ).height == 0,
        "Retained target rows contain missing or unresolved motion.",
    )

    require(
        all(
            joined[column].null_count() == 0
            for column in team.TEAM_COLUMNS
        ),
        "Retained target rows contain missing Team Context.",
    )

    matrices = {}

    for horizon in (5, 10):
        frame = joined.filter(
            pl.col("horizon_sec") == horizon
        )

        require(
            frame.height > 0,
            f"No retained +{horizon}s target rows.",
        )

        control = features.build_control_features(frame)

        candidate = features.build_candidate_features(
            frame,
            control,
        )

        require(
            control.shape == (frame.height, 24)
            and candidate.shape == (frame.height, 56)
            and control.dtype == np.float32
            and candidate.dtype == np.float32
            and np.isfinite(control).all()
            and np.isfinite(candidate).all()
            and np.array_equal(candidate[:, :24], control),
            f"Frozen +{horizon}s feature matrix invariant failed.",
        )

        matrices[horizon] = {
            "rows": frame,
            "control": control,
            "candidate": candidate,
        }

    return {
        "targets": target_frame,
        "motion": motion_frame,
        "team_context": team_frame,
        "joined": joined,
        "matrices": matrices,
        "exclusions": extracted["exclusions"],
        "round_exclusions": extracted["round_exclusions"],
    }
