"""V4-A target extraction from one parsed raw demo.

Uses the frozen V3 observation grid, target semantics, and exclusions.
The implementation below is extracted from the V4 Target Replay
that reproduced all 53 frozen Development demos.

No Development cache, confirmation queue, model, or scoring access.
"""

import json
from collections import Counter

from cs2_tactical_intelligence import (
    v4_a_confirm_target_semantics_v1 as target,
    v4_a_confirm_feature_matrix_v1 as features,
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def extract_target_rows(inputs):
    """Build target rows and exclusions from one parsed raw demo."""

    demo = inputs.demo

    ticks_by_round = inputs.ticks_by_round
    events_by_round = inputs.events_by_round

    plant_lookup = inputs.plant_lookup
    unresolved_plant_rounds = inputs.unresolved_plant_rounds
    drop_lookup = inputs.drop_lookup

    rules = json.loads(
        target._TARGET_CONTRACT_PATH.read_text()
    )

    start_seconds = int(
        rules["observation_grid"][
            "first_current_observation_sec"
        ]
    )

    stride_seconds = int(
        rules["observation_grid"]["stride_sec"]
    )

    horizons = list(
        rules["prediction_horizons_sec"]
    )

    ticks_per_second = int(
        rules["tick_clock"]["raw_ticks_per_second"]
    )

    require(
        start_seconds == 5
        and stride_seconds == 5
        and horizons == [5, 10]
        and ticks_per_second == 64,
        "Frozen target timing contract mismatch.",
    )

    class_order = list(features.class_order)

    require(
        class_order == list(target._mapping["zones"]),
        "Frozen target class order mismatch.",
    )

    class_to_index = {
        zone: index
        for index, zone in enumerate(class_order)
    }

    timing_records = []
    required_snapshot_keys = set()
    actual_exclusions = Counter()
    actual_round_exclusions = Counter()


    def exclude(round_num, nominal, horizon, reason):
        actual_exclusions[
            (int(round_num), int(nominal), int(horizon), reason)
        ] += 1


    # Stage 1: independently rebuild all nominal observations and
    # horizon-specific timing decisions from the original demo.
    for round_row in (
        demo.rounds
        .sort("round_num")
        .iter_rows(named=True)
    ):
        round_num = int(round_row["round_num"])
        freeze_end = round_row["freeze_end"]
        round_end = round_row["end"]

        if freeze_end is None:
            actual_round_exclusions[
                (round_num, "MISSING_FREEZE_END")
            ] += 1
            continue

        if round_end is None:
            actual_round_exclusions[
                (round_num, "MISSING_ROUND_END")
            ] += 1
            continue

        freeze_end = int(freeze_end)
        round_end = int(round_end)

        if freeze_end >= round_end:
            actual_round_exclusions[
                (round_num, "INVALID_TIMING_ORDER")
            ] += 1
            continue

        if round_num in unresolved_plant_rounds:
            actual_round_exclusions[
                (round_num, "PLANT_LABEL_UNRESOLVED")
            ] += 1
            continue

        plant = plant_lookup.get(round_num)
        plant_tick = None if plant is None else int(plant["plant_tick"])
        plant_label = None if plant is None else str(plant["label"])

        available_ticks = ticks_by_round.get(round_num, [])
        nominal = freeze_end + start_seconds * ticks_per_second
        stride_ticks = stride_seconds * ticks_per_second

        while (
            nominal < round_end
            and (plant_tick is None or nominal < plant_tick)
        ):
            current_resolution = target.resolve_snapshot_tick(
                available_ticks,
                nominal,
                upper_bound_exclusive=round_end,
            )

            if current_resolution is None:
                for horizon in horizons:
                    exclude(
                        round_num, nominal, horizon,
                        "CURRENT_SNAPSHOT_UNAVAILABLE",
                    )
                nominal += stride_ticks
                continue

            current_tick = int(current_resolution["tick"])

            if plant_tick is not None and current_tick >= plant_tick:
                for horizon in horizons:
                    exclude(
                        round_num, nominal, horizon,
                        "CURRENT_STATE_AT_OR_AFTER_PLANT",
                    )
                nominal += stride_ticks
                continue

            required_snapshot_keys.add((round_num, current_tick))

            for horizon in horizons:
                target_nominal = (
                    current_tick + horizon * ticks_per_second
                )

                if target_nominal >= round_end:
                    exclude(
                        round_num, nominal, horizon,
                        "TARGET_AFTER_ROUND_END",
                    )
                    continue

                future_resolution = target.resolve_snapshot_tick(
                    available_ticks,
                    target_nominal,
                    upper_bound_exclusive=round_end,
                )

                if future_resolution is None:
                    exclude(
                        round_num, nominal, horizon,
                        "TARGET_SNAPSHOT_UNAVAILABLE",
                    )
                    continue

                future_tick = int(future_resolution["tick"])

                required_snapshot_keys.add((round_num, future_tick))

                timing_records.append({
                    "round_num": round_num,
                    "current_nominal_tick": nominal,
                    "current_tick": current_tick,
                    "current_lateness": int(
                        current_resolution["lateness"]
                    ),
                    "horizon_sec": horizon,
                    "target_nominal_tick": target_nominal,
                    "target_tick": future_tick,
                    "target_lateness": int(
                        future_resolution["lateness"]
                    ),
                    "plant_tick": plant_tick,
                    "plant_label": plant_label,
                })

            nominal += stride_ticks


    snapshots = target.materialize_snapshots(
        demo,
        required_snapshot_keys,
    )


    # Stage 2: resolve current and future bomb semantics. Future state
    # is used only as the target, never as a current-time feature.
    actual_rows = {}

    for row in timing_records:
        round_num = row["round_num"]
        nominal = row["current_nominal_tick"]
        horizon = row["horizon_sec"]
        current_tick = row["current_tick"]
        future_tick = row["target_tick"]

        current_snapshot = snapshots.get((round_num, current_tick))
        future_snapshot = snapshots.get((round_num, future_tick))

        if not current_snapshot:
            exclude(
                round_num, nominal, horizon,
                "CURRENT_SNAPSHOT_UNAVAILABLE",
            )
            continue

        if not future_snapshot:
            exclude(
                round_num, nominal, horizon,
                "TARGET_SNAPSHOT_UNAVAILABLE",
            )
            continue

        events = events_by_round.get(round_num, [])

        current = target.resolve_unplanted_bomb(
            round_num=round_num,
            tick=current_tick,
            snapshot=current_snapshot,
            events=events,
            drop_lookup=drop_lookup,
            state_error="CURRENT_BOMB_STATE_UNRESOLVED",
            place_error="CURRENT_BOMB_PLACE_UNRESOLVED",
        )

        if not current["ok"]:
            exclude(round_num, nominal, horizon, current["error"])
            continue

        if row["plant_tick"] is not None and (
            row["plant_tick"] <= future_tick
        ):
            if row["plant_label"] == "A_PLANT":
                future = {
                    "ok": True,
                    "source": "PLANTED",
                    "place": "BombsiteA",
                    "macro_zone": "A_SITE_COMPLEX",
                }
            elif row["plant_label"] == "B_PLANT":
                future = {
                    "ok": True,
                    "source": "PLANTED",
                    "place": "BombsiteB",
                    "macro_zone": "B_SITE_COMPLEX",
                }
            else:
                raise RuntimeError("Unexpected canonical plant label.")
        else:
            future = target.resolve_unplanted_bomb(
                round_num=round_num,
                tick=future_tick,
                snapshot=future_snapshot,
                events=events,
                drop_lookup=drop_lookup,
                state_error="TARGET_BOMB_STATE_UNRESOLVED",
                place_error="TARGET_BOMB_PLACE_UNRESOLVED",
            )

        if not future["ok"]:
            exclude(round_num, nominal, horizon, future["error"])
            continue

        current_zone = current["macro_zone"]
        future_zone = future["macro_zone"]

        require(
            current_zone in class_to_index
            and future_zone in class_to_index,
            "Unexpected macro-zone.",
        )

        key = (round_num, nominal, horizon)

        require(key not in actual_rows, f"Duplicate target key: {key}")

        actual_rows[key] = {
            "current_tick": current_tick,
            "current_lateness": row["current_lateness"],
            "target_nominal_tick": row["target_nominal_tick"],
            "target_tick": future_tick,
            "target_lateness": row["target_lateness"],
            "current_source": current["source"],
            "current_place": current["place"],
            "current_macro_zone": current_zone,
            "current_class_index": class_to_index[current_zone],
            "current_bomb_X": float(current["X"]),
            "current_bomb_Y": float(current["Y"]),
            "current_bomb_Z": float(current["Z"]),
            "target_source": future["source"],
            "target_place": future["place"],
            "target_macro_zone": future_zone,
            "target_class_index": class_to_index[future_zone],
            "stay_same_zone": current_zone == future_zone,
        }

    # Metadata is attached only after the target/exclusion decisions.
    # Both horizons retain the same resolved current-time observation.
    rows = []

    for key, values in actual_rows.items():
        round_num, current_nominal_tick, horizon_sec = key

        rows.append({
            "demo_filename": inputs.demo_filename,
            "round_num": round_num,
            "current_nominal_tick": current_nominal_tick,
            "horizon_sec": horizon_sec,
            **values,
        })

    require(
        len(rows) == len(actual_rows),
        "Target-row uniqueness check failed.",
    )

    return {
        "rows": rows,
        "exclusions": actual_exclusions,
        "round_exclusions": actual_round_exclusions,
    }
