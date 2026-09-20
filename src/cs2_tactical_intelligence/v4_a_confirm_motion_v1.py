"""V4-A confirmation: causal bomb-motion calculation.

Reproduces frozen V3 B1 motion semantics.
This module does not load demos, fit models, or score predictions.
"""

from bisect import bisect_left, bisect_right

import numpy as np

from cs2_tactical_intelligence.v0.features import has_c4


RAW_TICKS_PER_SECOND = 64
HISTORY_TICKS = 64
MAX_SNAPSHOT_LATENESS = 1


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def finite_xyz(x, y, z):
    if x is None or y is None or z is None:
        return False

    return bool(np.isfinite([
        float(x),
        float(y),
        float(z),
    ]).all())


def resolve_snapshot_tick(available_ticks, nominal_tick):
    """First available tick >= nominal, at most one tick late."""

    if not available_ticks:
        return None

    require(
        all(a < b for a, b in zip(
            available_ticks,
            available_ticks[1:],
        )),
        "Snapshot ticks must be strictly increasing.",
    )

    index = bisect_left(available_ticks, nominal_tick)

    if index >= len(available_ticks):
        return None

    tick = int(available_ticks[index])
    lateness = tick - nominal_tick

    if lateness < 0 or lateness > MAX_SNAPSHOT_LATENESS:
        return None

    return tick


def latest_event_at_or_before(events, tick):
    """Use only bomb events observed at or before the requested tick."""

    if not events:
        return None

    event_ticks = [int(event["tick"]) for event in events]

    require(
        event_ticks == sorted(event_ticks),
        "Bomb events must be sorted by tick.",
    )

    index = bisect_right(event_ticks, tick) - 1

    return None if index < 0 else events[index]


def resolve_prior_bomb_xyz(*, snapshot, events, tick):
    """Reproduce the frozen V3 historical bomb-position priority."""

    carriers = [
        player
        for player in snapshot
        if player["side"] == "t" and has_c4(player["inventory"])
    ]

    require(
        len(carriers) <= 1,
        "Invariant violation: multiple C4 carriers.",
    )

    if len(carriers) == 1:
        carrier = carriers[0]

        # Preserve V3 behavior: this branch does not add a health check.
        if not finite_xyz(
            carrier["X"], carrier["Y"], carrier["Z"]
        ):
            return {
                "ok": False,
                "reason": "PRIOR_CARRIER_XYZ_INVALID",
            }

        return {
            "ok": True,
            "source": "CARRIED_INVENTORY",
            "X": float(carrier["X"]),
            "Y": float(carrier["Y"]),
            "Z": float(carrier["Z"]),
        }

    event = latest_event_at_or_before(events, tick)

    if event is None:
        return {
            "ok": False,
            "reason": "PRIOR_BOMB_STATE_UNRESOLVED",
        }

    event_type = str(event["event"])

    if event_type == "pickup":
        steamid = event["steamid"]

        if steamid is None:
            return {
                "ok": False,
                "reason": "PRIOR_PICKUP_STEAMID_MISSING",
            }

        matching = [
            player
            for player in snapshot
            if (
                player["side"] == "t"
                and player["steamid"] == steamid
            )
        ]

        if len(matching) != 1:
            return {
                "ok": False,
                "reason": "PRIOR_PICKUP_PLAYER_UNRESOLVED",
            }

        carrier = matching[0]

        if carrier["health"] is None or carrier["health"] <= 0:
            return {
                "ok": False,
                "reason": "PRIOR_PICKUP_PLAYER_NOT_ALIVE",
            }

        if not finite_xyz(
            carrier["X"], carrier["Y"], carrier["Z"]
        ):
            return {
                "ok": False,
                "reason": "PRIOR_PICKUP_XYZ_INVALID",
            }

        return {
            "ok": True,
            "source": "CARRIED_PICKUP_FALLBACK",
            "X": float(carrier["X"]),
            "Y": float(carrier["Y"]),
            "Z": float(carrier["Z"]),
        }

    if event_type == "drop":
        if not finite_xyz(
            event["X"], event["Y"], event["Z"]
        ):
            return {
                "ok": False,
                "reason": "PRIOR_DROP_XYZ_INVALID",
            }

        return {
            "ok": True,
            "source": "DROPPED",
            "X": float(event["X"]),
            "Y": float(event["Y"]),
            "Z": float(event["Z"]),
        }

    return {
        "ok": False,
        "reason": (
            "PRIOR_BOMB_EVENT_UNSUPPORTED_"
            + event_type.upper()
        ),
    }


def calculate_motion(
    *,
    current_tick,
    current_xyz,
    available_ticks,
    snapshots,
    events,
):
    """Calculate one causal velocity vector from current and prior state.

    available_ticks: sorted player-snapshot ticks for the SAME round.
    snapshots: mapping from prior tick to that round's player rows.
    events: sorted bomb events for the SAME round.

    Missing history returns UNRESOLVED. The eventual confirmation
    pipeline must stop before scoring if required motion is unresolved.
    """

    current_tick = int(current_tick)
    prior_nominal_tick = current_tick - HISTORY_TICKS

    prior_tick = resolve_snapshot_tick(
        available_ticks,
        prior_nominal_tick,
    )

    base = {
        "prior_nominal_tick": prior_nominal_tick,
        "prior_tick": prior_tick,
    }

    if prior_tick is None:
        return {
            **base,
            "status": "UNRESOLVED",
            "reason": "PRIOR_SNAPSHOT_UNAVAILABLE",
        }

    snapshot = snapshots.get(prior_tick)

    if not snapshot:
        return {
            **base,
            "status": "UNRESOLVED",
            "reason": "PRIOR_SNAPSHOT_EMPTY",
        }

    prior = resolve_prior_bomb_xyz(
        snapshot=snapshot,
        events=events,
        tick=prior_tick,
    )

    if not prior["ok"]:
        return {
            **base,
            "status": "UNRESOLVED",
            "reason": prior["reason"],
        }

    elapsed_ticks = current_tick - prior_tick

    require(
        elapsed_ticks > 0,
        "Non-positive history interval.",
    )

    elapsed_sec = elapsed_ticks / RAW_TICKS_PER_SECOND

    require(
        len(current_xyz) == 3
        and finite_xyz(*current_xyz),
        "Invalid current bomb XYZ.",
    )

    velocity = tuple(
        (float(current_xyz[i]) - float(prior[axis]))
        / elapsed_sec
        for i, axis in enumerate(("X", "Y", "Z"))
    )

    require(
        bool(np.isfinite(velocity).all()),
        "Non-finite causal velocity.",
    )

    return {
        **base,
        "prior_source": prior["source"],
        "prior_bomb_X": prior["X"],
        "prior_bomb_Y": prior["Y"],
        "prior_bomb_Z": prior["Z"],
        "elapsed_ticks": elapsed_ticks,
        "elapsed_sec": elapsed_sec,
        "velocity_X": velocity[0],
        "velocity_Y": velocity[1],
        "velocity_Z": velocity[2],
        "status": "RESOLVED",
        "reason": "RESOLVED",
    }
