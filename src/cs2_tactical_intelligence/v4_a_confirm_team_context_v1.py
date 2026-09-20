"""Strict current-time Team Context for V4-A confirmation.

Input: player records from ONE resolved current-time snapshot.
Output: frozen 32D team occupancy vector.

Does not read future state, load models, or calculate predictions.
"""

from cs2_tactical_intelligence import (
    v4_a_confirm_target_semantics_v1 as target,
    v4_a_confirm_feature_matrix_v1 as features,
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


ZONES = list(features.class_order)
TEAM_COLUMNS = list(features.team_columns)
PLACE_TO_ZONE = dict(target.place_to_zone)

require(
    len(ZONES) == 15
    and len(TEAM_COLUMNS) == 32
    and TEAM_COLUMNS == (
        [f"v4_t__{zone}" for zone in ZONES]
        + [f"v4_ct__{zone}" for zone in ZONES]
        + ["v4_t__UNKNOWN_PLACE", "v4_ct__UNKNOWN_PLACE"]
    ),
    "Frozen Team Context feature order mismatch.",
)

require(
    set(PLACE_TO_ZONE.values()) == set(ZONES),
    "Frozen macro-zone mapping mismatch.",
)


def usable_steamid(raw):
    """Require a positive integer Steam ID for a living player."""

    require(raw is not None, "Living player has missing steamid.")

    value = str(raw).strip()

    require(
        value.isdecimal(),
        f"Living player has unusable steamid: {raw!r}",
    )

    steamid = int(value)

    require(
        steamid > 0,
        f"Living player has invalid steamid: {raw!r}",
    )

    return steamid


def build_team_context(snapshot):
    """Count living players in one resolved current-time snapshot."""

    require(
        snapshot is not None and len(snapshot) > 0,
        "Current player snapshot is missing or empty.",
    )

    counts = {
        "t": {zone: 0 for zone in ZONES},
        "ct": {zone: 0 for zone in ZONES},
    }

    unknown = {"t": 0, "ct": 0}
    living = {"t": 0, "ct": 0}
    seen_ids = set()

    for player in snapshot:
        side = player["side"]
        health = player["health"]

        if side not in ("t", "ct"):
            continue

        if health is None or health <= 0:
            continue

        steamid = usable_steamid(player["steamid"])

        require(
            steamid not in seen_ids,
            f"Duplicate living player steamid: {steamid}",
        )

        seen_ids.add(steamid)
        living[side] += 1

        zone = PLACE_TO_ZONE.get(player["place"])

        if zone is None:
            unknown[side] += 1
        else:
            counts[side][zone] += 1

    require(
        1 <= living["t"] <= 5
        and 1 <= living["ct"] <= 5,
        (
            "Invalid living team size: "
            f"T={living['t']} CT={living['ct']}"
        ),
    )

    vector = (
        [counts["t"][zone] for zone in ZONES]
        + [counts["ct"][zone] for zone in ZONES]
        + [unknown["t"], unknown["ct"]]
    )

    require(
        len(vector) == 32
        and all(isinstance(v, int) and v >= 0 for v in vector),
        "Invalid 32D occupancy vector.",
    )

    require(
        sum(vector[:15]) + vector[30] == living["t"],
        "T-side occupancy conservation failed.",
    )

    require(
        sum(vector[15:30]) + vector[31] == living["ct"],
        "CT-side occupancy conservation failed.",
    )

    require(
        len(seen_ids) == living["t"] + living["ct"],
        "Living player identity count mismatch.",
    )

    return {
        "vector": vector,
        "living_t": living["t"],
        "living_ct": living["ct"],
    }
