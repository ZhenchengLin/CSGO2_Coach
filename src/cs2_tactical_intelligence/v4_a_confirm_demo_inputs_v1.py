"""Shared raw-demo inputs for V4-A confirmation extraction.

Parse a demo once and expose the frozen target/motion dependencies.
No Development cache, confirmation queue, model, or scoring access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
    build_plant_lookup,
)
from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
)

from cs2_tactical_intelligence import (
    v4_a_confirm_target_semantics_v1 as target,
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


@dataclass
class DemoInputs:
    demo_filename: str
    demo: object
    ticks_by_round: dict
    events_by_round: dict
    plant_lookup: dict
    unresolved_plant_rounds: set
    drop_lookup: dict
    drop_audit_rows: list

    def materialize_snapshots(self, required_keys):
        """Retrieve exact (round_num, tick) player snapshots."""

        return target.materialize_snapshots(
            self.demo,
            required_keys,
        )


def parse_demo_inputs(
    demo_path,
    *,
    demo_filename=None,
    expected_sha256=None,
):
    """Parse one raw demo without reading existing feature datasets.

    expected_sha256 is optional for this reusable function.
    The confirmation acquisition pipeline must supply and verify it.
    """

    path = Path(demo_path)

    require(
        path.is_file(),
        f"Raw demo does not exist: {path}",
    )

    name = (
        path.name
        if demo_filename is None
        else str(demo_filename)
    )

    if expected_sha256 is not None:
        require(
            sha256(path) == expected_sha256,
            f"Raw demo SHA256 mismatch: {name}",
        )

    require(
        V0_DEMO_TICKS_PER_SECOND == 64,
        "Frozen parser tick-clock constant is not 64.",
    )

    demo = open_v0_demo(
        path,
        verbose=False,
    )

    require(
        demo.header.get("map_name") == "de_mirage",
        f"Expected de_mirage: {name}",
    )

    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )

    required_tick_columns = {
        "round_num",
        "tick",
        "steamid",
        "side",
        "health",
        "X",
        "Y",
        "Z",
        "inventory",
        "place",
    }

    require(
        required_tick_columns <= set(demo.ticks.columns),
        f"Required player telemetry missing: {name}",
    )

    ticks_by_round = target.build_ticks_by_round(demo)
    events_by_round = target.build_events_by_round(demo)

    plant_lookup, plant_issues = build_plant_lookup(demo)

    unresolved_plant_rounds = {
        int(issue["round_num"])
        for issue in plant_issues
        if issue["reason"] == "PLANT_LABEL_UNRESOLVED"
    }

    drop_lookup, drop_audit_rows = target.build_drop_lookup(
        demo,
        name,
    )

    return DemoInputs(
        demo_filename=name,
        demo=demo,
        ticks_by_round=ticks_by_round,
        events_by_round=events_by_round,
        plant_lookup=plant_lookup,
        unresolved_plant_rounds=unresolved_plant_rounds,
        drop_lookup=drop_lookup,
        drop_audit_rows=drop_audit_rows,
    )
