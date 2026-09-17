from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser


INCOMING = Path("data/raw/v2_incoming")
MANIFEST = Path("docs/v2_confirm_manifest.csv")

DIRECTED_OUTPUT = Path(
    "data/interim/v3_place_transitions_directed.csv"
)

EDGE_OUTPUT = Path(
    "data/interim/v3_place_transition_edges.csv"
)

ADJACENCY_OUTPUT = Path(
    "data/interim/v3_place_adjacency_candidates.csv"
)

TICK_RATE = 64.0
MAX_TICK_GAP = 4
MAX_SPEED_UNITS_PER_SEC = 1000.0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# Reserve identity
# ============================================================

confirm = set(
    pd.read_csv(
        MANIFEST
    )["demo_filename"]
)

reserve = sorted(
    p
    for p in INCOMING.glob("*.dem")
    if p.name not in confirm
)

require(
    len(reserve) == 13,
    f"Expected 13 reserve demos, found {len(reserve)}",
)


print("=" * 100)
print("V3 GATE 1D — TRAJECTORY-DERIVED SEMANTIC ADJACENCY")
print("=" * 100)

print()
print("Reserve demos:", len(reserve))
print("V2 D_CONFIRM excluded:", len(confirm))

print()
print(
    "Transition contract:"
)

print(
    f"  tick gap <= {MAX_TICK_GAP}"
)

print(
    f"  speed <= {MAX_SPEED_UNITS_PER_SEC:.0f}"
    " units/sec"
)


# ============================================================
# Accumulators
# ============================================================

directed_counts = defaultdict(int)
directed_demos = defaultdict(set)
directed_players = defaultdict(set)

edge_counts = defaultdict(int)
edge_demos = defaultdict(set)
edge_players = defaultdict(set)
edge_distances = defaultdict(list)
edge_tick_gaps = defaultdict(list)


# ============================================================
# Parse one demo at a time
# ============================================================

for demo_index, path in enumerate(
    reserve,
    start=1,
):

    print()
    print(
        f"[{demo_index:02d}/13] "
        f"{path.name}"
    )

    parser = DemoParser(
        str(path)
    )

    df = parser.parse_ticks([
        "X",
        "Y",
        "Z",
        "last_place_name",
        "health",
        "is_alive",
        "team_num",
        "is_warmup_period",
        "is_freeze_period",
    ])

    place_valid = (
        df["last_place_name"].notna()
        &
        (
            df["last_place_name"]
            .astype(str)
            .str.len()
            > 0
        )
    )

    active = (
        df["is_alive"]
        .fillna(False)
        .astype(bool)
        &
        (
            df["health"]
            .fillna(0)
            > 0
        )
        &
        df["team_num"].isin([2, 3])
        &
        ~df["is_warmup_period"]
        .fillna(False)
        .astype(bool)
        &
        ~df["is_freeze_period"]
        .fillna(False)
        .astype(bool)
    )

    usable = (
        df.loc[
            active & place_valid,
            [
                "tick",
                "steamid",
                "X",
                "Y",
                "Z",
                "last_place_name",
            ],
        ]
        .copy()
        .sort_values(
            [
                "steamid",
                "tick",
            ]
        )
    )

    # ----------------------------------------------
    # Previous active player state
    # ----------------------------------------------

    grouped = usable.groupby(
        "steamid",
        sort=False,
    )

    usable["prev_tick"] = (
        grouped["tick"].shift(1)
    )

    usable["prev_X"] = (
        grouped["X"].shift(1)
    )

    usable["prev_Y"] = (
        grouped["Y"].shift(1)
    )

    usable["prev_Z"] = (
        grouped["Z"].shift(1)
    )

    usable["prev_place"] = (
        grouped[
            "last_place_name"
        ].shift(1)
    )

    usable["delta_tick"] = (
        usable["tick"]
        - usable["prev_tick"]
    )

    usable["dx"] = (
        usable["X"]
        - usable["prev_X"]
    )

    usable["dy"] = (
        usable["Y"]
        - usable["prev_Y"]
    )

    usable["dz"] = (
        usable["Z"]
        - usable["prev_Z"]
    )

    usable["step_xyz"] = np.sqrt(
        usable["dx"] ** 2
        + usable["dy"] ** 2
        + usable["dz"] ** 2
    )

    usable["delta_sec"] = (
        usable["delta_tick"]
        / TICK_RATE
    )

    usable["speed"] = (
        usable["step_xyz"]
        / usable["delta_sec"]
    )

    # ----------------------------------------------
    # True short-range semantic transitions
    # ----------------------------------------------

    transitions = usable[
        usable["prev_place"].notna()
        &
        (
            usable["last_place_name"]
            != usable["prev_place"]
        )
        &
        (
            usable["delta_tick"]
            >= 1
        )
        &
        (
            usable["delta_tick"]
            <= MAX_TICK_GAP
        )
        &
        np.isfinite(
            usable["speed"]
        )
        &
        (
            usable["speed"]
            <= MAX_SPEED_UNITS_PER_SEC
        )
    ].copy()

    print(
        "  active labelled rows:",
        f"{len(usable):,}",
    )

    print(
        "  semantic transitions:",
        f"{len(transitions):,}",
    )

    # ----------------------------------------------
    # Accumulate transition evidence
    # ----------------------------------------------

    for row in transitions.itertuples():

        source = str(
            row.prev_place
        )

        target = str(
            row.last_place_name
        )

        player = str(
            row.steamid
        )

        directed_key = (
            source,
            target,
        )

        directed_counts[
            directed_key
        ] += 1

        directed_demos[
            directed_key
        ].add(
            path.name
        )

        directed_players[
            directed_key
        ].add(
            player
        )

        a, b = sorted([
            source,
            target,
        ])

        edge_key = (
            a,
            b,
        )

        edge_counts[
            edge_key
        ] += 1

        edge_demos[
            edge_key
        ].add(
            path.name
        )

        edge_players[
            edge_key
        ].add(
            player
        )

        edge_distances[
            edge_key
        ].append(
            float(
                row.step_xyz
            )
        )

        edge_tick_gaps[
            edge_key
        ].append(
            int(
                row.delta_tick
            )
        )


# ============================================================
# Directed table
# ============================================================

directed_rows = []

for (
    source,
    target
), count in directed_counts.items():

    directed_rows.append({
        "source_place":
            source,

        "target_place":
            target,

        "n_transitions":
            count,

        "demos_present":
            len(
                directed_demos[
                    (source, target)
                ]
            ),

        "unique_players":
            len(
                directed_players[
                    (source, target)
                ]
            ),
    })


directed = (
    pd.DataFrame(
        directed_rows
    )
    .sort_values(
        [
            "n_transitions",
            "demos_present",
        ],
        ascending=False,
    )
)


# ============================================================
# Undirected semantic edges
# ============================================================

edge_rows = []

for (
    place_a,
    place_b
), count in edge_counts.items():

    distances = np.asarray(
        edge_distances[
            (place_a, place_b)
        ],
        dtype=float,
    )

    gaps = np.asarray(
        edge_tick_gaps[
            (place_a, place_b)
        ],
        dtype=float,
    )

    demos_present = len(
        edge_demos[
            (place_a, place_b)
        ]
    )

    players = len(
        edge_players[
            (place_a, place_b)
        ]
    )

    if demos_present >= 7:
        evidence = "ROBUST"

    elif demos_present >= 3:
        evidence = "SUPPORTED"

    else:
        evidence = "SPARSE"

    edge_rows.append({
        "place_a":
            place_a,

        "place_b":
            place_b,

        "n_transitions":
            count,

        "demos_present":
            demos_present,

        "unique_players":
            players,

        "evidence":
            evidence,

        "step_distance_median":
            float(
                np.median(
                    distances
                )
            ),

        "step_distance_p95":
            float(
                np.percentile(
                    distances,
                    95,
                )
            ),

        "tick_gap_median":
            float(
                np.median(
                    gaps
                )
            ),
    })


edges = (
    pd.DataFrame(
        edge_rows
    )
    .sort_values(
        [
            "demos_present",
            "n_transitions",
        ],
        ascending=False,
    )
)


# ============================================================
# Candidate adjacency graph
#
# IMPORTANT:
# No classifier error information is used here.
# ============================================================

adjacency = edges[
    edges["evidence"].isin(
        [
            "ROBUST",
            "SUPPORTED",
        ]
    )
].copy()


# ============================================================
# Save
# ============================================================

DIRECTED_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

directed.to_csv(
    DIRECTED_OUTPUT,
    index=False,
)

edges.to_csv(
    EDGE_OUTPUT,
    index=False,
)

adjacency.to_csv(
    ADJACENCY_OUTPUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("SEMANTIC EDGE SUMMARY")
print("=" * 100)

print(
    edges[
        [
            "place_a",
            "place_b",
            "n_transitions",
            "demos_present",
            "unique_players",
            "evidence",
            "step_distance_median",
            "step_distance_p95",
        ]
    ]
    .head(50)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("GRAPH SUMMARY")
print("=" * 100)

print(
    "Observed edges:",
    len(edges),
)

print(
    "ROBUST edges:",
    int(
        (
            edges["evidence"]
            == "ROBUST"
        ).sum()
    ),
)

print(
    "SUPPORTED edges:",
    int(
        (
            edges["evidence"]
            == "SUPPORTED"
        ).sum()
    ),
)

print(
    "SPARSE edges:",
    int(
        (
            edges["evidence"]
            == "SPARSE"
        ).sum()
    ),
)

print(
    "Candidate graph edges:",
    len(adjacency),
)


all_places = sorted(
    set(
        edges["place_a"]
    )
    |
    set(
        edges["place_b"]
    )
)

connected_places = sorted(
    set(
        adjacency["place_a"]
    )
    |
    set(
        adjacency["place_b"]
    )
)

isolated = sorted(
    set(all_places)
    - set(connected_places)
)


print()
print(
    "Places observed in transitions:",
    len(all_places),
)

print(
    "Places represented in candidate graph:",
    len(connected_places),
)

print(
    "Candidate isolated places:",
    isolated,
)


print()
print("=" * 100)
print("INTERPRETATION RULE")
print("=" * 100)

print(
    "ROBUST/SUPPORTED means observed trajectory adjacency."
)

print(
    "It does NOT mean the two places should be merged."
)

print(
    "Classifier confusion was NOT used to construct this graph."
)

print()
print(
    "NEXT_ACTION="
    "CROSSCHECK_CONFUSIONS_WITH_TRANSITION_GRAPH"
)

print("=" * 100)

print()
print("Artifacts:")
print(" ", DIRECTED_OUTPUT)
print(" ", EDGE_OUTPUT)
print(" ", ADJACENCY_OUTPUT)
