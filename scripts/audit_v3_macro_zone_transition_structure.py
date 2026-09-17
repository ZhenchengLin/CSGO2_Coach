from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


MAPPING_PATH = Path(
    "docs/v3_macro_zone_candidate_v1.json"
)

EDGE_PATH = Path(
    "data/interim/v3_place_transition_edges.csv"
)

DIRECTED_PATH = Path(
    "data/interim/v3_place_transitions_directed.csv"
)

MACRO_EDGE_OUTPUT = Path(
    "data/interim/v3_macro_zone_transition_edges.csv"
)

AUDIT_OUTPUT = Path(
    "docs/v3_macro_zone_candidate_v1_structure_audit.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def connected_component(
    nodes,
    adjacency,
):
    nodes = set(nodes)

    if not nodes:
        return set()

    start = next(
        iter(nodes)
    )

    visited = {
        start
    }

    queue = deque([
        start
    ])

    while queue:

        current = (
            queue.popleft()
        )

        for neighbor in adjacency.get(
            current,
            set(),
        ):

            if (
                neighbor in nodes
                and neighbor not in visited
            ):

                visited.add(
                    neighbor
                )

                queue.append(
                    neighbor
                )

    return visited


# ============================================================
# Load
# ============================================================

mapping_record = json.loads(
    MAPPING_PATH.read_text()
)

edges = pd.read_csv(
    EDGE_PATH
)

directed = pd.read_csv(
    DIRECTED_PATH
)


zones = mapping_record[
    "zones"
]


# ============================================================
# Invert macro mapping
# ============================================================

place_to_zone = {}

for zone, places in zones.items():

    for place in places:

        require(
            place not in place_to_zone,
            f"Duplicate place mapping: {place}",
        )

        place_to_zone[
            place
        ] = zone


observed_places = (
    set(
        edges["place_a"]
    )
    |
    set(
        edges["place_b"]
    )
)


require(
    set(place_to_zone)
    == observed_places,
    (
        "Macro mapping does not exactly match "
        "trajectory-observed semantic places."
    ),
)


# ============================================================
# Strong fine-grained graph
# ============================================================

strong_edges = edges[
    edges["evidence"].isin([
        "ROBUST",
        "SUPPORTED",
    ])
].copy()


fine_adjacency = defaultdict(set)

for row in strong_edges.itertuples():

    fine_adjacency[
        row.place_a
    ].add(
        row.place_b
    )

    fine_adjacency[
        row.place_b
    ].add(
        row.place_a
    )


# ============================================================
# Audit internal connectivity of every macro-zone
# ============================================================

zone_rows = []

all_multi_connected = True

for zone, places in zones.items():

    places = list(
        places
    )

    if len(places) == 1:

        connected = True
        internal_edges = 0

    else:

        visited = connected_component(
            places,
            fine_adjacency,
        )

        connected = (
            len(visited)
            == len(places)
        )

        internal_edges = int(
            strong_edges.apply(
                lambda row: (
                    row["place_a"]
                    in places
                    and
                    row["place_b"]
                    in places
                ),
                axis=1,
            ).sum()
        )

    if not connected:
        all_multi_connected = False

    zone_rows.append({
        "zone":
            zone,

        "n_places":
            len(places),

        "places":
            places,

        "internally_connected":
            connected,

        "n_strong_internal_edges":
            internal_edges,
    })


# ============================================================
# Aggregate real transitions into macro transitions
# ============================================================

macro_directed_counts = defaultdict(int)

total_fine_transitions = 0
internalized_transitions = 0


for row in directed.itertuples():

    source_zone = (
        place_to_zone[
            str(
                row.source_place
            )
        ]
    )

    target_zone = (
        place_to_zone[
            str(
                row.target_place
            )
        ]
    )

    count = int(
        row.n_transitions
    )

    total_fine_transitions += (
        count
    )

    if (
        source_zone
        == target_zone
    ):

        internalized_transitions += (
            count
        )

        continue

    macro_directed_counts[
        (
            source_zone,
            target_zone,
        )
    ] += count


# ============================================================
# Undirected macro graph
# ============================================================

macro_undirected = defaultdict(
    lambda: {
        "n_transitions": 0,
        "n_fine_edges": 0,
        "max_fine_demos_present": 0,
        "robust_fine_edges": 0,
        "supported_fine_edges": 0,
    }
)


for row in edges.itertuples():

    zone_a = (
        place_to_zone[
            str(
                row.place_a
            )
        ]
    )

    zone_b = (
        place_to_zone[
            str(
                row.place_b
            )
        ]
    )

    if zone_a == zone_b:
        continue

    a, b = sorted([
        zone_a,
        zone_b,
    ])

    key = (
        a,
        b,
    )

    item = (
        macro_undirected[
            key
        ]
    )

    item[
        "n_transitions"
    ] += int(
        row.n_transitions
    )

    item[
        "n_fine_edges"
    ] += 1

    item[
        "max_fine_demos_present"
    ] = max(
        item[
            "max_fine_demos_present"
        ],
        int(
            row.demos_present
        ),
    )

    if row.evidence == "ROBUST":

        item[
            "robust_fine_edges"
        ] += 1

    elif row.evidence == "SUPPORTED":

        item[
            "supported_fine_edges"
        ] += 1


macro_edge_rows = []

macro_adjacency = defaultdict(set)

for (
    zone_a,
    zone_b
), item in macro_undirected.items():

    has_strong_edge = (
        item[
            "robust_fine_edges"
        ]
        +
        item[
            "supported_fine_edges"
        ]
        > 0
    )

    macro_edge_rows.append({
        "zone_a":
            zone_a,

        "zone_b":
            zone_b,

        **item,

        "has_strong_fine_edge":
            has_strong_edge,
    })

    if has_strong_edge:

        macro_adjacency[
            zone_a
        ].add(
            zone_b
        )

        macro_adjacency[
            zone_b
        ].add(
            zone_a
        )


macro_edges = (
    pd.DataFrame(
        macro_edge_rows
    )
    .sort_values(
        "n_transitions",
        ascending=False,
    )
)

MACRO_EDGE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

macro_edges.to_csv(
    MACRO_EDGE_OUTPUT,
    index=False,
)


# ============================================================
# Whole macro graph connectivity
# ============================================================

all_zones = set(
    zones
)

macro_component = connected_component(
    all_zones,
    macro_adjacency,
)

macro_graph_connected = (
    macro_component
    == all_zones
)

isolated_zones = sorted(
    zone
    for zone in all_zones
    if not macro_adjacency.get(
        zone
    )
)


# ============================================================
# Check preserved route distinctions
# ============================================================

preserved_rows = []

all_preserved_supported = True


for distinction in mapping_record[
    "preserved_route_distinctions"
]:

    parts = [
        item.strip()
        for item
        in distinction.split(
            " vs "
        )
    ]

    require(
        len(parts) == 2,
        (
            "Could not parse preserved distinction: "
            f"{distinction}"
        ),
    )

    zone_a, zone_b = parts

    require(
        zone_a in zones,
        f"Unknown macro zone: {zone_a}",
    )

    require(
        zone_b in zones,
        f"Unknown macro zone: {zone_b}",
    )

    fine_places_a = set(
        zones[
            zone_a
        ]
    )

    fine_places_b = set(
        zones[
            zone_b
        ]
    )

    supporting = strong_edges[
        (
            strong_edges[
                "place_a"
            ].isin(
                fine_places_a
            )
            &
            strong_edges[
                "place_b"
            ].isin(
                fine_places_b
            )
        )
        |
        (
            strong_edges[
                "place_a"
            ].isin(
                fine_places_b
            )
            &
            strong_edges[
                "place_b"
            ].isin(
                fine_places_a
            )
        )
    ]

    supported = (
        len(
            supporting
        )
        > 0
    )

    if not supported:
        all_preserved_supported = False

    preserved_rows.append({
        "distinction":
            distinction,

        "supported_by_transition":
            supported,

        "n_supporting_fine_edges":
            len(
                supporting
            ),

        "supporting_edges": [
            (
                f"{row.place_a}"
                f"<->{row.place_b}"
                f":{row.evidence}"
            )
            for row
            in supporting.itertuples()
        ],
    })


# ============================================================
# Macro directed transitions
# ============================================================

macro_directed_rows = [
    {
        "source_zone":
            source,

        "target_zone":
            target,

        "n_transitions":
            count,
    }
    for (
        source,
        target
    ), count
    in macro_directed_counts.items()
]

macro_directed = (
    pd.DataFrame(
        macro_directed_rows
    )
    .sort_values(
        "n_transitions",
        ascending=False,
    )
)


# ============================================================
# Structural verdict
# ============================================================

all_zones_represented = (
    set(
        macro_edges["zone_a"]
    )
    |
    set(
        macro_edges["zone_b"]
    )
    ==
    all_zones
)


structural_pass = all([
    all_multi_connected,
    macro_graph_connected,
    all_zones_represented,
    len(
        isolated_zones
    ) == 0,
    all_preserved_supported,
])


# ============================================================
# Save audit
# ============================================================

audit = {
    "version":
        "V3",

    "candidate":
        "macro-zone-v1",

    "status":
        (
            "STRUCTURALLY_VIABLE"
            if structural_pass
            else "STRUCTURAL_REVIEW_REQUIRED"
        ),

    "mapping_frozen":
        False,

    "n_macro_zones":
        len(
            zones
        ),

    "all_multi_place_zones_connected":
        all_multi_connected,

    "macro_graph_connected":
        macro_graph_connected,

    "all_zones_represented":
        all_zones_represented,

    "isolated_zones":
        isolated_zones,

    "all_preserved_distinctions_supported":
        all_preserved_supported,

    "total_fine_semantic_transitions":
        total_fine_transitions,

    "transitions_internalized_by_abstraction":
        internalized_transitions,

    "internalized_transition_fraction":
        (
            internalized_transitions
            / total_fine_transitions
        ),

    "zones":
        zone_rows,

    "preserved_route_distinctions":
        preserved_rows,

    "guardrails": [
        (
            "Structural viability does not freeze "
            "the candidate mapping."
        ),
        (
            "High transition count does not imply "
            "two macro-zones should be merged."
        ),
        (
            "The final freeze still requires "
            "visual/spatial audit."
        ),
        (
            "V2 D_CONFIRM remains excluded."
        ),
    ],
}


AUDIT_OUTPUT.write_text(
    json.dumps(
        audit,
        indent=2,
    )
    + "\n"
)


# ============================================================
# Report
# ============================================================

print("=" * 100)
print("V3 GATE 1G — MACRO-ZONE STRUCTURAL AUDIT")
print("=" * 100)

print()
print("MULTI-PLACE ZONES")
print("-" * 100)

for item in zone_rows:

    if item[
        "n_places"
    ] <= 1:
        continue

    print(
        f"{item['zone']:<20}"
        f" places={item['n_places']}"
        f" | connected="
        f"{item['internally_connected']}"
        f" | strong internal edges="
        f"{item['n_strong_internal_edges']}"
    )

    print(
        "   ",
        ", ".join(
            item[
                "places"
            ]
        ),
    )


print()
print("PRESERVED ROUTE DISTINCTIONS")
print("-" * 100)

for item in preserved_rows:

    print(
        f"{item['distinction']:<30}"
        f" supported="
        f"{item['supported_by_transition']}"
        f" | fine edges="
        f"{item['n_supporting_fine_edges']}"
    )


print()
print("TOP MACRO TRANSITIONS")
print("-" * 100)

print(
    macro_directed.head(30)
    .to_string(
        index=False
    )
)


print()
print("GRAPH SUMMARY")
print("-" * 100)

print(
    "Macro zones:",
    len(zones),
)

print(
    "Macro edges:",
    len(
        macro_edges
    ),
)

print(
    "Macro graph connected:",
    macro_graph_connected,
)

print(
    "Isolated zones:",
    isolated_zones,
)

print(
    "All multi-place zones internally connected:",
    all_multi_connected,
)

print(
    "All preserved distinctions supported:",
    all_preserved_supported,
)

print()

print(
    "Fine semantic transitions:",
    f"{total_fine_transitions:,}",
)

print(
    "Transitions internalized:",
    f"{internalized_transitions:,}",
)

print(
    "Internalized fraction:",
    f"{internalized_transitions / total_fine_transitions:.2%}",
)


print()
print("=" * 100)

if structural_pass:

    print(
        "✅ MACRO-ZONE CANDIDATE V1 "
        "IS STRUCTURALLY VIABLE"
    )

    print(
        "NEXT_ACTION=VISUAL_SPATIAL_AUDIT"
    )

else:

    print(
        "⚠️ MACRO-ZONE CANDIDATE V1 "
        "REQUIRES STRUCTURAL REVIEW"
    )

    print(
        "NEXT_ACTION=INVESTIGATE_FAILED_CONTRACTS"
    )

print("=" * 100)

print()
print("Artifacts:")
print(" ", MACRO_EDGE_OUTPUT)
print(" ", AUDIT_OUTPUT)
