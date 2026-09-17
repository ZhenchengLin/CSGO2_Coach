from __future__ import annotations

import hashlib
import json
from pathlib import Path


CANDIDATE = Path(
    "docs/v3_macro_zone_candidate_v1.json"
)

SEMANTIC_FREEZE = Path(
    "docs/v3_semantic_seed_freeze.json"
)

SPATIAL_EVIDENCE = Path(
    "docs/v3_gate1_spatial_semantic_evidence.json"
)

STRUCTURE_AUDIT = Path(
    "docs/v3_macro_zone_candidate_v1_structure_audit.json"
)

EVALUATION = Path(
    "data/interim/v3_macro_zone_candidate_v1_evaluation.csv"
)

SPATIAL_SUMMARY = Path(
    "data/interim/v3_macro_zone_candidate_v1_spatial_summary.csv"
)

VISUAL_DIR = Path(
    "artifacts/v3_macro_zone_candidate_v1_visual_audit"
)

VISUAL_IMAGES = [
    VISUAL_DIR / "v3_macro_zones_xy.png",
    VISUAL_DIR / "v3_macro_zones_xz.png",
    VISUAL_DIR / "v3_macro_zones_yz.png",
]

OUTPUT = Path(
    "docs/v3_macro_zone_mapping_v1_frozen.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


required = [
    CANDIDATE,
    SEMANTIC_FREEZE,
    SPATIAL_EVIDENCE,
    STRUCTURE_AUDIT,
    EVALUATION,
    SPATIAL_SUMMARY,
    *VISUAL_IMAGES,
]

for path in required:
    require(
        path.exists(),
        f"Missing Gate 1 evidence: {path}",
    )

require(
    not OUTPUT.exists(),
    f"Refusing to overwrite frozen mapping: {OUTPUT}",
)


candidate = json.loads(
    CANDIDATE.read_text()
)

semantic = json.loads(
    SEMANTIC_FREEZE.read_text()
)

structure = json.loads(
    STRUCTURE_AUDIT.read_text()
)


require(
    candidate["status"]
    == "CANDIDATE_NOT_FROZEN",
    "Unexpected candidate status.",
)

require(
    structure["status"]
    == "STRUCTURALLY_VIABLE",
    "Structural audit did not pass.",
)

require(
    structure["macro_graph_connected"]
    is True,
    "Macro graph is not connected.",
)

require(
    structure["isolated_zones"]
    == [],
    "Macro graph contains isolated zones.",
)

require(
    structure[
        "all_multi_place_zones_connected"
    ]
    is True,
    "A multi-place zone is disconnected.",
)

require(
    structure[
        "all_preserved_distinctions_supported"
    ]
    is True,
    "A preserved route distinction lacks transition evidence.",
)


zones = candidate["zones"]

all_places = sorted(
    place
    for places in zones.values()
    for place in places
)

require(
    len(zones) == 15,
    f"Expected 15 macro-zones, found {len(zones)}",
)

require(
    len(all_places) == 23,
    f"Expected 23 semantic places, found {len(all_places)}",
)

require(
    len(set(all_places)) == 23,
    "Duplicate fine place in macro mapping.",
)

require(
    sorted(
        semantic["places"]
    )
    == all_places,
    (
        "Macro mapping does not exactly cover "
        "the frozen semantic seed."
    ),
)


evidence_hashes = {
    str(path):
        sha256_file(path)
    for path in required
}


record = {
    "version":
        "V3",

    "artifact":
        "Mirage tactical macro-zone mapping",

    "mapping_version":
        "v1",

    "status":
        "FROZEN",

    "frozen_date":
        "2026-09-15",

    "map":
        "de_mirage",

    "n_fine_places":
        23,

    "n_macro_zones":
        15,

    "zones":
        zones,

    "preserved_route_distinctions":
        candidate[
            "preserved_route_distinctions"
        ],

    "manual_visual_review": {
        "status":
            "PASS",

        "views_reviewed": [
            "XY",
            "XZ",
            "YZ",
        ],

        "finding":
            (
                "No obvious unrelated-region bridge or "
                "structurally invalid macro-zone was found. "
                "Important route and vertical distinctions "
                "remain represented."
            ),
    },

    "scientific_boundary": {
        "v2_d_confirm_used":
            False,

        "mapping_may_be_changed_during_gate2":
            False,

        "revision_policy":
            (
                "Any future map-representation change must "
                "create a new mapping version. Frozen v1 "
                "must not be silently mutated."
            ),
    },

    "evidence_sha256":
        evidence_hashes,
}


OUTPUT.write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n"
)


print("=" * 92)
print("V3 GATE 1 — MACRO-ZONE MAPPING FREEZE")
print("=" * 92)

print()
print("Fine places:", 23)
print("Macro-zones:", 15)
print("Structural audit: PASS")
print("Visual XY/XZ/YZ review: PASS")
print("V2 D_CONFIRM used: False")

print()
print("✅ V3 GATE 1 CLOSED")
print("✅ MACRO-ZONE MAPPING V1 FROZEN")
print("NEXT_ACTION=GATE_2_TARGET_SOURCE_AUDIT")

print()
print("Frozen artifact:")
print(" ", OUTPUT)
