from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PLACE_AUDIT = Path(
    "data/interim/v3_place_audit_by_place.csv"
)

ACTIVE_AUDIT = Path(
    "data/interim/v3_active_place_coverage_all_reserves.csv"
)

OUTPUT = Path(
    "docs/v3_semantic_seed_freeze.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(
    PLACE_AUDIT.exists(),
    f"Missing {PLACE_AUDIT}",
)

require(
    ACTIVE_AUDIT.exists(),
    f"Missing {ACTIVE_AUDIT}",
)

require(
    not OUTPUT.exists(),
    f"Refusing to overwrite {OUTPUT}",
)


places = pd.read_csv(
    PLACE_AUDIT
)

active = pd.read_csv(
    ACTIVE_AUDIT
)


# --------------------------------------------------
# Freeze checks
# --------------------------------------------------

require(
    len(active) == 13,
    f"Expected 13 reserve demos, got {len(active)}",
)

require(
    len(places) == 23,
    f"Expected 23 places, got {len(places)}",
)

require(
    (places["demos_present"] == 13).all(),
    "Not every place appears in every reserve demo.",
)

minimum_coverage = float(
    active["coverage"].min()
)

weighted_coverage = float(
    active["labelled_active_rows"].sum()
    / active["active_rows"].sum()
)

require(
    minimum_coverage >= 0.99,
    (
        "Active semantic coverage below freeze threshold: "
        f"{minimum_coverage:.6%}"
    ),
)


place_names = sorted(
    places["place"]
    .astype(str)
    .tolist()
)


record = {
    "version":
        "V3",

    "role":
        "Mirage semantic seed freeze",

    "status":
        "FROZEN",

    "semantic_source":
        "Valve player last_place_name / m_szLastPlaceName",

    "scope": {
        "map":
            "de_mirage",

        "state":
            "active-round alive competitive players",

        "design_corpus":
            "13 V2 reserve demos",

        "v2_d_confirm_used":
            False,
    },

    "evidence": {
        "n_reserve_demos":
            13,

        "n_places":
            23,

        "total_active_rows":
            int(
                active[
                    "active_rows"
                ].sum()
            ),

        "weighted_active_coverage":
            weighted_coverage,

        "minimum_demo_active_coverage":
            minimum_coverage,

        "place_audit_sha256":
            sha256_file(
                PLACE_AUDIT
            ),

        "active_audit_sha256":
            sha256_file(
                ACTIVE_AUDIT
            ),
    },

    "known_exception": {
        "demo":
            (
                "2026-09-12_iowa-stormboar_"
                "vs_sportsbetexpert_mirage.dem"
            ),

        "missing_active_rows":
            440,

        "missing_players":
            1,

        "scientific_treatment":
            (
                "Recorded as a small source-level missingness "
                "exception. No synthetic place labels are created."
            ),
    },

    "places":
        place_names,

    "contract": [
        (
            "Valve last_place_name is accepted as a "
            "fine-grained semantic seed."
        ),
        (
            "It is not yet the final V3 prediction class."
        ),
        (
            "V3 D_CONFIRM from V2 remains excluded from "
            "representation design."
        ),
        (
            "Macro-zone merging requires a separate "
            "Gate 1 freeze."
        ),
        (
            "Missing place labels are never fabricated."
        ),
    ],
}


OUTPUT.write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n"
)


print("=" * 88)
print("V3 GATE 0 — SEMANTIC SEED FREEZE")
print("=" * 88)

print()
print(
    "Reserve demos:",
    record["evidence"]["n_reserve_demos"],
)

print(
    "Places:",
    record["evidence"]["n_places"],
)

print(
    "Active rows:",
    f"{record['evidence']['total_active_rows']:,}",
)

print(
    "Weighted coverage:",
    f"{weighted_coverage:.6%}",
)

print(
    "Minimum demo coverage:",
    f"{minimum_coverage:.6%}",
)

print()
print("Frozen places:")

for place in place_names:
    print(" ", place)

print()
print("✅ V3 GATE 0 FROZEN")
print("NEXT_ACTION=AUDIT_XYZ_TO_PLACE_RESOLUTION")
print()
print("Record:")
print(" ", OUTPUT)
