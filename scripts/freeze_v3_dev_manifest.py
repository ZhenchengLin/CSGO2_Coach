from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
)


# ============================================================
# Inputs
# ============================================================

RAW_ROOT = Path(
    "data/raw"
)

V0_MANIFEST = Path(
    "data/raw/demo_manifest.csv"
)

V2_INCOMING = Path(
    "data/raw/v2_incoming"
)

V2_CONFIRM_MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

V3_INCOMING = Path(
    "data/raw/v3_incoming"
)

V3_QUEUE = Path(
    "docs/v3_fresh_acquisition_queue_v2.csv"
)

V3_PROTOCOL = Path(
    "docs/v3_dev_acquisition_protocol_v2.json"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

OUTPUT_IDENTITY = Path(
    "docs/v3_dev_manifest_identity.json"
)


EXPECTED_V0 = 20
EXPECTED_V2_RESERVE = 13
EXPECTED_FRESH = 20
EXPECTED_TOTAL = 53


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


def sha256_text(path: Path) -> str:
    return sha256_file(path)


def find_demo_by_name(
    filename: str,
):
    matches = [
        path
        for path in RAW_ROOT.rglob(
            filename
        )
        if path.is_file()
    ]

    require(
        len(matches) == 1,
        (
            f"Expected exactly one demo named "
            f"{filename}, found {len(matches)}: "
            f"{matches}"
        ),
    )

    return matches[0]


def filename_date(
    filename: str,
):
    if (
        len(filename) >= 10
        and filename[4] == "-"
        and filename[7] == "-"
    ):
        return filename[:10]

    return None


# ============================================================
# Required inputs
# ============================================================

for path in [
    V0_MANIFEST,
    V2_CONFIRM_MANIFEST,
    V3_QUEUE,
    V3_PROTOCOL,
]:

    require(
        path.exists(),
        f"Missing required input: {path}",
    )


require(
    V2_INCOMING.exists(),
    f"Missing directory: {V2_INCOMING}",
)

require(
    V3_INCOMING.exists(),
    f"Missing directory: {V3_INCOMING}",
)


protocol = json.loads(
    V3_PROTOCOL.read_text()
)

require(
    protocol[
        "status"
    ] == "FROZEN",
    "V3 acquisition protocol v2 is not frozen.",
)

require(
    protocol[
        "development_design"
    ][
        "historical_v0_development_matches"
    ]
    == EXPECTED_V0,
    "Protocol V0 count mismatch.",
)

require(
    protocol[
        "development_design"
    ][
        "historical_v2_reserve_matches"
    ]
    == EXPECTED_V2_RESERVE,
    "Protocol V2 reserve count mismatch.",
)

require(
    protocol[
        "development_design"
    ][
        "fresh_mirage_target"
    ]
    == EXPECTED_FRESH,
    "Protocol fresh-demo count mismatch.",
)


# ============================================================
# Build global filename registry
# ============================================================

all_raw_demos = sorted(
    RAW_ROOT.rglob(
        "*.dem"
    )
)

print("=" * 104)
print("V3 GATE 3B — FREEZE D_V3_DEV MANIFEST")
print("=" * 104)

print()
print(
    "Raw .dem files visible:",
    len(
        all_raw_demos
    ),
)


# ============================================================
# Group 1:
# Original V0 development corpus
# ============================================================

v0 = pl.read_csv(
    V0_MANIFEST,
    infer_schema_length=None,
)


require(
    v0.height == EXPECTED_V0,
    (
        f"Expected {EXPECTED_V0} V0 demos, "
        f"found {v0.height}."
    ),
)

require(
    set(
        v0[
            "map"
        ].to_list()
    )
    == {
        "de_mirage"
    },
    "V0 manifest contains non-Mirage rows.",
)

require(
    set(
        v0[
            "parse_status"
        ].to_list()
    )
    == {
        "parsed_ok"
    },
    "V0 manifest contains non-passing demos.",
)


v0_records = []


for row in v0.iter_rows(
    named=True
):

    filename = str(
        row[
            "demo filename"
        ]
    )

    path = find_demo_by_name(
        filename
    )

    v0_records.append({
        "filename":
            filename,

        "path":
            str(
                path
            ),

        "match_date":
            str(
                row[
                    "date"
                ]
            ),

        "dev_source_role":
            "V0_DEVELOPMENT",

        "prior_scientific_role":
            "V0/V1 development",

        "expected_map":
            "de_mirage",
    })


require(
    len(v0_records)
    == EXPECTED_V0,
    "V0 record count mismatch.",
)


# ============================================================
# Group 2:
# V2 reserves only
#
# V2 D_CONFIRM remains sealed.
# ============================================================

confirm = pl.read_csv(
    V2_CONFIRM_MANIFEST,
    infer_schema_length=None,
)


require(
    confirm.height == 30,
    (
        "Expected exactly 30 frozen V2 "
        f"confirmation demos, found {confirm.height}."
    ),
)


confirm_names = set(
    confirm[
        "demo_filename"
    ].to_list()
)

confirm_sha = set(
    confirm[
        "sha256"
    ].to_list()
)


v2_all = sorted(
    V2_INCOMING.glob(
        "*.dem"
    )
)


v2_reserve_paths = [
    path
    for path in v2_all
    if path.name
    not in confirm_names
]


require(
    len(v2_reserve_paths)
    == EXPECTED_V2_RESERVE,
    (
        f"Expected {EXPECTED_V2_RESERVE} V2 reserves, "
        f"found {len(v2_reserve_paths)}."
    ),
)


v2_reserve_records = []


for path in v2_reserve_paths:

    date = filename_date(
        path.name
    )

    require(
        date is not None,
        (
            "Could not infer V2 reserve date "
            f"from filename: {path.name}"
        ),
    )

    v2_reserve_records.append({
        "filename":
            path.name,

        "path":
            str(
                path
            ),

        "match_date":
            date,

        "dev_source_role":
            "V2_RESERVE",

        "prior_scientific_role":
            "Gate 0-2 design/audit evidence",

        "expected_map":
            "de_mirage",
    })


# ============================================================
# Group 3:
# Fresh V3 demos
# ============================================================

with V3_QUEUE.open(
    newline="",
) as f:

    queue = list(
        csv.DictReader(
            f
        )
    )


require(
    len(queue)
    == EXPECTED_FRESH,
    (
        f"Expected {EXPECTED_FRESH} queue rows, "
        f"found {len(queue)}."
    ),
)


expected_fresh_names = {
    (
        f"{row['date']}_"
        f"{row['team1']}_vs_"
        f"{row['team2']}_mirage.dem"
    )
    for row in queue
}


fresh_paths = sorted(
    V3_INCOMING.glob(
        "*.dem"
    )
)


fresh_names = {
    path.name
    for path in fresh_paths
}


require(
    len(fresh_paths)
    == EXPECTED_FRESH,
    (
        f"Expected exactly {EXPECTED_FRESH} fresh demos, "
        f"found {len(fresh_paths)}."
    ),
)


require(
    fresh_names
    == expected_fresh_names,
    (
        "Fresh V3 directory does not exactly match "
        "the frozen 20-match queue.\n"
        f"Missing: {sorted(expected_fresh_names - fresh_names)}\n"
        f"Unexpected: {sorted(fresh_names - expected_fresh_names)}"
    ),
)


queue_by_filename = {}

for row in queue:

    filename = (
        f"{row['date']}_"
        f"{row['team1']}_vs_"
        f"{row['team2']}_mirage.dem"
    )

    queue_by_filename[
        filename
    ] = row


fresh_records = []


for path in fresh_paths:

    q = queue_by_filename[
        path.name
    ]

    fresh_records.append({
        "filename":
            path.name,

        "path":
            str(
                path
            ),

        "match_date":
            q[
                "date"
            ],

        "dev_source_role":
            "V3_FRESH",

        "prior_scientific_role":
            "fresh development acquisition",

        "expected_map":
            "de_mirage",
    })


# ============================================================
# Combined 53-match candidate manifest
# ============================================================

records = (
    v0_records
    + v2_reserve_records
    + fresh_records
)


require(
    len(records)
    == EXPECTED_TOTAL,
    (
        f"Expected {EXPECTED_TOTAL} total development "
        f"demos, found {len(records)}."
    ),
)


filenames = [
    row[
        "filename"
    ]
    for row in records
]


require(
    len(filenames)
    == len(
        set(
            filenames
        )
    ),
    "Duplicate filename in D_V3_DEV.",
)


# ============================================================
# Validate every physical demo
#
# One at a time:
# low RAM usage.
# ============================================================

print()
print("VALIDATING 53 PHYSICAL DEMOS")
print("-" * 104)


validated = []


for index, row in enumerate(
    records,
    start=1,
):

    path = Path(
        row[
            "path"
        ]
    )


    require(
        path.exists(),
        f"Demo missing: {path}",
    )


    digest = sha256_file(
        path
    )


    # --------------------------------------------------------
    # Absolute protection for V2 D_CONFIRM
    # --------------------------------------------------------

    require(
        row[
            "filename"
        ]
        not in confirm_names,
        (
            "V2 D_CONFIRM filename leaked into D_V3_DEV: "
            f"{row['filename']}"
        ),
    )


    require(
        digest
        not in confirm_sha,
        (
            "V2 D_CONFIRM SHA leaked into D_V3_DEV: "
            f"{row['filename']}"
        ),
    )


    # --------------------------------------------------------
    # Header-level parser validation
    # --------------------------------------------------------

    demo = open_v0_demo(
        path,
        verbose=False,
    )


    map_name = demo.header.get(
        "map_name"
    )


    require(
        map_name
        == "de_mirage",
        (
            f"Non-Mirage demo detected: "
            f"{row['filename']} -> {map_name}"
        ),
    )


    require(
        demo.tickrate
        == V0_DEMO_TICKS_PER_SECOND,
        (
            f"Unexpected tickrate: "
            f"{row['filename']} -> {demo.tickrate}"
        ),
    )


    validated.append({
        **row,

        "sha256":
            digest,

        "map_name":
            map_name,

        "raw_ticks_per_sec":
            int(
                demo.tickrate
            ),

        "validation_status":
            "PASS",
    })


    print(
        f"[{index:02d}/53]",
        f"{row['dev_source_role']:<16}",
        row[
            "filename"
        ],
        "✅",
    )


# ============================================================
# Cross-corpus SHA uniqueness
# ============================================================

sha_values = [
    row[
        "sha256"
    ]
    for row in validated
]


require(
    len(
        sha_values
    )
    == len(
        set(
            sha_values
        )
    ),
    (
        "Duplicate physical demo SHA detected "
        "inside D_V3_DEV."
    ),
)


# ============================================================
# Freeze ordering
#
# Chronological order first.
# Role is NOT used to shuffle matches.
# ============================================================

validated.sort(
    key=lambda row: (
        row[
            "match_date"
        ],
        row[
            "filename"
        ],
    )
)


for rank, row in enumerate(
    validated,
    start=1,
):

    row[
        "dev_rank"
    ] = rank


# ============================================================
# Write manifest to temporary content first
# ============================================================

fieldnames = [
    "dev_rank",
    "demo_filename",
    "match_date",
    "dev_source_role",
    "prior_scientific_role",
    "path",
    "sha256",
    "map_name",
    "raw_ticks_per_sec",
    "validation_status",
]


manifest_rows = []


for row in validated:

    manifest_rows.append({
        "dev_rank":
            row[
                "dev_rank"
            ],

        "demo_filename":
            row[
                "filename"
            ],

        "match_date":
            row[
                "match_date"
            ],

        "dev_source_role":
            row[
                "dev_source_role"
            ],

        "prior_scientific_role":
            row[
                "prior_scientific_role"
            ],

        "path":
            row[
                "path"
            ],

        "sha256":
            row[
                "sha256"
            ],

        "map_name":
            row[
                "map_name"
            ],

        "raw_ticks_per_sec":
            row[
                "raw_ticks_per_sec"
            ],

        "validation_status":
            row[
                "validation_status"
            ],
    })


# ------------------------------------------------------------
# Prevent silent mutation if already frozen
# ------------------------------------------------------------

if OUTPUT_MANIFEST.exists():

    existing = pl.read_csv(
        OUTPUT_MANIFEST,
        infer_schema_length=None,
    )

    proposed = pl.DataFrame(
        manifest_rows,
        infer_schema_length=None,
    )


    same = (
        existing.equals(
            proposed
        )
    )


    require(
        same,
        (
            "Existing frozen V3 development manifest "
            "differs from current proposed manifest. "
            "Do not overwrite silently."
        ),
    )


    print()
    print(
        "✅ Existing D_V3_DEV manifest verified"
    )


else:

    with OUTPUT_MANIFEST.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


    print()
    print(
        "✅ D_V3_DEV manifest written"
    )


# ============================================================
# Re-read frozen file and establish identity
# ============================================================

frozen_manifest = pl.read_csv(
    OUTPUT_MANIFEST,
    infer_schema_length=None,
)


require(
    frozen_manifest.height
    == EXPECTED_TOTAL,
    "Frozen manifest does not contain 53 rows.",
)


role_counts = (
    frozen_manifest
    .group_by(
        "dev_source_role"
    )
    .len()
    .sort(
        "dev_source_role"
    )
)


role_count_dict = {
    row[
        "dev_source_role"
    ]:
        int(
            row[
                "len"
            ]
        )
    for row in role_counts.iter_rows(
        named=True
    )
}


require(
    role_count_dict
    == {
        "V0_DEVELOPMENT":
            20,

        "V2_RESERVE":
            13,

        "V3_FRESH":
            20,
    },
    (
        "Unexpected source-role counts: "
        f"{role_count_dict}"
    ),
)


manifest_sha = sha256_text(
    OUTPUT_MANIFEST
)


identity = {
    "version":
        "V3",

    "dataset":
        "D_V3_DEV",

    "status":
        "FROZEN",

    "manifest":
        str(
            OUTPUT_MANIFEST
        ),

    "manifest_sha256":
        manifest_sha,

    "n_matches":
        EXPECTED_TOTAL,

    "role_counts":
        role_count_dict,

    "map":
        "de_mirage",

    "raw_ticks_per_sec":
        V0_DEMO_TICKS_PER_SECOND,

    "v2_d_confirm_included":
        False,

    "sha_unique_within_dev":
        True,

    "historical_development_reuse":
        {
            "V0_development":
                20,

            "V2_reserves":
                13,

            "reason":
                (
                    "Storage-aware protocol v2 permits "
                    "development-eligible historical data "
                    "to participate in V3 model development."
                ),
        },

    "fresh_matches":
        20,

    "scientific_boundary":
        (
            "D_V3_DEV is development-only. "
            "It may be used for grouped CV, baseline fitting, "
            "feature design, model selection, calibration, "
            "and hyperparameter selection. "
            "It must not be presented as untouched "
            "V3 confirmation."
        ),

    "future_confirmation":
        (
            "D_V3_CONFIRM must be separately acquired "
            "from a later untouched chronological period "
            "after V3 development decisions are frozen."
        ),
}


if OUTPUT_IDENTITY.exists():

    existing_identity = json.loads(
        OUTPUT_IDENTITY.read_text()
    )

    require(
        existing_identity
        == identity,
        (
            "Existing D_V3_DEV identity differs "
            "from the current frozen identity."
        ),
    )

    print(
        "✅ Existing manifest identity verified"
    )


else:

    OUTPUT_IDENTITY.write_text(
        json.dumps(
            identity,
            indent=2,
        )
        + "\n"
    )

    print(
        "✅ D_V3_DEV identity frozen"
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 104)
print("D_V3_DEV FREEZE SUMMARY")
print("=" * 104)

print(
    "Matches:",
    frozen_manifest.height,
)

print(
    "V0 development:",
    role_count_dict[
        "V0_DEVELOPMENT"
    ],
)

print(
    "V2 reserves:",
    role_count_dict[
        "V2_RESERVE"
    ],
)

print(
    "Fresh V3:",
    role_count_dict[
        "V3_FRESH"
    ],
)

print(
    "V2 D_CONFIRM included:",
    False,
)

print(
    "All map headers:",
    "de_mirage",
)

print(
    "All raw tick contracts:",
    "64 ticks/sec",
)

print(
    "SHA duplicates:",
    0,
)

print(
    "Manifest SHA256:",
    manifest_sha,
)


print()
print(
    "Chronological range:"
)

print(
    " ",
    frozen_manifest[
        "match_date"
    ].min(),
    "→",
    frozen_manifest[
        "match_date"
    ].max(),
)


print()
print("=" * 104)

print(
    "✅ D_V3_DEV FROZEN"
)

print(
    "✅ 53 DEVELOPMENT MATCHES VERIFIED"
)

print(
    "NEXT_ACTION=BUILD_V3_DEV_TARGET_DATASET"
)

print("=" * 104)

print()
print("Artifacts:")
print(" ", OUTPUT_MANIFEST)
print(" ", OUTPUT_IDENTITY)
