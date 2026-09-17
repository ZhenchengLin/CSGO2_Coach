from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path


# ============================================================
# V3 development-corpus acquisition contract
# ============================================================

CANDIDATE_DIR = Path(
    "data/raw/v3_incoming"
)

RAW_ROOT = Path(
    "data/raw"
)

PROTOCOL_OUTPUT = Path(
    "docs/v3_dev_acquisition_protocol.json"
)

AUDIT_OUTPUT = Path(
    "data/interim/v3_dev_candidate_pool_audit.csv"
)


TARGET_DEV_MATCHES = 60

# Last Gate 0-2 design demo was 2026-09-12.
# D_V3_DEV must begin strictly after this date.
MIN_DATE_EXCLUSIVE = date(
    2026,
    9,
    12,
)

MAP_TOKEN = "mirage"

DATE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_"
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


def parse_filename_date(
    filename: str,
):
    match = DATE_PATTERN.match(
        filename
    )

    if match is None:
        return None

    try:
        return date.fromisoformat(
            match.group(1)
        )

    except ValueError:
        return None


# ============================================================
# Freeze acquisition rule BEFORE manifest selection
# ============================================================

protocol = {
    "version":
        "V3",

    "artifact":
        "D_V3_DEV acquisition protocol",

    "status":
        "FROZEN",

    "map":
        "de_mirage",

    "target_match_count":
        TARGET_DEV_MATCHES,

    "selection_order":
        [
            "match_date ascending",
            "filename ascending",
        ],

    "selection_rule":
        (
            "Select the first 60 eligible chronological "
            "Mirage demos after 2026-09-12."
        ),

    "minimum_date_exclusive":
        MIN_DATE_EXCLUSIVE.isoformat(),

    "historical_overlap_allowed":
        False,

    "within_candidate_duplicate_sha_allowed":
        False,

    "historical_data_boundary": {
        "v0_v1_v2_demos":
            "FORBIDDEN",

        "v2_d_confirm":
            "SEALED_AND_FORBIDDEN",

        "v2_13_reserves":
            (
                "Gate 0-2 design evidence only; "
                "FORBIDDEN for V3 model fitting."
            ),
    },

    "development_role":
        (
            "D_V3_DEV may be used for baseline fitting, "
            "feature development, model selection, "
            "hyperparameter selection, calibration, "
            "and grouped development evaluation."
        ),

    "confirmation_boundary":
        (
            "D_V3_CONFIRM must be collected later from "
            "a strictly newer chronological period and "
            "must remain untouched until V3 is frozen."
        ),

    "split_policy": {
        "unit":
            "match",

        "planned_group_cv_folds":
            5,

        "random_row_split_allowed":
            False,
    },

    "rationale":
        (
            "V3 is a 15-class forecasting task. "
            "A 60-match development corpus provides "
            "substantially more match-level diversity "
            "than the earlier 20-match V0 development "
            "corpus and supports approximately 12 held-out "
            "matches per fold in 5-fold grouped CV. "
            "This is a development-scale design choice, "
            "not a statistical power guarantee."
        ),
}


if PROTOCOL_OUTPUT.exists():

    existing = json.loads(
        PROTOCOL_OUTPUT.read_text()
    )

    require(
        existing == protocol,
        (
            "Existing acquisition protocol differs "
            "from the expected frozen contract. "
            "Do not silently overwrite it."
        ),
    )

    print(
        "✅ Existing V3 acquisition protocol verified"
    )

else:

    PROTOCOL_OUTPUT.write_text(
        json.dumps(
            protocol,
            indent=2,
        )
        + "\n"
    )

    print(
        "✅ V3 acquisition protocol frozen"
    )


# ============================================================
# Candidate demos
# ============================================================

candidate_paths = sorted(
    CANDIDATE_DIR.glob(
        "*.dem"
    )
)


# ============================================================
# Historical demo SHA registry
#
# Everything under data/raw except v3_incoming is treated
# as historical and unavailable to D_V3_DEV.
# ============================================================

historical_paths = sorted(
    path
    for path in RAW_ROOT.rglob(
        "*.dem"
    )
    if (
        CANDIDATE_DIR
        not in path.parents
    )
)


print()
print("=" * 104)
print("V3 GATE 3A — D_V3_DEV CANDIDATE POOL AUDIT")
print("=" * 104)

print()
print(
    "Candidate directory:",
    CANDIDATE_DIR,
)

print(
    "Candidate .dem files:",
    len(
        candidate_paths
    ),
)

print(
    "Historical .dem files scanned:",
    len(
        historical_paths
    ),
)

print(
    "Target D_V3_DEV matches:",
    TARGET_DEV_MATCHES,
)

print(
    "Minimum date:",
    (
        "strictly after "
        f"{MIN_DATE_EXCLUSIVE.isoformat()}"
    ),
)


# ============================================================
# Hash historical corpus
# ============================================================

historical_sha = {}

if historical_paths:

    print()
    print(
        "Hashing historical demos..."
    )


for index, path in enumerate(
    historical_paths,
    start=1,
):

    digest = sha256_file(
        path
    )

    historical_sha.setdefault(
        digest,
        [],
    ).append(
        str(
            path
        )
    )

    if (
        index % 20 == 0
        or index
        == len(
            historical_paths
        )
    ):

        print(
            " ",
            f"{index}/"
            f"{len(historical_paths)}"
        )


# ============================================================
# Audit candidates
# ============================================================

candidate_hash_counts = {}

candidate_records = []


if candidate_paths:

    print()
    print(
        "Hashing V3 candidates..."
    )


for index, path in enumerate(
    candidate_paths,
    start=1,
):

    digest = sha256_file(
        path
    )

    candidate_hash_counts[
        digest
    ] = (
        candidate_hash_counts.get(
            digest,
            0,
        )
        + 1
    )

    parsed_date = parse_filename_date(
        path.name
    )

    filename_lower = (
        path.name.lower()
    )

    looks_like_mirage = (
        MAP_TOKEN
        in filename_lower
    )

    historical_overlap = (
        digest
        in historical_sha
    )

    candidate_records.append({
        "filename":
            path.name,

        "path":
            str(
                path
            ),

        "sha256":
            digest,

        "match_date":
            (
                parsed_date.isoformat()
                if parsed_date
                is not None
                else None
            ),

        "date_parse_ok":
            parsed_date
            is not None,

        "strictly_after_cutoff":
            (
                parsed_date
                is not None
                and parsed_date
                > MIN_DATE_EXCLUSIVE
            ),

        "filename_mirage_token":
            looks_like_mirage,

        "historical_sha_overlap":
            historical_overlap,

        "historical_overlap_paths":
            (
                " | ".join(
                    historical_sha[
                        digest
                    ]
                )
                if historical_overlap
                else ""
            ),

        # Filled after all candidate hashes known.
        "duplicate_within_candidate":
            False,

        "eligible_preparse":
            False,
    })

    if (
        index % 20 == 0
        or index
        == len(
            candidate_paths
        )
    ):

        print(
            " ",
            f"{index}/"
            f"{len(candidate_paths)}"
        )


# ============================================================
# Complete eligibility
# ============================================================

for record in candidate_records:

    digest = record[
        "sha256"
    ]

    duplicate = (
        candidate_hash_counts[
            digest
        ]
        > 1
    )

    record[
        "duplicate_within_candidate"
    ] = duplicate

    record[
        "eligible_preparse"
    ] = all([
        record[
            "date_parse_ok"
        ],

        record[
            "strictly_after_cutoff"
        ],

        record[
            "filename_mirage_token"
        ],

        not record[
            "historical_sha_overlap"
        ],

        not duplicate,
    ])


# ============================================================
# Write audit artifact
# ============================================================

AUDIT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fieldnames = [
    "filename",
    "path",
    "sha256",
    "match_date",
    "date_parse_ok",
    "strictly_after_cutoff",
    "filename_mirage_token",
    "historical_sha_overlap",
    "historical_overlap_paths",
    "duplicate_within_candidate",
    "eligible_preparse",
]


with AUDIT_OUTPUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        candidate_records
    )


# ============================================================
# Summary
# ============================================================

eligible = [
    record
    for record
    in candidate_records
    if record[
        "eligible_preparse"
    ]
]


eligible.sort(
    key=lambda row: (
        row[
            "match_date"
        ],
        row[
            "filename"
        ],
    )
)


historical_overlap_count = sum(
    1
    for record
    in candidate_records
    if record[
        "historical_sha_overlap"
    ]
)

candidate_duplicate_count = sum(
    1
    for record
    in candidate_records
    if record[
        "duplicate_within_candidate"
    ]
)

bad_date_count = sum(
    1
    for record
    in candidate_records
    if not record[
        "date_parse_ok"
    ]
)

old_date_count = sum(
    1
    for record
    in candidate_records
    if (
        record[
            "date_parse_ok"
        ]
        and not record[
            "strictly_after_cutoff"
        ]
    )
)

non_mirage_name_count = sum(
    1
    for record
    in candidate_records
    if not record[
        "filename_mirage_token"
    ]
)


print()
print("=" * 104)
print("CANDIDATE POOL SUMMARY")
print("=" * 104)

print(
    "Total candidate demos:",
    len(
        candidate_records
    ),
)

print(
    "Pre-parse eligible:",
    len(
        eligible
    ),
)

print(
    "Historical SHA overlaps:",
    historical_overlap_count,
)

print(
    "Within-candidate duplicates:",
    candidate_duplicate_count,
)

print(
    "Unparseable filename dates:",
    bad_date_count,
)

print(
    "Date <= 2026-09-12:",
    old_date_count,
)

print(
    "Filename lacks Mirage token:",
    non_mirage_name_count,
)


if eligible:

    print()
    print(
        "Eligible chronological range:"
    )

    print(
        "  first:",
        eligible[
            0
        ][
            "match_date"
        ],
        eligible[
            0
        ][
            "filename"
        ],
    )

    print(
        "  last :",
        eligible[
            -1
        ][
            "match_date"
        ],
        eligible[
            -1
        ][
            "filename"
        ],
    )


print()
print("=" * 104)
print("GATE 3A RESULT")
print("=" * 104)


if len(
    eligible
) >= TARGET_DEV_MATCHES:

    selected_preview = (
        eligible[
            :TARGET_DEV_MATCHES
        ]
    )

    print(
        "✅ CANDIDATE POOL LARGE ENOUGH"
    )

    print(
        "Eligible demos:",
        len(
            eligible
        ),
    )

    print(
        "Frozen selection rule would choose "
        "the first:",
        TARGET_DEV_MATCHES,
    )

    print()
    print(
        "Selection preview:"
    )

    print(
        "  first selected:",
        selected_preview[
            0
        ][
            "filename"
        ],
    )

    print(
        "  last selected :",
        selected_preview[
            -1
        ][
            "filename"
        ],
    )

    print()
    print(
        "NEXT_ACTION="
        "PARSE_VALIDATE_AND_FREEZE_D_V3_DEV_MANIFEST"
    )

else:

    needed = (
        TARGET_DEV_MATCHES
        - len(
            eligible
        )
    )

    print(
        "⚠️ D_V3_DEV CANDIDATE POOL NOT READY"
    )

    print(
        "Eligible demos:",
        len(
            eligible
        ),
    )

    print(
        "Additional new Mirage demos needed:",
        needed,
    )

    print()
    print(
        "Put NEW demos here:"
    )

    print(
        " ",
        CANDIDATE_DIR,
    )

    print()
    print(
        "Required filename pattern example:"
    )

    print(
        "  "
        "2026-09-13_team-a_vs_team-b_mirage.dem"
    )

    print()
    print(
        "NEXT_ACTION=ACQUIRE_NEW_V3_MIRAGE_DEMOS"
    )


print("=" * 104)

print()
print("Artifacts:")
print(" ", PROTOCOL_OUTPUT)
print(" ", AUDIT_OUTPUT)
