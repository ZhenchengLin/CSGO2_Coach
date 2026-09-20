"""Full Development replay for the unified V4-A extractor.

Reparse all 53 original demos and compare targets, exclusions,
causal motion, Team Context, and exact 24D/56D matrices.

Development data only. Read-only. No model loading or scoring.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from cs2_tactical_intelligence.v4_a_confirm_demo_inputs_v1 import (
    parse_demo_inputs,
)
from cs2_tactical_intelligence.v4_a_confirm_extract_features_v1 import (
    extract_features,
    MOTION_KEY,
    TEAM_KEY,
    TARGET_KEY,
)
from cs2_tactical_intelligence import (
    v4_a_confirm_feature_matrix_v1 as features,
)


MANIFEST = Path("docs/v3_dev_manifest.csv")

SOURCES = {
    "targets": Path("data/processed/v3_dev_targets_v1.parquet"),
    "motion": Path("data/interim/v3_b1_motion_inputs_v1.parquet"),
    "team_context": Path("data/interim/v4_dev_team_context_v1.parquet"),
}


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


def check_disk():
    free = shutil.disk_usage(".").free / 1024**3
    require(free >= 12, f"Free disk below 12 GiB: {free:.2f}")


def compare_frames(actual, expected, keys, *, label, approximate=()):
    """Compare row keys and all newly generated fields."""

    require(
        actual.select(keys).unique().height == actual.height,
        f"{label}: duplicate new keys.",
    )

    require(
        expected.select(keys).unique().height == expected.height,
        f"{label}: duplicate frozen keys.",
    )

    require(
        actual.height == expected.height,
        f"{label}: row-count mismatch "
        f"{actual.height} != {expected.height}",
    )

    require(
        set(actual.columns) <= set(expected.columns),
        f"{label}: unexpected new fields: "
        f"{sorted(set(actual.columns) - set(expected.columns))}",
    )

    new_rows = {
        tuple(row[key] for key in keys): row
        for row in actual.iter_rows(named=True)
    }

    frozen_rows = {
        tuple(row[key] for key in keys): row
        for row in expected.iter_rows(named=True)
    }

    require(
        set(new_rows) == set(frozen_rows),
        f"{label}: row keys differ.\n"
        f"Only new: {list(set(new_rows) - set(frozen_rows))[:3]}\n"
        f"Only frozen: {list(set(frozen_rows) - set(new_rows))[:3]}",
    )

    for key, new_row in new_rows.items():
        reference = frozen_rows[key]

        for field, value in new_row.items():
            frozen_value = reference[field]

            if field in approximate:
                matched = (
                    value is not None
                    and frozen_value is not None
                    and math.isclose(
                        float(value),
                        float(frozen_value),
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                )
            else:
                matched = value == frozen_value

            require(
                matched,
                f"{label}: mismatch at key={key}, field={field}\n"
                f"New: {value!r}\n"
                f"Frozen: {frozen_value!r}",
            )


def exclusion_rows(path, name, fields):
    with path.open(newline="") as file:
        return Counter(
            tuple(
                int(row[field])
                if field in (
                    "round_num",
                    "current_nominal_tick",
                    "horizon_sec",
                )
                else row[field]
                for field in fields
            )
            for row in csv.DictReader(file)
            if row["demo_filename"] == name
        )


print("\n=== V4 UNIFIED FULL DEVELOPMENT NUMERICAL AUDIT ===")

check_disk()

manifest_identity = json.loads(
    Path("docs/v3_dev_manifest_identity.json").read_text()
)

require(
    manifest_identity["status"] == "FROZEN"
    and sha256(MANIFEST) == manifest_identity["manifest_sha256"],
    "Development manifest identity mismatch.",
)

fit = json.loads(
    Path("docs/v4_a_confirm_fit_contract_v1_frozen.json").read_text()
)

for label, path in SOURCES.items():
    require(
        path.is_file()
        and sha256(path) == fit["source_sha256"][label],
        f"Frozen {label} dataset SHA256 mismatch.",
    )

with MANIFEST.open(newline="") as file:
    manifest = list(csv.DictReader(file))

require(len(manifest) == 53, "Expected exactly 53 Development demos.")

frozen_targets = pl.read_parquet(SOURCES["targets"])
frozen_motion = pl.read_parquet(SOURCES["motion"])
frozen_team = pl.read_parquet(SOURCES["team_context"])

require(frozen_targets.height == 29065, "Frozen target population mismatch.")
require(frozen_motion.height == 14869, "Frozen motion population mismatch.")
require(frozen_team.height == 14869, "Frozen Team Context population mismatch.")

exclusion_fields = [
    "round_num",
    "current_nominal_tick",
    "horizon_sec",
    "reason",
]
round_exclusion_fields = ["round_num", "reason"]

totals = Counter()

for index, record in enumerate(manifest, start=1):
    check_disk()

    name = record["demo_filename"]

    inputs = parse_demo_inputs(
        record["path"],
        demo_filename=name,
        expected_sha256=record["sha256"],
    )

    # Production extraction runs BEFORE any frozen reference comparison.
    new = extract_features(inputs)

    expected_targets = frozen_targets.filter(
        pl.col("demo_filename") == name
    )

    expected_motion = frozen_motion.filter(
        pl.col("demo_filename") == name
    )

    expected_team = frozen_team.filter(
        pl.col("demo_filename") == name
    )

    compare_frames(
        new["targets"],
        expected_targets,
        TARGET_KEY,
        label=f"{name}: Target",
        approximate={
            "current_bomb_X",
            "current_bomb_Y",
            "current_bomb_Z",
        },
    )

    compare_frames(
        new["motion"],
        expected_motion,
        MOTION_KEY,
        label=f"{name}: Motion",
        approximate={
            "prior_bomb_X",
            "prior_bomb_Y",
            "prior_bomb_Z",
            "elapsed_sec",
            "velocity_X",
            "velocity_Y",
            "velocity_Z",
        },
    )

    compare_frames(
        new["team_context"],
        expected_team,
        TEAM_KEY,
        label=f"{name}: Team Context",
    )

    expected_exclusions = exclusion_rows(
        Path("data/interim/v3_dev_target_exclusions.csv"),
        name,
        exclusion_fields,
    )

    expected_round_exclusions = exclusion_rows(
        Path("data/interim/v3_dev_round_exclusions.csv"),
        name,
        round_exclusion_fields,
    )

    require(
        new["exclusions"] == expected_exclusions,
        f"{name}: observation/horizon exclusions differ.",
    )

    require(
        new["round_exclusions"] == expected_round_exclusions,
        f"{name}: round exclusions differ.",
    )

    # Rebuild the frozen feature input using the frozen Development
    # target, motion, and Team Context data in their original order.
    frozen_joined = (
        expected_targets
        .join(
            expected_motion.select(
                MOTION_KEY
                + ["velocity_X", "velocity_Y", "velocity_Z", "status"]
            ),
            on=MOTION_KEY,
            how="left",
            validate="m:1",
        )
        .join(
            expected_team,
            on=TEAM_KEY,
            how="left",
            validate="m:1",
        )
        .sort(TARGET_KEY)
    )

    require(
        frozen_joined.height == new["joined"].height,
        f"{name}: frozen feature join changed row count.",
    )

    for horizon in (5, 10):
        actual_item = new["matrices"][horizon]

        frozen_frame = frozen_joined.filter(
            pl.col("horizon_sec") == horizon
        )

        actual_keys = actual_item["rows"].select(TARGET_KEY)
        frozen_keys = frozen_frame.select(TARGET_KEY)

        require(
            actual_keys.equals(frozen_keys),
            f"{name}: +{horizon}s feature-row ordering differs.",
        )

        frozen_control = features.build_control_features(
            frozen_frame
        )

        frozen_candidate = features.build_candidate_features(
            frozen_frame,
            frozen_control,
        )

        for label, actual_matrix, expected_matrix in (
            (
                "Control 24D",
                actual_item["control"],
                frozen_control,
            ),
            (
                "Candidate 56D",
                actual_item["candidate"],
                frozen_candidate,
            ),
        ):
            require(
                actual_matrix.shape == expected_matrix.shape,
                f"{name}: +{horizon}s {label} shape mismatch.",
            )

            if not np.array_equal(actual_matrix, expected_matrix):
                mismatch = np.argwhere(
                    actual_matrix != expected_matrix
                )[0]

                row, column = map(int, mismatch)

                raise RuntimeError(
                    f"STOP: {name}: +{horizon}s {label} differs.\n"
                    f"First mismatch: row={row}, column={column}\n"
                    f"New: {actual_matrix[row, column]!r}\n"
                    f"Frozen: {expected_matrix[row, column]!r}"
                )

    totals["matches"] += 1
    totals["targets"] += new["targets"].height
    totals["motion"] += new["motion"].height
    totals["team"] += new["team_context"].height
    totals["unknown"] += int(
        new["team_context"]
        .select(
            (
                pl.col("v4_t__UNKNOWN_PLACE")
                + pl.col("v4_ct__UNKNOWN_PLACE")
            ).sum()
        )
        .item()
    )

    print(
        f"[{index:02d}/53] {name} | "
        f"targets={new['targets'].height} | "
        f"motion={new['motion'].height} | "
        f"team={new['team_context'].height} | "
        "24D/56D EXACT PASS",
        flush=True,
    )

    del inputs, new, frozen_joined
    del expected_targets, expected_motion, expected_team
    gc.collect()

require(totals["matches"] == 53, "Match count mismatch.")
require(totals["targets"] == 29065, "Total target rows mismatch.")
require(totals["motion"] == 14869, "Total motion rows mismatch.")
require(totals["team"] == 14869, "Total Team Context rows mismatch.")
require(totals["unknown"] == 1, "UNKNOWN_PLACE total mismatch.")

print("\n=== FINAL UNIFIED NUMERICAL AUDIT SUMMARY ===")
print("Development demos:", totals["matches"], "/ 53")
print("Target rows:", f"{totals['targets']:,}")
print("Causal motion rows:", f"{totals['motion']:,}")
print("Team Context snapshots:", f"{totals['team']:,}")
print("UNKNOWN_PLACE count:", totals["unknown"])
print("Target, exclusions, motion and Team Context: PASS")
print("24D Control and 56D Candidate: EXACT PASS")
print("\nV4_UNIFIED_FULL_DEV_NUMERICAL_AUDIT_PASS")
print("No Confirmation data was accessed or scored.")
print("No frozen datasets or models were modified.")
