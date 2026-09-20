"""V4-A Gate 6C: freeze independent confirmation acquisition rules.

Metadata and artifact-identity checks only.
No demo download, parsing, model prediction, or scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FREEZE = Path("docs/v4_a_confirm_model_freeze_v1.json")
FIT_CONTRACT = Path("docs/v4_a_confirm_fit_contract_v1_frozen.json")
CLOSURE = Path("docs/v4_a_development_closure_v1.json")

DEV_MANIFEST = Path("docs/v3_dev_manifest.csv")
V2_MANIFEST = Path("docs/v2_confirm_manifest.csv")
V3_MANIFEST = Path("docs/v3_confirm_manifest.csv")

OUTPUT = Path(
    "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json"
)

EARLIEST_MATCH_DATE = "2026-09-20"


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_manifest(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


print("\n=== V4-A GATE 6C: CONFIRMATION ACQUISITION ===")

free_gib = shutil.disk_usage(".").free / 1024**3

require(
    free_gib >= 12,
    f"Free disk below 12 GiB: {free_gib:.2f}",
)

for path in (
    FREEZE,
    FIT_CONTRACT,
    CLOSURE,
    DEV_MANIFEST,
    V2_MANIFEST,
    V3_MANIFEST,
):
    require(path.is_file(), f"Missing input: {path}")

require(
    not OUTPUT.exists(),
    f"Protocol already exists: {OUTPUT}",
)

freeze = read_json(FREEZE)
contract = read_json(FIT_CONTRACT)
closure = read_json(CLOSURE)

require(
    freeze["status"] == "FOUR_FULL_DEVELOPMENT_MODELS_FITTED",
    "Final models are not frozen.",
)

require(
    freeze["fit_contract_sha256"] == sha256(FIT_CONTRACT),
    "Final model/fitting contract identity mismatch.",
)

require(
    contract["source_sha256"]["development_closure"]
    == sha256(CLOSURE),
    "Fitting contract/development closure identity mismatch.",
)

require(
    closure["status"]
    == "DEVELOPMENT_CLOSED_NOT_INDEPENDENTLY_CONFIRMED",
    "Unexpected development closure status.",
)

require(
    freeze["confirmation_data_accessed"] is False
    and freeze["confirmation_scoring_performed"] is False,
    "Model freeze records prior confirmation use.",
)

expected_models = {
    "control_plus5",
    "candidate_plus5",
    "control_plus10",
    "candidate_plus10",
}

require(
    set(freeze["models"]) == expected_models,
    "Expected four frozen final models.",
)

# Check that every frozen model still exists locally and
# matches the identity recorded before confirmation.

for name in sorted(expected_models):
    info = freeze["models"][name]
    model_path = Path(info["artifact"])

    require(
        model_path.is_file(),
        f"Missing frozen model: {model_path}",
    )

    require(
        sha256(model_path) == info["sha256"],
        f"Frozen model SHA256 mismatch: {name}",
    )

print("Four frozen final model identities: PASS")


# Historical corpus checks inspect metadata only.
# They do not read or score historical demos.

historical = {
    "development": (
        DEV_MANIFEST,
        read_manifest(DEV_MANIFEST),
        53,
    ),
    "v2_confirmation": (
        V2_MANIFEST,
        read_manifest(V2_MANIFEST),
        30,
    ),
    "v3_confirmation": (
        V3_MANIFEST,
        read_manifest(V3_MANIFEST),
        20,
    ),
}

old_demo_hashes = set()

for name, (path, rows, expected_count) in historical.items():
    require(
        len(rows) == expected_count,
        f"Unexpected {name} manifest size.",
    )

    require(
        all(row.get("sha256") for row in rows),
        f"Missing historical SHA256: {name}",
    )

    require(
        all(row.get("match_date") for row in rows),
        f"Missing historical match date: {name}",
    )

    old_demo_hashes.update(row["sha256"] for row in rows)

    print(
        f"{name}: {len(rows)} historical manifest rows — PASS"
    )

require(
    max(
        row["match_date"]
        for _, rows, _ in historical.values()
        for row in rows
    ) < EARLIEST_MATCH_DATE,
    "Historical corpus overlaps the proposed date boundary.",
)

print("Fresh confirmation date boundary: PASS")


# Freeze the scientific rules, not a list of unverified matches.

protocol = {
    "version": "V4_A_CONFIRM_ACQUISITION_PROTOCOL_V1",

    "status": "FROZEN_BEFORE_V4_CONFIRMATION_ACQUISITION",

    "frozen_utc": datetime.now(timezone.utc).isoformat(),

    "research_role": {
        "dataset": "D_V4_A_CONFIRM",
        "purpose": (
            "Independent confirmation of the already frozen "
            "V4-A 56D Candidate versus 24D Control."
        ),
        "development_use": False,
        "feature_selection_use": False,
        "hyperparameter_tuning_use": False,
        "model_replacement_use": False,
    },

    "domain": {
        "game": "Counter-Strike 2",
        "map": "de_mirage",
        "required_raw_tick_rate": 64,
        "prediction_horizons_sec": [5, 10],
        "target_classes": 15,
    },

    "temporal_boundary": {
        "model_freeze_date": "2026-09-19",
        "earliest_allowed_match_date": EARLIEST_MATCH_DATE,
        "rule": "match_date >= 2026-09-20",
        "earlier_matches_allowed": False,
    },

    "sample_plan": {
        "required_eligible_matches": 20,
        "initial_candidate_queue": 25,
        "formal_power_calculation": False,
        "selection_rule": (
            "Select the first 20 technically eligible "
            "candidates in frozen candidate_rank order."
        ),
        "eligible_earlier_candidate_may_be_skipped": False,
        "manual_replacement": False,
        "post_scoring_replacement": False,
        "shortfall_rule": (
            "If the 25-candidate queue is exhausted before "
            "20 eligible matches are obtained, stop. "
            "Freeze and commit an explicit protocol amendment "
            "before acquiring additional candidates."
        ),
    },

    "candidate_queue": {
        "output": "docs/v4_a_confirm_acquisition_queue_v1.csv",
        "must_be_frozen_before_raw_demo_download": True,
        "preferred_metadata_source": "HLTV",
        "required_fields": [
            "candidate_rank",
            "match_date",
            "source_match_id",
            "event",
            "team1",
            "team2",
            "source",
            "source_url",
        ],
        "ordering": [
            "match_date ascending",
            "source_match_id ascending",
        ],
        "source_match_id_unique_within_queue": True,
        "selection_must_not_depend_on_model_performance": True,
        "selection_must_not_depend_on_target_distribution": True,
        "technical_eligibility_checked_after_queue_freeze": True,
    },

    "eligibility": {
        "required": [
            "Match date satisfies the frozen temporal boundary.",
            "The acquired demo is de_mirage.",
            "Raw demo is available, readable and non-corrupt.",
            "Raw tick rate is 64.",
            "Required frozen V3 bomb telemetry is available.",
            "Required frozen V4 current-time player telemetry is available.",
            "Frozen player-identity and occupancy checks pass.",
            "Frozen target pipeline yields at least one eligible +5s row.",
            "Frozen target pipeline yields at least one eligible +10s row.",
            "Demo SHA256 is unique within the new candidate corpus.",
            "Demo SHA256 does not overlap any historical development, "
            "V2 confirmation or V3 confirmation demo.",
            "Match identity does not duplicate any already used match "
            "where source_match_id is available.",
        ],

        "allowed_exclusion_reasons": [
            "DATE_OUT_OF_RANGE",
            "DOWNLOAD_FAILURE",
            "ARCHIVE_CORRUPT",
            "DEMO_CORRUPT",
            "WRONG_MAP",
            "DUPLICATE_SHA",
            "HISTORICAL_DEMO_SHA_OVERLAP",
            "HISTORICAL_MATCH_ID_OVERLAP",
            "PARSER_FAILURE",
            "WRONG_TICK_RATE",
            "REQUIRED_BOMB_TELEMETRY_UNAVAILABLE",
            "REQUIRED_PLAYER_TELEMETRY_UNAVAILABLE",
            "FROZEN_PLAYER_IDENTITY_CHECK_FAILURE",
            "FROZEN_OCCUPANCY_CHECK_FAILURE",
            "ZERO_ELIGIBLE_PLUS5_ROWS",
            "ZERO_ELIGIBLE_PLUS10_ROWS",
        ],

        "forbidden_exclusion_reasons": [
            "Candidate or Control predicts poorly.",
            "The observed target-zone distribution is undesirable.",
            "Rare macro-zones are present.",
            "A particular team or tournament is undesirable.",
            "The match has an unexpected tactical style.",
            "The Candidate performs worse than the Control.",
        ],

        "unknown_player_place_policy": (
            "Preserve the frozen T/CT UNKNOWN_PLACE feature semantics. "
            "Unknown place alone is not a new discretionary "
            "match exclusion reason."
        ),
    },

    "storage_policy": {
        "minimum_free_disk_gib": 12,
        "stop_before_next_download_if_below_threshold": True,
        "sequential_acquisition": True,
        "do_not_delete_historical_demos_or_frozen_models": True,
        "do_not_download_before_candidate_queue_freeze": True,
    },

    "confirmation_execution_boundary": {
        "model_prediction_before_manifest_freeze": False,
        "confirmation_metric_before_manifest_freeze": False,
        "model_refitting_from_confirmation": False,
        "confirmation_based_feature_changes": False,
        "confirmation_scoring_protocol_must_be_frozen_before_scoring": True,
        "confirmation_feature_extraction_must_follow_frozen_v4_contract": True,
        "final_manifest": "docs/v4_a_confirm_manifest_v1.csv",
        "final_manifest_must_be_frozen_before_model_scoring": True,
    },

    "provenance": {
        "model_freeze_path": str(FREEZE),
        "model_freeze_sha256": sha256(FREEZE),
        "fit_contract_sha256": sha256(FIT_CONTRACT),
        "development_closure_sha256": sha256(CLOSURE),

        "historical_manifests": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(rows),
            }
            for name, (path, rows, _) in historical.items()
        },

        "historical_demo_sha256_count": len(old_demo_hashes),
    },

    "next_action": (
        "Freeze the V4-A confirmation scoring and feature-extraction "
        "rules, then build and commit a deterministic 25-candidate "
        "metadata queue before acquiring any new demos."
    ),

    "revision_policy": (
        "Do not overwrite this frozen V1 protocol. "
        "Any necessary changes require an explicit, "
        "versioned and committed amendment."
    ),
}

with OUTPUT.open("x") as file:
    json.dump(protocol, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("\n=== GATE 6C SUMMARY ===")
print("Historical corpora: metadata checked")
print("Four final models: SHA256 PASS")
print("Earliest allowed match date:", EARLIEST_MATCH_DATE)
print("Confirmation target: 20 technically eligible matches")
print("Initial metadata queue: 25 candidates")
print("New demo acquisition: NOT STARTED")
print("Confirmation scoring: NOT STARTED")
print("Protocol:", OUTPUT)

print("\nV4_A_CONFIRM_ACQUISITION_PROTOCOL_FROZEN")
