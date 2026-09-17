from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(
    "docs/v3_confirm_acquisition_protocol_v1.json"
)

MODEL_FREEZE = Path(
    "docs/v3_confirm_model_freeze.json"
)

MODEL_VERIFY = Path(
    "docs/v3_confirm_model_verification.json"
)

PROVENANCE = Path(
    "docs/v3_confirm_fit_provenance_repair.json"
)

DEV_MANIFEST = Path(
    "docs/v3_dev_manifest.csv"
)

V2_CONFIRM_MANIFEST = Path(
    "docs/v2_confirm_manifest.csv"
)

DEV_ACQUISITION = Path(
    "docs/v3_dev_acquisition_protocol_v2.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        Path(path).read_text()
    )


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


for path in [
    MODEL_FREEZE,
    MODEL_VERIFY,
    PROVENANCE,
    DEV_MANIFEST,
    V2_CONFIRM_MANIFEST,
    DEV_ACQUISITION,
]:
    require(
        path.exists(),
        f"Missing prerequisite: {path}",
    )


require(
    not OUT.exists(),
    (
        "V3 confirmation acquisition protocol "
        "already exists. Refusing overwrite."
    ),
)


freeze = read_json(
    MODEL_FREEZE
)

verify = read_json(
    MODEL_VERIFY
)

provenance = read_json(
    PROVENANCE
)

dev_protocol = read_json(
    DEV_ACQUISITION
)


# ============================================================
# Pre-confirmation scientific boundary
# ============================================================

require(
    freeze["confirmation_data_accessed"]
    is False,
    "Model freeze records confirmation access.",
)

require(
    freeze[
        "confirmation_performance_calculated"
    ]
    is False,
    "Model freeze records confirmation scoring.",
)

require(
    verify["overall_status"]
    == "PASS",
    "Frozen model verification did not pass.",
)

require(
    provenance["status"]
    == "RESOLVED_BEFORE_CONFIRMATION_ACCESS",
    "Fit provenance is not resolved.",
)

require(
    provenance[
        "confirmation_boundary"
    ][
        "D_V3_CONFIRM_accessed"
    ]
    is False,
    "Provenance record indicates confirmation access.",
)

require(
    dev_protocol[
        "future_confirmation"
    ][
        "D_V3_CONFIRM"
    ]
    == (
        "Must be collected from a later untouched "
        "period after V3 model development is frozen."
    ),
    "Historical V3 future-confirmation rule changed.",
)


parent_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()


record = {
    "version":
        "V3_CONFIRM_ACQUISITION_PROTOCOL_V1",

    "status":
        "FROZEN_BEFORE_CONFIRMATION_ACQUISITION",

    "frozen_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "parent_preconfirmation_model_freeze_commit":
        parent_commit,

    "research_role": {
        "dataset_name":
            "D_V3_CONFIRM",

        "purpose":
            (
                "Independent later-period confirmation "
                "of the frozen V3 forecasting candidate."
            ),

        "development_use":
            False,

        "hyperparameter_tuning_use":
            False,

        "feature_selection_use":
            False,

        "representation_design_use":
            False,
    },

    "temporal_boundary": {
        "latest_D_V3_DEV_match_date":
            "2026-09-15",

        "confirmation_match_date_rule":
            "match_date > 2026-09-15",

        "earliest_allowed_match_date":
            "2026-09-16",

        "same_day_as_latest_development_allowed":
            False,
    },

    "domain": {
        "game":
            "Counter-Strike 2",

        "map":
            "de_mirage",

        "prediction_unit":
            "one eligible de_mirage demo",

        "tick_rate_required":
            64,
    },

    "sample_size": {
        "required_final_eligible_matches":
            20,

        "initial_candidate_queue_capacity":
            25,

        "technical_failure_buffer":
            5,

        "formal_power_calculation_used":
            False,

        "interpretation":
            (
                "Storage-aware predeclared independent "
                "replication sample. Statistical reporting "
                "must include match-level uncertainty rather "
                "than claiming guaranteed power."
            ),
    },

    "candidate_enumeration": {
        "selection_must_be_frozen_before_model_scoring":
            True,

        "source_selection_must_not_depend_on_model_results":
            True,

        "team_event_region_filtering":
            "FORBIDDEN",

        "performance_based_candidate_selection":
            "FORBIDDEN",

        "class_balance_based_candidate_selection":
            "FORBIDDEN",

        "preferred_public_source":
            "HLTV match/demo metadata pipeline",

        "required_preacquisition_fields": [
            "candidate_rank",
            "match_date",
            "event",
            "team1",
            "team2",
            "source",
            "source_match_id",
            "source_url",
        ],

        "ordering": [
            "match_date ascending",
            "source_match_id ascending"
        ],

        "ordering_note":
            (
                "Ordering is deterministic and must not be "
                "changed after raw-demo behavior or model "
                "performance is observed."
            ),
    },

    "eligibility": {
        "required": [
            "match_date > 2026-09-15",
            "map == de_mirage",
            "raw demo acquired successfully",
            "raw demo is readable and non-corrupt",
            "parser completes under the frozen V3 environment",
            "tick rate is 64",
            "required V3 telemetry fields are available",
            "demo SHA256 is unique within D_V3_CONFIRM candidates",
            "demo SHA256 does not overlap D_V3_DEV",
            "demo SHA256 does not overlap V2 D_CONFIRM",
            (
                "frozen target/feature pipeline produces "
                "at least one eligible +5s row"
            ),
            (
                "frozen target/feature pipeline produces "
                "at least one eligible +10s row"
            ),
        ],

        "allowed_exclusion_reasons": [
            "DOWNLOAD_FAILURE",
            "ARCHIVE_CORRUPT",
            "DEMO_CORRUPT",
            "WRONG_MAP",
            "DUPLICATE_SHA",
            "DEVELOPMENT_SHA_OVERLAP",
            "V2_CONFIRM_SHA_OVERLAP",
            "PARSER_FAILURE",
            "WRONG_TICK_RATE",
            "REQUIRED_TELEMETRY_UNAVAILABLE",
            "ZERO_ELIGIBLE_PLUS5_ROWS",
            "ZERO_ELIGIBLE_PLUS10_ROWS",
        ],

        "forbidden_exclusion_reasons": [
            "B3 performs poorly",
            "B2 performs better",
            "unexpected model predictions",
            "undesirable class distribution",
            "rare zones",
            "unexpected tactical style",
            "team identity",
            "tournament identity",
            "region identity",
        ],

        "partial_round_or_horizon_exclusions":
            (
                "Use only the already frozen V3 target "
                "contract. A match is not manually excluded "
                "merely because some rounds/horizons are "
                "excluded by that contract."
            ),
    },

    "selection_rule": {
        "final_manifest_rule":
            (
                "Take the first 20 technically eligible "
                "candidates in frozen candidate_rank order."
            ),

        "eligible_earlier_candidate_may_be_skipped":
            False,

        "manual_substitution":
            False,

        "replacement_after_manifest_freeze":
            False,

        "replacement_after_model_prediction":
            False,

        "replacement_after_metric_calculation":
            False,
    },

    "initial_queue_shortfall_rule": {
        "condition":
            (
                "Fewer than 20 eligible demos remain after "
                "the initial 25-candidate queue is exhausted."
            ),

        "action":
            (
                "Stop. Do not silently append candidates. "
                "Create and commit an explicit acquisition "
                "protocol amendment before acquiring any "
                "additional candidates."
            ),

        "model_scoring_before_amendment":
            False,
    },

    "acquisition_execution": {
        "sequential_download_allowed":
            True,

        "stop_once_20_eligible_identified":
            True,

        "all_25_raw_demos_need_not_be_downloaded":
            True,

        "archive_cleanup_after_success":
            True,

        "non_mirage_cleanup_after_success":
            True,

        "keep_selected_mirage_demos":
            True,
    },

    "manifest_freeze": {
        "output":
            "docs/v3_confirm_manifest.csv",

        "required_matches":
            20,

        "include_sha256":
            True,

        "must_be_committed_before_prediction":
            True,

        "overwrite_existing_manifest":
            False,

        "after_commit":
            (
                "The manifest permanently defines "
                "D_V3_CONFIRM for the V3 confirmation claim."
            ),
    },

    "anti_leakage": {
        "V2_D_CONFIRM":
            "FORBIDDEN",

        "D_V3_DEV":
            "FORBIDDEN_AS_CONFIRMATION",

        "confirmation_labels_before_manifest_freeze":
            "NOT_FOR_MODEL_SELECTION",

        "confirmation_model_predictions_before_manifest_freeze":
            "FORBIDDEN",

        "confirmation_metrics_before_manifest_freeze":
            "FORBIDDEN",
    },

    "frozen_evidence_sha256": {
        "v3_confirm_model_freeze":
            sha256(MODEL_FREEZE),

        "v3_confirm_model_verification":
            sha256(MODEL_VERIFY),

        "v3_confirm_fit_provenance":
            sha256(PROVENANCE),

        "v3_dev_manifest":
            sha256(DEV_MANIFEST),

        "v2_confirm_manifest":
            sha256(V2_CONFIRM_MANIFEST),

        "v3_dev_acquisition_protocol_v2":
            sha256(DEV_ACQUISITION),
    },

    "next_action":
        (
            "Construct and freeze the chronological "
            "D_V3_CONFIRM candidate queue before model "
            "prediction or confirmation scoring."
        ),
}


OUT.write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n"
)


print("V3 CONFIRM ACQUISITION PROTOCOL")
print()
print("latest development date: 2026-09-15")
print("earliest confirmation date: 2026-09-16")
print("map: de_mirage")
print("final eligible matches: 20")
print("initial candidate capacity: 25")
print("selection: first 20 eligible chronologically")
print("manual replacement: NO")
print("performance-based replacement: NO")
print("manifest replacement after freeze: NO")
print("confirmation prediction before manifest freeze: NO")
print("confirmation scoring before manifest freeze: NO")
print()
print("V3_CONFIRM_ACQUISITION_PROTOCOL_FROZEN")
