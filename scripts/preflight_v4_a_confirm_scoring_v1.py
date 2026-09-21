#!/usr/bin/env python3

"""
V4-A Confirmation: read-only pre-scoring identity checks.

Verifies:
    Frozen scoring protocol.
    Frozen candidate queue and initial intake ledger.
    Four frozen model files and their SHA256 identities.

Does NOT:
    Parse demos or inspect target labels.
    Load models or generate predictions.
    Calculate metrics.
    Create or approve the final Manifest.
    Authorize confirmation scoring.

A separate final-Manifest and row-identity gate must be
implemented and passed before any real model prediction.
"""

from __future__ import annotations

import hashlib
import json
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCORING_PROTOCOL = (
    ROOT / "docs/v4_a_confirm_scoring_protocol_v1_frozen.json"
)

EXPECTED_SCORING_SHA256 = (
    "4fcdc1e7c7e7a07a93eadedf83534eb4"
    "a6241f68ab20b539b7c3f494c91e396d"
)

MODEL_FREEZE = (
    ROOT / "docs/v4_a_confirm_model_freeze_v1.json"
)

INITIAL_LEDGER = (
    ROOT / "docs/v4_a_confirm_intake_initial_state_v1.json"
)

FROZEN_QUEUE = (
    ROOT / "docs/v4_a_confirm_acquisition_queue_v1.csv"
)

FINAL_MANIFEST = (
    ROOT / "docs/v4_a_confirm_manifest_v1.csv"
)

EXPECTED_MODELS = {
    "control_plus5",
    "candidate_plus5",
    "control_plus10",
    "candidate_plus10",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def regular_file(path):
    require(
        path.is_file() and not path.is_symlink(),
        f"Missing or invalid regular file: {path}",
    )


def load_json(path):
    regular_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def checked_repo_path(relative):
    require(
        isinstance(relative, str)
        and bool(relative)
        and not Path(relative).is_absolute(),
        "Invalid model artifact path.",
    )

    path = ROOT / relative

    require(
        path.resolve().is_relative_to(ROOT.resolve()),
        "Model artifact path escapes repository.",
    )

    regular_file(path)

    return path


def main():
    print("=== V4 CONFIRMATION PRE-SCORING GATE ===")

    # 1. Verify the exact frozen scoring protocol.
    regular_file(SCORING_PROTOCOL)

    require(
        sha256_file(SCORING_PROTOCOL)
        == EXPECTED_SCORING_SHA256,
        "Frozen scoring protocol SHA256 mismatch.",
    )

    scoring = load_json(SCORING_PROTOCOL)

    require(
        scoring["status"]
        == "FROZEN_BEFORE_CONFIRMATION_SCORING",
        "Scoring protocol is not frozen.",
    )

    require(
        scoring["corpus"]["required_eligible_matches"] == 20,
        "Unexpected confirmation sample size.",
    )

    require(
        scoring["prediction"]["horizons_sec"] == [5, 10],
        "Unexpected prediction horizons.",
    )

    print("Frozen scoring protocol: VERIFIED")

    # 2. Verify the original frozen queue through the
    #    immutable initial Ledger.
    ledger = load_json(INITIAL_LEDGER)

    regular_file(FROZEN_QUEUE)

    require(
        ledger["candidate_count"] == 25
        and ledger["required_eligible_matches"] == 20,
        "Initial Ledger sample plan mismatch.",
    )

    require(
        len(ledger["candidates"]) == 25,
        "Initial Ledger candidate count mismatch.",
    )

    require(
        sha256_file(FROZEN_QUEUE)
        == ledger["frozen_queue_sha256"],
        "Frozen candidate queue SHA256 mismatch.",
    )

    print("Frozen queue and initial Ledger: VERIFIED")

    # 3. Verify actual model bytes WITHOUT loading models.
    freeze = load_json(MODEL_FREEZE)

    require(
        set(freeze["models"]) == EXPECTED_MODELS,
        "Unexpected frozen model inventory.",
    )

    require(
        freeze["confirmation_data_accessed"] is False
        and freeze["confirmation_scoring_performed"] is False,
        "Frozen model record has unexpected confirmation activity.",
    )

    for horizon in (5, 10):
        for variant, dimensions in (
            ("control", 24),
            ("candidate", 56),
        ):
            name = f"{variant}_plus{horizon}"
            model = freeze["models"][name]

            require(
                model["feature_dimensions"] == dimensions,
                f"{name}: feature dimension mismatch.",
            )

            require(
                model["class_order"]
                == scoring["prediction"]["class_order"],
                f"{name}: target class order mismatch.",
            )

            require(
                model["feature_order"]
                == scoring["prediction"][
                    f"{variant}_feature_order"
                ],
                f"{name}: feature order mismatch.",
            )

            path = checked_repo_path(model["artifact"])

            require(
                sha256_file(path) == model["sha256"],
                f"{name}: model artifact SHA256 mismatch.",
            )

            print(f"{name}: SHA256 VERIFIED")

    # 4. Identity checks above do not authorize scoring.
    if not FINAL_MANIFEST.exists():
        print("Final Confirmation Manifest: NOT CREATED")
    else:
        regular_file(FINAL_MANIFEST)
        print("Final Confirmation Manifest: EXISTS, NOT VERIFIED")

    print()
    print("STATUS: BLOCKED")
    print("Reason: Final Manifest and target-row identities")
    print("        have not passed the future scoring gate.")
    print("Models loaded: NONE")
    print("Predictions generated: NONE")
    print("Confirmation metrics calculated: NONE")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(
            "PRE-SCORING GATE STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
