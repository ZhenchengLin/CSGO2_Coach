#!/usr/bin/env python3

"""
V4-A Rank 1: write structured data-integrity evidence once.

The existing Runner performs the actual audit. This Writer
records its returned observations and file identities.

It does not assign technical eligibility, create the final
Manifest, select matches, load models or calculate metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from pathlib import Path

from audit_v4_a_confirm_rank1_technical_intake_v1 import (
    ROOT,
    audit_rank1,
    require,
    sha256_file,
)


OUTPUT = ROOT / (
    "docs/v4_a_confirm_intake_audits/"
    "rank_01_data_integrity_v1.json"
)

FINAL_MANIFEST = ROOT / "docs/v4_a_confirm_manifest_v1.csv"

# These are the code, contract and provenance files used to
# interpret the observation. Store their exact byte identities.
FINGERPRINT_PATHS = (
    "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json",
    "docs/v4_a_confirm_acquisition_amendment_v1.json",
    "docs/v4_a_confirm_feature_extraction_contract_v1_frozen.json",
    "docs/v4_a_confirm_scoring_protocol_v1_frozen.json",
    "docs/v4_a_confirm_acquisition_queue_v1.csv",
    "docs/v4_a_confirm_intake_initial_state_v1.json",
    "docs/v4_a_confirm_intake_events_v1/"
    "rank_01_2397691_archive_acquired.json",
    "docs/v4_a_confirm_rank1_technical_intake_plan_v1.json",
    "docs/v3_dev_manifest.csv",
    "docs/v2_confirm_manifest.csv",
    "docs/v3_confirm_manifest.csv",
    "scripts/audit_v4_a_confirm_rank1_technical_intake_v1.py",
    "scripts/write_v4_a_confirm_rank1_technical_evidence_v1.py",
    "scripts/audit_v4_a_confirm_raw_tick_clock_v1.py",
    "scripts/audit_v4_a_confirm_demo_identity_v1.py",
    "scripts/read_v4_a_confirm_rar_extraction_record_v1.py",
    "scripts/load_v4_a_confirm_verified_demos_v1.py",
    "scripts/manage_v4_a_confirm_intake_state_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_demo_inputs_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_target_rows_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_target_semantics_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_motion_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_team_context_v1.py",
    "src/cs2_tactical_intelligence/v4_a_confirm_feature_matrix_v1.py",
)


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def fingerprints():
    result = {}

    for relative in FINGERPRINT_PATHS:
        path = ROOT / relative

        require(
            path.is_file() and not path.is_symlink(),
            f"Required provenance file unavailable: {relative}",
        )

        result[relative] = sha256_file(path)

    return result


def validate_observation(observation):
    require(
        isinstance(observation, dict)
        and observation.get("version")
        == "V4_A_RANK1_DATA_INTEGRITY_OBSERVATION_V1"
        and observation.get("record_type")
        == "DATA_INTEGRITY_NOT_ELIGIBILITY_DECISION"
        and observation.get("status")
        == "RANK1_DATA_INTEGRITY_AUDIT_PASSED",
        "Unexpected Runner observation.",
    )

    require(
        observation.get("candidate_rank") == 1
        and observation.get("source_match_id") == "2397691",
        "Observation candidate identity mismatch.",
    )

    require(
        observation.get("demo_sha256")
        == "848a9e9d8b0593e3684a19f681d4f9c3902a9490ae530fe20667cdb7cc5843b1",
        "Observation Demo identity mismatch.",
    )

    require(
        observation.get("parsed_map") == "de_mirage",
        "Observation map mismatch.",
    )

    clock = observation.get("raw_clock", {})

    require(
        clock.get("within_v3_precedent_tolerance") is True
        and clock.get("valid_clock_intervals", 0) > 0,
        "Observation raw clock did not pass.",
    )

    plus5 = observation.get("plus5_rows")
    plus10 = observation.get("plus10_rows")

    require(
        isinstance(plus5, int)
        and isinstance(plus10, int)
        and plus5 > 0
        and plus10 > 0
        and observation.get("target_rows") == plus5 + plus10,
        "Observation Target Row accounting mismatch.",
    )

    require(
        observation.get("causal_motion_rows")
        == observation.get("team_context_rows")
        == observation.get("independently_audited_snapshots"),
        "Observation snapshot accounting mismatch.",
    )

    require(
        observation.get("joined_rows")
        == observation.get("target_rows"),
        "Observation Feature Join changed Target Rows.",
    )

    matrices = observation.get("matrices", {})

    require(
        isinstance(matrices, dict)
        and set(matrices) == {"5", "10"},
        "Observation horizon coverage mismatch.",
    )

    for horizon, expected_rows in (("5", plus5), ("10", plus10)):
        matrix = matrices[horizon]

        require(
            matrix.get("rows") == expected_rows
            and matrix.get("control_shape") == [expected_rows, 24]
            and matrix.get("candidate_shape") == [expected_rows, 56]
            and matrix.get("dtype") == "float32"
            and matrix.get("all_values_finite") is True
            and matrix.get("candidate_first24_equal_control") is True,
            f"Observation +{horizon}s matrix integrity mismatch.",
        )

    require(
        observation.get("legacy_development_match_id_coverage")
        == "INCOMPLETE"
        and observation.get(
            "actual_match_date_independently_verified"
        ) is False
        and observation.get("technical_eligibility_evaluated") is False
        and observation.get("model_scoring_performed") is False
        and observation.get("final_manifest_created") is False,
        "Observation crossed an unresolved scientific boundary.",
    )


def build_payload(observation):
    validate_observation(observation)

    return {
        "version": "V4_A_RANK1_TECHNICAL_EVIDENCE_V1",
        "record_type": "DATA_INTEGRITY_EVIDENCE_ONLY",
        "audit_source_commit": git("rev-parse", "HEAD"),
        "file_sha256": fingerprints(),
        "observation": observation,
        "technical_eligibility_decision": "NOT_RECORDED",
        "actual_match_date_verification": "PENDING",
        "legacy_development_match_id_coverage": "INCOMPLETE",
        "final_confirmation_manifest": "NOT_CREATED",
        "model_scoring": "NOT_PERFORMED",
    }


def validate_payload(payload):
    require(
        isinstance(payload, dict)
        and payload.get("version")
        == "V4_A_RANK1_TECHNICAL_EVIDENCE_V1"
        and payload.get("record_type")
        == "DATA_INTEGRITY_EVIDENCE_ONLY",
        "Unexpected Evidence Record schema.",
    )

    validate_observation(payload.get("observation"))

    require(
        payload.get("technical_eligibility_decision")
        == "NOT_RECORDED"
        and payload.get("actual_match_date_verification")
        == "PENDING"
        and payload.get("legacy_development_match_id_coverage")
        == "INCOMPLETE"
        and payload.get("final_confirmation_manifest")
        == "NOT_CREATED"
        and payload.get("model_scoring")
        == "NOT_PERFORMED",
        "Evidence Record crossed the scientific boundary.",
    )

    require(
        payload.get("file_sha256") == fingerprints(),
        "Evidence source file SHA256 mismatch.",
    )

    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            payload["audit_source_commit"],
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )


def write_once(path, payload):
    """Atomically create an evidence file without overwriting it."""

    require(
        not path.exists() and not path.is_symlink(),
        f"Evidence already exists: {path}",
    )

    content = (
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)

    # A temporary file and exclusive hard link prevent partial
    # publication or replacement of an existing evidence record.
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".rank01_evidence_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary, path)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def self_test():
    """Fast storage test; does not parse a Demo."""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.json"
        value = {"test": "WRITE_ONCE"}

        write_once(path, value)

        require(
            json.loads(path.read_text(encoding="utf-8"))
            == value,
            "Evidence write/read self-test failed.",
        )

        try:
            write_once(path, {"test": "SHOULD_NOT_REPLACE"})
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Evidence overwrite-protection self-test failed."
            )

        require(
            json.loads(path.read_text(encoding="utf-8"))
            == value,
            "Evidence changed during overwrite-protection test.",
        )

    print("Evidence atomic write: PASS")
    print("Evidence overwrite protection: PASS")
    print("Raw Demo parsed: NO")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    require(
        not FINAL_MANIFEST.exists(),
        "Final Confirmation Manifest already exists.",
    )

    if args.check:
        require(
            OUTPUT.is_file() and not OUTPUT.is_symlink(),
            "Evidence Record is missing.",
        )

        payload = json.loads(
            OUTPUT.read_text(encoding="utf-8")
        )

        validate_payload(payload)

        print("Evidence Record:", OUTPUT.relative_to(ROOT))
        print("Evidence SHA256:", sha256_file(OUTPUT))
        print("Evidence provenance and schema: VERIFIED")
        print("Raw Demo re-parsed by --check: NO")
        print("Technical eligibility decision: NOT RECORDED")
        return

    require(
        args.write,
        "Unexpected Evidence Writer mode.",
    )

    require(
        not OUTPUT.exists() and not OUTPUT.is_symlink(),
        "Evidence Record already exists; refusing to overwrite.",
    )

    require(
        git("branch", "--show-current") == "main"
        and git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "Repository is not at synchronized main.",
    )

    subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=True,
    )

    print("=== RUN VERIFIED RANK 1 TECHNICAL AUDIT ===")

    observation = audit_rank1()

    payload = build_payload(observation)

    validate_payload(payload)

    write_once(OUTPUT, payload)

    print()
    print("=== EVIDENCE RECORD CREATED ===")
    print("Evidence Record:", OUTPUT.relative_to(ROOT))
    print("Evidence SHA256:", sha256_file(OUTPUT))
    print("Source Commit:", payload["audit_source_commit"])
    print("Technical eligibility decision: NOT RECORDED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            "TECHNICAL EVIDENCE STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
