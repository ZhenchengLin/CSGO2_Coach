"""Offline tests for the generic V4-A Technical Auditor."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v4_a_confirm_generic_technical_v1 import (
    AuditStop,
    validate_observation,
    write_once,
)


def observation():
    return {
        "raw_clock": {
            "within_v3_precedent_tolerance": True,
            "valid_clock_intervals": 100,
        },
        "parsed_map": "de_mirage",
        "plus5_rows": 3,
        "plus10_rows": 2,
        "target_rows": 5,
        "joined_rows": 5,
        "motion_rows": 4,
        "team_context_rows": 3,
        "independently_audited_snapshots": 3,
        "historical_demo_sha256_overlap": False,
        "earlier_v4_demo_sha256_overlap": False,
        "available_historical_match_id_overlap": False,
        "matrices": {
            "5": {
                "rows": 3,
                "control_shape": [3, 24],
                "candidate_shape": [3, 56],
                "dtype": "float32",
                "all_values_finite": True,
                "candidate_first24_equal_control": True,
            },
            "10": {
                "rows": 2,
                "control_shape": [2, 24],
                "candidate_shape": [2, 56],
                "dtype": "float32",
                "all_values_finite": True,
                "candidate_first24_equal_control": True,
            },
        },
    }


def test_valid_observation():
    validate_observation(
        observation()
    )


def test_zero_plus10_rejected():
    value = observation()
    value["plus10_rows"] = 0

    with pytest.raises(
        AuditStop,
        match=r"Missing \+5/\+10",
    ):
        validate_observation(value)


def test_join_row_change_rejected():
    value = observation()
    value["joined_rows"] = 4

    with pytest.raises(
        AuditStop,
        match="row accounting",
    ):
        validate_observation(value)


def test_missing_snapshot_audit_rejected():
    value = observation()
    value["independently_audited_snapshots"] = 2

    with pytest.raises(
        AuditStop,
        match="identity/occupancy",
    ):
        validate_observation(value)


def test_motion_and_snapshot_counts_may_differ():
    value = observation()
    value["motion_rows"] = 10

    validate_observation(value)


def test_wrong_candidate_shape_rejected():
    value = observation()
    value["matrices"]["5"]["candidate_shape"] = [3, 55]

    with pytest.raises(
        AuditStop,
        match="matrix invariant",
    ):
        validate_observation(value)


def test_candidate_prefix_mismatch_rejected():
    value = observation()
    value["matrices"]["10"][
        "candidate_first24_equal_control"
    ] = False

    with pytest.raises(
        AuditStop,
        match="matrix invariant",
    ):
        validate_observation(value)


@pytest.mark.parametrize(
    "field",
    [
        "historical_demo_sha256_overlap",
        "earlier_v4_demo_sha256_overlap",
        "available_historical_match_id_overlap",
    ],
)
def test_identity_overlap_rejected(field):
    value = observation()
    value[field] = True

    with pytest.raises(AuditStop):
        validate_observation(value)


def test_raw_clock_not_verified_rejected():
    value = observation()

    value["raw_clock"][
        "within_v3_precedent_tolerance"
    ] = False

    with pytest.raises(
        AuditStop,
        match="Raw clock",
    ):
        validate_observation(value)


def test_wrong_map_rejected():
    value = observation()
    value["parsed_map"] = "de_anubis"

    with pytest.raises(
        AuditStop,
        match="de_mirage",
    ):
        validate_observation(value)


def test_write_once_is_create_only(tmp_path):
    path = tmp_path / "evidence.json"

    first = {
        "candidate_rank": 14,
        "model_scoring_performed": False,
    }

    write_once(
        path,
        first,
    )

    original = path.read_bytes()

    with pytest.raises(
        AuditStop,
        match="already exists",
    ):
        write_once(
            path,
            {"candidate_rank": 99},
        )

    assert path.read_bytes() == original


def test_write_rejects_symlink_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()

    linked = tmp_path / "linked"
    linked.symlink_to(
        real,
        target_is_directory=True,
    )

    with pytest.raises(
        AuditStop,
        match="Unsafe Technical Evidence directory",
    ):
        write_once(
            linked / "evidence.json",
            {"candidate_rank": 14},
        )

    assert not (
        real / "evidence.json"
    ).exists()
