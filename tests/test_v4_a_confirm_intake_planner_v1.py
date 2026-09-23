"""Offline tests for the generic V4-A Intake state classifier."""

from __future__ import annotations

import sys

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

from plan_v4_a_confirm_intake_v1 import (
    IntakeStop,
    classify_stage,
)


FIELDS = (
    "source",
    "staging",
    "event",
    "extraction",
    "audit",
)


def classify(flags):
    return classify_stage(**dict(zip(FIELDS, flags)))


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            (False, False, False, False, False),
            "SOURCE_CAPTURE_PENDING",
        ),
        (
            (True, False, False, False, False),
            "SOURCE_REVIEW_PENDING",
        ),
        (
            (True, True, False, False, False),
            "ARCHIVE_EVENT_PENDING",
        ),
        (
            (True, True, True, False, False),
            "DEMO_EXTRACTION_PENDING",
        ),
        (
            (True, True, True, True, False),
            "TECHNICAL_AUDIT_PENDING",
        ),
        (
            (True, True, True, True, True),
            "FORMAL_DECISION_PENDING",
        ),
    ],
)
def test_valid_resume_stages(flags, expected):
    assert classify(flags) == expected


@pytest.mark.parametrize(
    "flags",
    [
        (False, True, False, False, False),
        (True, False, True, False, False),
        (True, True, False, True, False),
        (True, True, True, False, True),
        (False, False, False, False, True),
    ],
)
def test_invalid_evidence_sequences_stop(flags):
    with pytest.raises(IntakeStop):
        classify(flags)


def test_resume_stage_is_deterministic():
    flags = (True, True, True, False, False)

    assert classify(flags) == "DEMO_EXTRACTION_PENDING"
    assert classify(flags) == "DEMO_EXTRACTION_PENDING"
