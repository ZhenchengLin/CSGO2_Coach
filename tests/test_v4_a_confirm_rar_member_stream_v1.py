"""Offline tests for bounded, create-only selected-member extraction."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_v4_a_confirm_rar_member_v1 import (
    ExtractionStop,
    stream_selected_member,
)


def command_for(payload, exit_code=0):
    return [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.stdout.buffer.write({payload!r}); "
            "sys.stdout.buffer.flush(); "
            f"sys.exit({exit_code})"
        ),
    ]


def extract(tmp_path, payload=b"demo", declared=None, exit_code=0):
    if declared is None:
        declared = len(payload)

    destination = tmp_path / "selected.dem"

    return stream_selected_member(
        command=command_for(payload, exit_code),
        destination=destination,
        declared_bytes=declared,
        reserve_bytes=0,
        timeout_seconds=10,
    )


def test_successful_stream(tmp_path):
    payload = b"example-demo-bytes"

    result = extract(tmp_path, payload)

    assert (tmp_path / "selected.dem").read_bytes() == payload
    assert result["extracted_demo_size_bytes"] == len(payload)
    assert result["extracted_demo_sha256"] == (
        hashlib.sha256(payload).hexdigest()
    )
    assert result["demo_header_map_verified"] is False
    assert result["technical_eligibility_evaluated"] is False
    assert result["model_scoring_performed"] is False


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "selected.dem"
    destination.write_bytes(b"original")

    with pytest.raises(ExtractionStop, match="already exists"):
        extract(tmp_path, b"replacement")

    assert destination.read_bytes() == b"original"


def test_declared_size_too_small(tmp_path):
    with pytest.raises(ExtractionStop, match="exceeded"):
        extract(tmp_path, b"abcdef", declared=3)

    assert not (tmp_path / "selected.dem").exists()
    assert not list(tmp_path.glob("*.partial.*"))


def test_declared_size_too_large(tmp_path):
    with pytest.raises(ExtractionStop, match="size differs"):
        extract(tmp_path, b"abc", declared=10)

    assert not (tmp_path / "selected.dem").exists()
    assert not list(tmp_path.glob("*.partial.*"))


def test_extractor_failure(tmp_path):
    with pytest.raises(ExtractionStop, match="returned 7"):
        extract(tmp_path, b"abc", exit_code=7)

    assert not (tmp_path / "selected.dem").exists()
    assert not list(tmp_path.glob("*.partial.*"))


def test_source_verification_failure_does_not_publish(tmp_path):
    destination = tmp_path / "selected.dem"

    def reject_source():
        raise ExtractionStop("Source Archive identity mismatch.")

    with pytest.raises(ExtractionStop, match="Source Archive"):
        stream_selected_member(
            command=command_for(b"demo"),
            destination=destination,
            declared_bytes=4,
            reserve_bytes=0,
            timeout_seconds=10,
            verify_source=reject_source,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial.*"))


def test_zero_declared_size_rejected(tmp_path):
    with pytest.raises(ExtractionStop):
        extract(tmp_path, b"", declared=0)


def test_stream_result_is_deterministic(tmp_path):

    a = tmp_path / "a"
    b = tmp_path / "b"

    a.mkdir()
    b.mkdir()

    result_a = extract(a, b"demo")
    result_b = extract(b, b"demo")

    assert (
        result_a["extracted_demo_sha256"]
        == result_b["extracted_demo_sha256"]
    )
