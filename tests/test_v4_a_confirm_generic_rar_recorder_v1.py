"""Offline tests for generic bounded replay and create-only Evidence."""

from __future__ import annotations

import hashlib
import sys

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from record_v4_a_confirm_rar_extraction_v1 import (
    EvidenceStop,
    publish_record,
    replay_member,
)


def command(payload, exit_code=0):
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


def test_replay_success():
    payload = b"synthetic-demo"

    result = replay_member(
        command(payload),
        len(payload),
        timeout=10,
    )

    assert result == {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_replay_exceeds_limit():
    with pytest.raises(EvidenceStop, match="exceeded"):
        replay_member(
            command(b"abcdef"),
            3,
            timeout=10,
        )


def test_replay_truncated():
    with pytest.raises(EvidenceStop, match="size mismatch"):
        replay_member(
            command(b"abc"),
            8,
            timeout=10,
        )


def test_replay_nonzero_exit():
    with pytest.raises(EvidenceStop, match="returned 7"):
        replay_member(
            command(b"abc", exit_code=7),
            3,
            timeout=10,
        )


@pytest.mark.parametrize(
    "size",
    [0, -1, 1024**3 + 1],
)
def test_invalid_declared_size(size):
    with pytest.raises(EvidenceStop):
        replay_member(
            command(b"x"),
            size,
            timeout=10,
        )


def test_record_create_only(tmp_path):
    path = tmp_path / "evidence.json"

    record = {
        "candidate_rank": 14,
        "model_scoring_performed": False,
    }

    record_sha = publish_record(
        path,
        record,
    )

    assert path.is_file()

    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == record_sha
    )

    original = path.read_bytes()

    with pytest.raises(
        EvidenceStop,
        match="already exists",
    ):
        publish_record(
            path,
            {"candidate_rank": 99},
        )

    assert path.read_bytes() == original


def test_symlink_destination_rejected(tmp_path):
    target = tmp_path / "target.json"
    target.write_text(
        "original",
        encoding="utf-8",
    )

    link = tmp_path / "evidence.json"
    link.symlink_to(target)

    with pytest.raises(
        EvidenceStop,
        match="already exists",
    ):
        publish_record(
            link,
            {"candidate_rank": 14},
        )

    assert (
        target.read_text(encoding="utf-8")
        == "original"
    )


def test_symlink_parent_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()

    linked = tmp_path / "linked"
    linked.symlink_to(
        actual,
        target_is_directory=True,
    )

    with pytest.raises(
        EvidenceStop,
        match="Unsafe Evidence directory",
    ):
        publish_record(
            linked / "evidence.json",
            {"candidate_rank": 14},
        )

    assert not (
        actual / "evidence.json"
    ).exists()
