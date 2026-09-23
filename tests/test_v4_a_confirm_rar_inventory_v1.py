"""Offline safety tests for the generic RAR Inventory Validator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_v4_a_confirm_rar_inventory_v1 import (
    InventoryStop,
    inspect_inventory,
)


NAMES = [
    "misa-vs-honved-m1-mirage.dem",
    "misa-vs-honved-m2-anubis.dem",
    "misa-vs-honved-m3-dust2.dem",
]

SIZES = [
    316535672,
    341773107,
    275895077,
]


def listing(names=None, sizes=None, modes=None):
    names = NAMES if names is None else names
    sizes = SIZES if sizes is None else sizes

    if modes is None:
        modes = ["-rw-r--r--"] * len(names)

    names_text = "\n".join(names) + "\n"

    details_text = "\n".join(
        f"{mode} 0 0 0 {size} Sep 21 06:51 {name}"
        for name, size, mode in zip(names, sizes, modes)
    ) + "\n"

    return names_text, details_text


def inspect(**kwargs):
    return inspect_inventory(*listing(**kwargs))


def test_rank14_inventory():
    result = inspect()

    assert result["selected_member"] == NAMES[0]
    assert result["selected_member_declared_bytes"] == SIZES[0]
    assert len(result["archive_members"]) == 3

    assert result["demo_header_map_verified"] is False
    assert result["extraction_performed"] is False
    assert result["technical_eligibility_evaluated"] is False
    assert result["model_scoring_performed"] is False


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.dem",
        "/tmp/escape.dem",
        "folder/mirage.dem",
        r"folder\mirage.dem",
        ".hidden.dem",
        "demo name.dem",
        "équipe-m1-mirage.dem",
    ],
)
def test_unsafe_member_names(unsafe):
    with pytest.raises(InventoryStop):
        inspect(names=[unsafe, NAMES[1], NAMES[2]])


def test_duplicate_member():
    with pytest.raises(InventoryStop, match="Duplicate"):
        inspect(names=[NAMES[0], NAMES[0], NAMES[2]])


def test_symlink_member():
    with pytest.raises(InventoryStop, match="regular file"):
        inspect(
            modes=[
                "lrwxr-xr-x",
                "-rw-r--r--",
                "-rw-r--r--",
            ]
        )


def test_no_mirage_candidate():
    with pytest.raises(InventoryStop, match="exactly one Mirage"):
        inspect(
            names=[
                "misa-vs-honved-m1-nuke.dem",
                NAMES[1],
                NAMES[2],
            ]
        )


def test_ambiguous_mirage_candidates():
    with pytest.raises(InventoryStop, match="exactly one Mirage"):
        inspect(
            names=[
                NAMES[0],
                "misa-vs-honved-m2-mirage.dem",
                NAMES[2],
            ]
        )


@pytest.mark.parametrize("size", [0, 1024**3 + 1])
def test_invalid_member_size(size):
    with pytest.raises(InventoryStop, match="size"):
        inspect(sizes=[size, SIZES[1], SIZES[2]])


def test_listing_identity_mismatch():
    names, details = listing()

    details = details.replace(
        NAMES[0],
        "different-m1-mirage.dem",
        1,
    )

    with pytest.raises(InventoryStop, match="identity"):
        inspect_inventory(names, details)


def test_listing_count_mismatch():
    names, details = listing()

    details = "\n".join(details.splitlines()[:-1]) + "\n"

    with pytest.raises(InventoryStop, match="counts differ"):
        inspect_inventory(names, details)


def test_malformed_metadata():
    names, details = listing()

    details = details.replace(
        "Sep 21 06:51",
        "Sep 21",
        1,
    )

    with pytest.raises(InventoryStop, match="metadata"):
        inspect_inventory(names, details)


def test_inventory_is_deterministic():
    assert inspect() == inspect()
