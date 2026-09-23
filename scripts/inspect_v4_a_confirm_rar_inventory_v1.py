#!/usr/bin/env python3
"""Read-only V4-A RAR inventory validation.

Requires an already verified acquisition and committed event.

A Mirage filename is only an extraction candidate, not proof
of the Demo's actual map.

Does not download, extract, create evidence, decide eligibility,
or perform model scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MAX_MEMBERS = 16
MAX_MEMBER_BYTES = 1024**3
MAX_LISTING_BYTES = 65536

SAFE_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,250}\.dem"
)

MIRAGE_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*-m[1-9][0-9]*-mirage\.dem"
)

REGULAR_FILE_MODE = re.compile(
    r"-[-rwxstST]{9}"
)


class InventoryStop(RuntimeError):
    """Unrecognized or unsafe archive metadata requires review."""


def require(condition, message):
    if not condition:
        raise InventoryStop(message)


def inspect_inventory(names_text, details_text):
    """Validate two independently requested bsdtar inventory views."""

    names = names_text.splitlines()
    details = details_text.splitlines()

    require(
        1 <= len(names) <= MAX_MEMBERS,
        "Unexpected Archive member count.",
    )

    require(
        len(details) == len(names),
        "Inventory listing counts differ.",
    )

    require(
        len(set(name.casefold() for name in names)) == len(names),
        "Duplicate Archive member names.",
    )

    members = []

    for index, (name, detail) in enumerate(
        zip(names, details),
        start=1,
    ):
        require(
            SAFE_NAME.fullmatch(name) is not None,
            f"Unsafe or unsupported member name: {index}.",
        )

        fields = detail.split()

        require(
            len(fields) == 9,
            f"Unexpected member metadata: {index}.",
        )

        mode, owner, group, links, size_text, month, day, time, listed_name = fields

        require(
            REGULAR_FILE_MODE.fullmatch(mode) is not None,
            f"Member {index} is not a regular file.",
        )

        require(
            listed_name == name,
            f"Member {index} listing identity mismatch.",
        )

        require(
            size_text.isdecimal(),
            f"Member {index} has an invalid size.",
        )

        size = int(size_text)

        require(
            0 < size <= MAX_MEMBER_BYTES,
            f"Member {index} exceeds permitted size limits.",
        )

        members.append({
            "path": name,
            "declared_uncompressed_bytes": size,
        })

    candidates = [
        member
        for member in members
        if MIRAGE_NAME.fullmatch(member["path"]) is not None
    ]

    require(
        len(candidates) == 1,
        "Expected exactly one Mirage filename candidate.",
    )

    selected = candidates[0]

    return {
        "archive_members": members,
        "selected_member": selected["path"],
        "selected_member_declared_bytes": (
            selected["declared_uncompressed_bytes"]
        ),
        "demo_header_map_verified": False,
        "extraction_performed": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
    }


def read_listing(archive, *, verbose):
    command = [
        "bsdtar",
        "-tvf" if verbose else "-tf",
        str(archive),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )

    require(
        result.returncode == 0,
        "RAR listing command failed.",
    )

    require(
        0 < len(result.stdout) <= MAX_LISTING_BYTES,
        "RAR listing output exceeds permitted limits.",
    )

    try:
        return result.stdout.decode(
            "utf-8",
            errors="strict",
        )

    except UnicodeDecodeError as exc:
        raise InventoryStop(
            "Unsupported RAR member-name encoding."
        ) from exc


def inspect_rank(rank):
    """Inspect the currently authorized frozen Rank only."""

    sys.path.insert(0, str(ROOT / "scripts"))

    from plan_v4_a_confirm_intake_v1 import resolve_plan

    from manage_v4_a_confirm_intake_state_v1 import (
        load_initial_candidates,
    )

    plan = resolve_plan()

    require(
        plan["rank"] == rank,
        "Requested Rank is not the current frozen candidate.",
    )

    require(
        plan["stage"] == "DEMO_EXTRACTION_PENDING",
        "Current Rank is not ready for Demo Extraction.",
    )

    row = load_initial_candidates()[rank - 1]

    require(
        row["candidate_rank"] == rank
        and row["source_match_id"] == plan["match_id"],
        "Frozen Rank identity mismatch.",
    )

    staging = ROOT / (
        "data/raw/v4_a_confirm_download_staging/"
        f'rank_{rank:02d}_{row["source_match_id"]}'
    )

    archive = staging / "archive.rar"

    require(
        archive.is_file()
        and not archive.is_symlink()
        and staging.is_dir()
        and not staging.is_symlink(),
        "Expected verified RAR Archive is missing or unsafe.",
    )

    event_path = ROOT / (
        "docs/v4_a_confirm_intake_events_v1/"
        f'rank_{rank:02d}_{row["source_match_id"]}_archive_acquired.json'
    )

    require(
        event_path.is_file()
        and not event_path.is_symlink(),
        "Acquisition Event is missing or unsafe.",
    )

    event = json.loads(event_path.read_text(encoding="utf-8"))

    require(
        event["archive_path"] == archive.relative_to(ROOT).as_posix(),
        "Archive path differs from verified Event.",
    )

    # resolve_plan() has already independently verified the
    # complete Archive SHA256 and the committed Event.
    initial_size = archive.stat().st_size

    result = inspect_inventory(
        read_listing(archive, verbose=False),
        read_listing(archive, verbose=True),
    )

    require(
        archive.stat().st_size == initial_size,
        "Archive size changed during inspection.",
    )

    return {
        "candidate_rank": rank,
        "source_match_id": row["source_match_id"],
        "source_archive_path": event["archive_path"],
        "source_archive_sha256": event["archive_sha256"],
        "source_archive_size_bytes": event["archive_size_bytes"],
        **result,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    require(
        1 <= args.rank <= 25,
        "Rank must belong to the frozen candidate queue.",
    )

    result = inspect_rank(args.rank)

    print("=== V4-A GENERIC RAR INVENTORY ===")
    print(json.dumps(result, indent=2))
    print()
    print("Archive extraction: NONE")
    print("Demo Header verification: NOT PERFORMED")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()

    except (
        InventoryStop,
        RuntimeError,
        KeyError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(
            "RAR INVENTORY STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
