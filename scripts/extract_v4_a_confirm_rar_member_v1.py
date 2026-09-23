#!/usr/bin/env python3
"""Safe selected-member RAR extraction for V4-A Confirmation.

The engine extracts exactly one previously validated member using
a bounded stream. It never extracts other archive members.

A successful extraction does not establish the Demo's actual map,
technical eligibility, or permission to perform model scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import select
import shutil
import subprocess
import sys
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHUNK_SIZE = 1024 * 1024

TIMEOUT_SECONDS = 300

MINIMUM_FREE_BYTES = 12 * 1024**3

SAFETY_RESERVE_BYTES = 64 * 1024**2


class ExtractionStop(RuntimeError):
    """Stop without publishing an incomplete selected member."""


def require(condition, message):
    if not condition:
        raise ExtractionStop(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def free_bytes(directory):
    return shutil.disk_usage(directory).free


def verify_budget(directory, declared_bytes, reserve_bytes):
    require(
        declared_bytes > 0,
        "Declared extraction size must be positive.",
    )

    require(
        free_bytes(directory)
        >= declared_bytes + reserve_bytes,
        "STORAGE_BLOCK: insufficient space for selected member.",
    )


def stream_selected_member(
    *,
    command,
    destination,
    declared_bytes,
    reserve_bytes,
    timeout_seconds=TIMEOUT_SECONDS,
    verify_source=None,
):
    """Extract to an exclusive temporary file and publish create-only.

    `command` must already identify one validated archive member.

    The function never overwrites an existing destination. An incomplete
    temporary file created by this invocation is removed on failure.
    """

    destination = Path(destination)

    require(
        destination.parent.is_dir()
        and not destination.parent.is_symlink(),
        "Unsafe extraction directory.",
    )

    require(
        not destination.exists()
        and not destination.is_symlink(),
        "Selected Demo already exists. Review before retrying.",
    )

    require(
        isinstance(declared_bytes, int)
        and 0 < declared_bytes <= 1024**3,
        "Invalid declared selected-member size.",
    )

    require(
        timeout_seconds > 0,
        "Invalid extraction timeout.",
    )

    verify_budget(
        destination.parent,
        declared_bytes,
        reserve_bytes,
    )

    temporary = destination.with_name(
        destination.name + f".partial.{os.getpid()}"
    )

    require(
        not temporary.exists()
        and not temporary.is_symlink(),
        "Temporary extraction path already exists.",
    )

    process = None
    temporary_created = False

    digest = hashlib.sha256()
    total = 0

    deadline = time.monotonic() + timeout_seconds

    try:
        with temporary.open("xb") as output:
            temporary_created = True

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            require(
                process.stdout is not None,
                "Selected-member stream was not opened.",
            )

            fd = process.stdout.fileno()

            while True:
                remaining = deadline - time.monotonic()

                require(
                    remaining > 0,
                    "Selected-member extraction timed out.",
                )

                ready, _, _ = select.select(
                    [fd],
                    [],
                    [],
                    min(remaining, 10.0),
                )

                if not ready:
                    continue

                chunk = os.read(fd, CHUNK_SIZE)

                if not chunk:
                    break

                require(
                    total + len(chunk) <= declared_bytes,
                    "Selected member exceeded its declared size.",
                )

                require(
                    free_bytes(destination.parent)
                    >= reserve_bytes + len(chunk),
                    "STORAGE_BLOCK: extraction would violate disk reserve.",
                )

                output.write(chunk)

                digest.update(chunk)
                total += len(chunk)

            remaining = deadline - time.monotonic()

            require(
                remaining > 0,
                "Selected-member extraction timed out.",
            )

            returncode = process.wait(timeout=remaining)

            require(
                returncode == 0,
                f"Selected-member extractor returned {returncode}.",
            )

            require(
                total == declared_bytes,
                "Extracted member size differs from verified inventory.",
            )

            output.flush()
            os.fsync(output.fileno())

        require(
            temporary.stat().st_size == declared_bytes
            and sha256_file(temporary) == digest.hexdigest(),
            "Extracted temporary file failed independent verification.",
        )

        if verify_source is not None:
            verify_source()

        require(
            not destination.exists()
            and not destination.is_symlink(),
            "Destination appeared during extraction.",
        )

        # Atomic, create-only publication on the same filesystem.
        # os.link refuses to replace an existing destination.
        os.link(temporary, destination)

        return {
            "extracted_demo_path": destination,
            "extracted_demo_size_bytes": total,
            "extracted_demo_sha256": digest.hexdigest(),
            "demo_header_map_verified": False,
            "technical_eligibility_evaluated": False,
            "model_scoring_performed": False,
        }

    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()

            if process.stdout is not None:
                process.stdout.close()

        if temporary_created and temporary.exists():
            temporary.unlink()


def prepare_rank(rank):
    """Resolve the current frozen Rank and its selected RAR member."""

    sys.path.insert(0, str(ROOT / "scripts"))

    from inspect_v4_a_confirm_rar_inventory_v1 import inspect_rank

    from plan_v4_a_confirm_intake_v1 import resolve_plan

    plan = resolve_plan()

    require(
        plan["rank"] == rank
        and plan["stage"] == "DEMO_EXTRACTION_PENDING",
        "Rank is not at the authorized Demo Extraction stage.",
    )

    inventory = inspect_rank(rank)

    archive = ROOT / inventory["source_archive_path"]

    selected = inventory["selected_member"]

    destination = archive.parent / selected

    require(
        archive.is_file()
        and not archive.is_symlink()
        and archive.suffix.lower() == ".rar",
        "Verified source RAR is unavailable.",
    )

    require(
        destination.parent == archive.parent,
        "Selected Demo destination differs from staging directory.",
    )

    require(
        not destination.exists()
        and not destination.is_symlink(),
        "Selected Demo already exists; inspect before resuming.",
    )

    declared_bytes = inventory["selected_member_declared_bytes"]

    verify_budget(
        archive.parent,
        declared_bytes,
        MINIMUM_FREE_BYTES + SAFETY_RESERVE_BYTES,
    )

    return inventory, archive, destination


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Explicitly authorize one selected-member extraction.",
    )

    args = parser.parse_args()

    require(
        1 <= args.rank <= 25,
        "Rank must belong to the frozen 25-candidate queue.",
    )

    inventory, archive, destination = prepare_rank(args.rank)

    print("=== V4-A SELECTED-MEMBER EXTRACTION ===")
    print("Rank:", args.rank)
    print("Match ID:", inventory["source_match_id"])
    print("Verified Archive:", inventory["source_archive_path"])
    print("Selected member:", inventory["selected_member"])
    print(
        "Declared selected-member bytes:",
        inventory["selected_member_declared_bytes"],
    )
    print(
        "Available disk:",
        f"{free_bytes(archive.parent) / 1024**3:.2f} GiB",
    )

    if not args.extract:
        print()
        print("MODE: READ-ONLY EXTRACTION PLAN")
        print("Demo extracted: NO")
        print("Technical eligibility: NOT EVALUATED")
        print("Model scoring: NONE")
        return

    # The actual extraction is intentionally an explicit operation.
    # Reconfirm that the original Archive still matches its recorded
    # identity immediately before publishing the extracted member.

    def verify_original_archive():
        require(
            archive.stat().st_size
            == inventory["source_archive_size_bytes"],
            "Source Archive size changed during extraction.",
        )

        require(
            sha256_file(archive)
            == inventory["source_archive_sha256"],
            "Source Archive SHA256 changed during extraction.",
        )

    result = stream_selected_member(
        command=[
            "bsdtar",
            "-xOf",
            str(archive),
            inventory["selected_member"],
        ],
        destination=destination,
        declared_bytes=inventory[
            "selected_member_declared_bytes"
        ],
        reserve_bytes=(
            MINIMUM_FREE_BYTES + SAFETY_RESERVE_BYTES
        ),
        verify_source=verify_original_archive,
    )

    print()
    print("SELECTED DEMO EXTRACTED")
    print(
        "Extracted Demo:",
        result["extracted_demo_path"].relative_to(ROOT),
    )
    print(
        "Extracted bytes:",
        result["extracted_demo_size_bytes"],
    )
    print(
        "Extracted SHA256:",
        result["extracted_demo_sha256"],
    )
    print("Demo Header verified: NO")
    print("Extraction Evidence created: NO")
    print("Technical eligibility: NOT EVALUATED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()

    except (
        ExtractionStop,
        RuntimeError,
        KeyError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(
            "EXTRACTION STOP:",
            exc,
            file=sys.stderr,
        )

        raise SystemExit(1)
