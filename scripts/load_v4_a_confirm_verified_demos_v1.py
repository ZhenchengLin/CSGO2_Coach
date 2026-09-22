#!/usr/bin/env python3

"""
Unified verified-Demo entry point for V4-A Confirmation.

ZIP and RAR retain their own extraction-record semantics.
This module returns already verified Demo paths for read-only
identity and raw-clock audits.

It does not extract, select an eligible match, modify an
intake event, create a manifest, or perform model scoring.
"""

from pathlib import Path

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    STAGING_ROOT,
    rank_dir_name,
    require,
)


def load_verified_demos(row, event):
    rank = row["candidate_rank"]
    match_id = row["source_match_id"]

    require(
        event["candidate_rank"] == rank
        and event["source_match_id"] == match_id
        and event["status"] == "ARCHIVE_ACQUIRED",
        "VERIFIED_DEMO_STOP: candidate/event identity mismatch.",
    )

    archive_format = Path(event["archive_path"]).suffix.lower()

    if archive_format == ".zip":
        # Preserve the existing ZIP extraction and CRC contract.
        from extract_v4_a_confirm_zip_v1 import EXTRACT_DIR_NAME

        from audit_v4_a_confirm_demo_identity_v1 import (
            verify_extracted_files,
        )

        extraction_dir = (
            STAGING_ROOT / rank_dir_name(row) / EXTRACT_DIR_NAME
        )

        require(
            extraction_dir.is_dir()
            and not extraction_dir.is_symlink(),
            "VERIFIED_DEMO_STOP: verified ZIP extraction is missing.",
        )

        return verify_extracted_files(
            extraction_dir,
            event,
            expected_rank=rank,
            expected_match_id=match_id,
        )

    if archive_format == ".rar":
        # The current RAR record contract is explicitly Rank 1 only.
        # Future ranks must not inherit Rank 1's hardcoded identities.
        require(
            rank == 1 and match_id == "2397691",
            "VERIFIED_DEMO_STOP: RAR support is currently Rank 1 only.",
        )

        from read_v4_a_confirm_rar_extraction_record_v1 import (
            read_verified_rank1_record,
        )

        verified = read_verified_rank1_record()

        require(
            verified["candidate_rank"] == rank
            and verified["source_match_id"] == match_id
            and verified["source_archive_sha256"]
                == event["archive_sha256"]
            and verified["technical_eligibility_evaluated"] is False
            and verified["model_scoring_performed"] is False,
            "VERIFIED_DEMO_STOP: RAR verification identity mismatch.",
        )

        demo = ROOT / verified["demo_path"]

        require(
            demo.is_file() and not demo.is_symlink(),
            "VERIFIED_DEMO_STOP: verified RAR Demo is missing.",
        )

        return [{
            "path": demo.name,
            "absolute_path": demo,
            "size_bytes": verified["demo_size_bytes"],
            "sha256": verified["demo_sha256"],
            "is_demo_filename": demo.suffix.lower() == ".dem",
        }]

    raise RuntimeError(
        "VERIFIED_DEMO_STOP: unsupported Archive format "
        f"{archive_format!r}."
    )
