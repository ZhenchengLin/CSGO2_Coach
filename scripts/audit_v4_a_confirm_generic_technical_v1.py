#!/usr/bin/env python3
"""Generic V4-A Confirmation technical data-integrity audit.

The Auditor:
  * verifies the current frozen Rank and committed provenance;
  * checks date/map/Demo identity;
  * checks historical Demo SHA and available Match-ID separation;
  * independently measures the raw Demo clock;
  * executes the frozen V4-A feature/target pipeline;
  * independently audits living-player identity and occupancy;
  * validates +5/+10 target populations and 24D/56D matrices.

--audit:
    run everything read-only; do not create Evidence.

--record:
    run the same audit and create one immutable data-integrity
    Evidence record.

--check:
    verify an existing Evidence record without reparsing the Demo.

This module NEVER assigns ELIGIBLE/EXCLUDED, creates the final
Confirmation Manifest, loads frozen models, or calculates model metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "docs/v4_a_confirm_intake_audits"
EVENTS = ROOT / "docs/v4_a_confirm_intake_events_v1"
STAGING = ROOT / "data/raw/v4_a_confirm_download_staging"
FINAL = ROOT / "docs/v4_a_confirm_manifest_v1.csv"

VERSION = "V4_A_CONFIRM_GENERIC_TECHNICAL_EVIDENCE_V1"


class AuditStop(RuntimeError):
    """Stop instead of guessing when scientific evidence is uncertain."""


def require(condition, message):
    if not condition:
        raise AuditStop(message)


def sha256_file(path):
    path = Path(path)

    require(
        path.is_file() and not path.is_symlink(),
        f"Missing or unsafe file: {path}",
    )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_file_matches(path, commit="HEAD"):
    path = Path(path)

    require(
        path.is_file() and not path.is_symlink(),
        f"Missing or unsafe committed file: {path}",
    )

    relative = path.relative_to(ROOT).as_posix()

    try:
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
        )

    except subprocess.CalledProcessError as exc:
        raise AuditStop(
            f"Required file is not committed at {commit}: {relative}"
        ) from exc

    require(
        hashlib.sha256(committed).hexdigest()
        == sha256_file(path),
        f"Working file differs from {commit}: {relative}",
    )


def keys(frame, columns):
    selected = frame.select(columns)

    result = {
        tuple(row[column] for column in columns)
        for row in selected.iter_rows(named=True)
    }

    require(
        len(result) == frame.height,
        f"Duplicate keys for {columns}.",
    )

    return result


def validate_observation(obs):
    """Pure scientific invariants; suitable for offline unit testing."""

    clock = obs.get("raw_clock", {})

    require(
        clock.get("within_v3_precedent_tolerance") is True
        and clock.get("valid_clock_intervals", 0) > 0,
        "Raw clock requirement failed.",
    )

    require(
        obs.get("parsed_map") == "de_mirage",
        "Parsed Demo is not de_mirage.",
    )

    plus5 = obs.get("plus5_rows")
    plus10 = obs.get("plus10_rows")

    require(
        isinstance(plus5, int)
        and isinstance(plus10, int)
        and plus5 > 0
        and plus10 > 0,
        "Missing +5/+10 retained rows.",
    )

    require(
        obs.get("target_rows")
        == obs.get("joined_rows")
        == plus5 + plus10,
        "Target / Feature Join row accounting mismatch.",
    )

    require(
        isinstance(obs.get("motion_rows"), int)
        and obs["motion_rows"] > 0,
        "No causal motion rows.",
    )

    require(
        obs.get("team_context_rows")
        == obs.get("independently_audited_snapshots"),
        "Independent identity/occupancy audit count mismatch.",
    )

    require(
        obs.get("historical_demo_sha256_overlap") is False,
        "Historical Demo SHA256 overlap.",
    )

    require(
        obs.get("earlier_v4_demo_sha256_overlap") is False,
        "Earlier V4 Demo SHA256 overlap.",
    )

    require(
        obs.get("available_historical_match_id_overlap") is False,
        "Available historical Match ID overlap.",
    )

    matrices = obs.get("matrices", {})

    require(
        set(matrices) == {"5", "10"},
        "Unexpected matrix horizon population.",
    )

    for horizon, rows in (
        ("5", plus5),
        ("10", plus10),
    ):
        expected = {
            "rows": rows,
            "control_shape": [rows, 24],
            "candidate_shape": [rows, 56],
            "dtype": "float32",
            "all_values_finite": True,
            "candidate_first24_equal_control": True,
        }

        require(
            matrices[horizon] == expected,
            f"+{horizon}s matrix invariant failed.",
        )


def known_historical_match_ids():
    """Use only exact IDs actually available in historical records."""

    sources = (
        (
            "V2_CONFIRM_MANIFEST",
            "docs/v2_confirm_manifest.csv",
            "source_url",
        ),
        (
            "V2_ACQUISITION_QUEUE",
            "docs/v2_acquisition_queue.csv",
            "source_url",
        ),
        (
            "V3_CONFIRM_MANIFEST",
            "docs/v3_confirm_manifest.csv",
            "source_match_id",
        ),
        (
            "V3_CONFIRM_ACQUISITION_QUEUE",
            "docs/v3_confirm_acquisition_queue_v1.csv",
            "match_id",
        ),
        (
            "V3_FRESH_DEVELOPMENT_QUEUE",
            "docs/v3_fresh_acquisition_queue_v2.csv",
            "match_id",
        ),
    )

    references = {}

    for label, relative, column in sources:
        path = ROOT / relative

        with path.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            require(
                reader.fieldnames is not None
                and column in reader.fieldnames,
                f"Historical Match ID column unavailable: {relative}",
            )

            for row in reader:
                value = str(row[column]).strip()

                if column == "source_url":
                    match = re.search(
                        r"/matches/(\d+)(?:/|$)",
                        value,
                    )

                    require(
                        match is not None,
                        f"Invalid historical source URL: {relative}",
                    )

                    value = match.group(1)

                require(
                    value.isdecimal(),
                    f"Invalid historical Match ID: {relative}",
                )

                references.setdefault(
                    value,
                    set(),
                ).add(label)

    return {
        match_id: sorted(labels)
        for match_id, labels in references.items()
    }


def earlier_v4_demo_hashes(rank):
    """Collect exact Demo hashes recorded by prior formal decisions."""

    values = {}

    for earlier in range(1, rank):
        path = AUDITS / (
            f"rank_{earlier:02d}_technical_eligibility_v1.json"
        )

        require(
            path.is_file() and not path.is_symlink(),
            f"Prior Rank {earlier} Formal Decision missing.",
        )

        decision = json.loads(
            path.read_text(encoding="utf-8")
        )

        require(
            decision.get("candidate_rank") == earlier
            and decision.get("decision") in {"ELIGIBLE", "EXCLUDED"}
            and decision.get("model_scoring_performed") is False,
            f"Prior Rank {earlier} Decision is invalid.",
        )

        digest = decision.get(
            "verified_observations",
            {},
        ).get("demo_sha256")

        if digest is None:
            continue

        require(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"Prior Rank {earlier} Demo SHA256 is invalid.",
        )

        values.setdefault(digest, []).append(earlier)

    return values


def fingerprint_paths(rank, match_id):
    paths = [
        "docs/v4_a_confirm_acquisition_protocol_v1_frozen.json",
        "docs/v4_a_confirm_feature_extraction_contract_v1_frozen.json",
        "docs/v4_a_confirm_scoring_protocol_v1_frozen.json",
        "docs/v4_a_confirm_acquisition_queue_v1.csv",
        "docs/v4_a_confirm_intake_initial_state_v1.json",
        (
            "docs/v4_a_confirm_intake_events_v1/"
            f"rank_{rank:02d}_{match_id}_archive_acquired.json"
        ),
        (
            "docs/v4_a_confirm_intake_audits/"
            f"rank_{rank:02d}_selected_demo_extraction_v1.json"
        ),
        (
            "docs/v4_a_confirm_intake_audits/"
            f"rank_{rank:02d}_match_page_source_v1.html"
        ),
        "docs/v3_dev_manifest.csv",
        "docs/v2_confirm_manifest.csv",
        "docs/v3_confirm_manifest.csv",
        "docs/v2_acquisition_queue.csv",
        "docs/v3_fresh_acquisition_queue_v2.csv",
        "docs/v3_confirm_acquisition_queue_v1.csv",
        "scripts/audit_v4_a_confirm_generic_technical_v1.py",
        "tests/test_v4_a_confirm_generic_technical_v1.py",
        "scripts/audit_v4_a_confirm_rank1_technical_intake_v1.py",
        "scripts/audit_v4_a_confirm_raw_tick_clock_v1.py",
        "scripts/audit_v4_a_confirm_demo_identity_v1.py",
        "scripts/manage_v4_a_confirm_intake_state_v1.py",
        "scripts/plan_v4_a_confirm_intake_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_demo_inputs_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_extract_features_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_target_rows_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_target_semantics_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_motion_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_team_context_v1.py",
        "src/cs2_tactical_intelligence/v4_a_confirm_feature_matrix_v1.py",
    ]

    for earlier in range(1, rank):
        paths.append(
            "docs/v4_a_confirm_intake_audits/"
            f"rank_{earlier:02d}_technical_eligibility_v1.json"
        )

    return tuple(paths)


def current_fingerprints(rank, match_id):
    result = {}

    for relative in fingerprint_paths(rank, match_id):
        path = ROOT / relative

        require(
            path.is_file() and not path.is_symlink(),
            f"Required audit provenance missing: {relative}",
        )

        result[relative] = sha256_file(path)

    return result


def verify_saved_fingerprints(payload):
    commit = payload.get("audit_source_commit")

    require(
        isinstance(commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "Invalid audit source commit.",
    )

    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )

    fingerprints = payload.get("file_sha256")

    require(
        isinstance(fingerprints, dict)
        and fingerprints,
        "Saved source fingerprints unavailable.",
    )

    for relative, expected in fingerprints.items():
        require(
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "Unsafe saved fingerprint path.",
        )

        try:
            data = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"],
                cwd=ROOT,
            )

        except subprocess.CalledProcessError as exc:
            raise AuditStop(
                f"Saved source missing at audit commit: {relative}"
            ) from exc

        require(
            hashlib.sha256(data).hexdigest() == expected,
            f"Saved source fingerprint mismatch: {relative}",
        )


def check_saved(payload):
    require(
        payload.get("version") == VERSION
        and payload.get("record_type")
        == "DATA_INTEGRITY_EVIDENCE_ONLY",
        "Unexpected Technical Evidence schema.",
    )

    rank = payload.get("candidate_rank")
    match_id = payload.get("source_match_id")

    require(
        isinstance(rank, int)
        and 1 <= rank <= 25
        and isinstance(match_id, str)
        and match_id.isdecimal(),
        "Saved Technical Evidence identity is invalid.",
    )

    require(
        payload.get("technical_eligibility_decision")
        == "NOT_RECORDED"
        and payload.get("final_manifest_created") is False
        and payload.get("model_scoring_performed") is False,
        "Technical Evidence crossed the research boundary.",
    )

    require(
        payload.get("legacy_development_match_id_coverage")
        == "INCOMPLETE",
        "Historical Match ID limitation disappeared.",
    )

    date_evidence = payload.get("date_evidence", {})

    require(
        date_evidence.get(
            "exact_gameplay_start_independently_measured"
        ) is False,
        "Unsupported exact gameplay-start claim.",
    )

    validate_observation(payload.get("observation", {}))

    verify_saved_fingerprints(payload)


def verify_inputs(rank):
    from bs4 import BeautifulSoup

    sys.path.insert(0, str(ROOT / "scripts"))

    from manage_v4_a_confirm_intake_state_v1 import (
        load_initial_candidates,
        read_event,
    )

    from build_v4_a_confirm_queue_v1 import (
        load_frozen_rules,
    )

    from plan_v4_a_confirm_intake_v1 import (
        resolve_plan,
    )

    from probe_v4_a_confirm_demo_source_v1 import (
        read_frozen_queue,
    )

    require(
        not FINAL.exists() and not FINAL.is_symlink(),
        "Final Confirmation Manifest already exists.",
    )

    require(
        git("branch", "--show-current") == "main"
        and git("rev-parse", "HEAD")
        == git("rev-parse", "origin/main"),
        "Repository is not synchronized main.",
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

    plan = resolve_plan()

    require(
        plan["rank"] == rank
        and plan["stage"] == "TECHNICAL_AUDIT_PENDING",
        "Requested Rank is not at TECHNICAL_AUDIT_PENDING.",
    )

    rows, _ = read_frozen_queue()
    protocol, _ = load_frozen_rules()
    ledger = load_initial_candidates()

    require(
        len(rows) == len(ledger) == 25,
        "Unexpected frozen candidate population.",
    )

    row = rows[rank - 1]
    initial = ledger[rank - 1]

    match_id = row["source_match_id"]

    require(
        int(row["candidate_rank"]) == rank
        and initial["candidate_rank"] == rank
        and initial["source_match_id"] == match_id
        and plan["match_id"] == match_id,
        "Frozen Rank identity mismatch.",
    )

    source = AUDITS / (
        f"rank_{rank:02d}_match_page_source_v1.html"
    )

    extraction = AUDITS / (
        f"rank_{rank:02d}_selected_demo_extraction_v1.json"
    )

    output = AUDITS / (
        f"rank_{rank:02d}_data_integrity_v1.json"
    )

    decision = AUDITS / (
        f"rank_{rank:02d}_technical_eligibility_v1.json"
    )

    event_path = EVENTS / (
        f"rank_{rank:02d}_{match_id}_archive_acquired.json"
    )

    for path in (
        source,
        extraction,
        event_path,
    ):
        committed_file_matches(path)

    require(
        not output.exists() and not output.is_symlink(),
        "Technical Evidence already exists.",
    )

    require(
        not decision.exists() and not decision.is_symlink(),
        "Formal Technical Decision already exists.",
    )

    event = read_event(initial)

    require(
        event is not None
        and event["status"] == "ARCHIVE_ACQUIRED",
        "Verified Archive Event unavailable.",
    )

    evidence = json.loads(
        extraction.read_text(encoding="utf-8")
    )

    require(
        evidence.get("candidate_rank") == rank
        and evidence.get("source_match_id") == match_id
        and evidence.get("record_type")
        == "SELECTED_MEMBER_EXTRACTION_EVIDENCE"
        and evidence.get("actual_demo_header_map") == "de_mirage"
        and evidence.get("demo_header_map_verified") is True
        and evidence.get("technical_eligibility_evaluated") is False
        and evidence.get("model_scoring_performed") is False,
        "Selected-Demo Extraction Evidence is invalid.",
    )

    require(
        evidence.get("source_archive_sha256")
        == event["archive_sha256"]
        and evidence.get("archive_event_sha256")
        == sha256_file(event_path)
        and evidence.get("source_snapshot_sha256")
        == sha256_file(source),
        "Extraction provenance linkage mismatch.",
    )

    demo_relative = evidence.get("extracted_demo_path")

    require(
        isinstance(demo_relative, str),
        "Selected Demo path unavailable.",
    )

    relative = Path(demo_relative)

    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("data", "raw"),
        "Unsafe selected Demo path.",
    )

    demo = ROOT / relative

    require(
        demo.is_file()
        and not demo.is_symlink()
        and demo.stat().st_size
        == evidence["extracted_demo_size_bytes"]
        and sha256_file(demo)
        == evidence["extracted_demo_sha256"],
        "Selected Demo identity mismatch.",
    )

    staging = demo.parent

    require(
        not list(staging.glob("*.partial.*")),
        "Unresolved extraction partial exists.",
    )

    parsed = BeautifulSoup(
        source.read_bytes(),
        "html.parser",
    )

    text = parsed.get_text(" ", strip=True)

    require(
        "Match over" in text,
        "Completed-match source state not verified.",
    )

    holders = parsed.select(".mapholder")

    mirage_holders = []

    for holder in holders:
        element = holder.select_one(".mapname")

        if element is None:
            continue

        if element.get_text(" ", strip=True) == "Mirage":
            mirage_holders.append(holder)

    require(
        len(mirage_holders) == 1
        and "STATS" in mirage_holders[0].get_text(
            " ",
            strip=True,
        ),
        "HLTV source does not establish played Mirage.",
    )

    dates = parsed.select(".timeAndEvent .date")

    require(
        len(dates) == 1
        and str(dates[0].get("data-unix", "")).isdecimal(),
        "Completed-match date unavailable.",
    )

    source_date = datetime.fromtimestamp(
        int(dates[0]["data-unix"]) / 1000,
        timezone.utc,
    ).date().isoformat()

    require(
        source_date == row["match_date"],
        "Completed-match date differs from frozen Queue.",
    )

    boundary = protocol["temporal_boundary"][
        "earliest_allowed_match_date"
    ]

    require(
        source_date >= boundary,
        "Match date violates frozen temporal boundary.",
    )

    require(
        shutil.disk_usage(ROOT).free >= 12 * 1024**3,
        "STORAGE_BLOCK: below frozen 12 GiB floor.",
    )

    return {
        "rank": rank,
        "match_id": match_id,
        "row": row,
        "event": event,
        "source": source,
        "source_date": source_date,
        "extraction": extraction,
        "evidence": evidence,
        "demo": demo,
        "member": evidence["selected_member"],
        "demo_sha": evidence["extracted_demo_sha256"],
        "archive_sha": evidence["source_archive_sha256"],
        "output": output,
    }


def run_audit(rank):
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))

    from audit_v4_a_confirm_demo_identity_v1 import (
        load_historical_hashes,
    )

    from audit_v4_a_confirm_raw_tick_clock_v1 import (
        measure_demo_raw_clock,
    )

    from audit_v4_a_confirm_rank1_technical_intake_v1 import (
        independent_occupancy,
        snapshot_keys_from_team_keys,
    )

    from cs2_tactical_intelligence.v4_a_confirm_demo_inputs_v1 import (
        parse_demo_inputs,
    )

    from cs2_tactical_intelligence.v4_a_confirm_extract_features_v1 import (
        extract_features,
        TARGET_KEY,
        MOTION_KEY,
        TEAM_KEY,
    )

    from cs2_tactical_intelligence import (
        v4_a_confirm_team_context_v1 as team,
    )

    context = verify_inputs(rank)

    match_id = context["match_id"]
    demo_sha = context["demo_sha"]

    print("=== V4-A GENERIC TECHNICAL AUDIT ===")
    print("Candidate Rank:", rank)
    print("Match ID:", match_id)
    print("Completed-match date:", context["source_date"])
    print("Actual Demo map: de_mirage")
    print("Demo SHA256:", demo_sha)
    print()

    historical = load_historical_hashes()

    require(
        demo_sha not in historical,
        "Historical Demo SHA256 overlap.",
    )

    historical_ids = known_historical_match_ids()

    require(
        match_id not in historical_ids,
        "Historical Match ID overlap: "
        + repr(historical_ids.get(match_id)),
    )

    prior_hashes = earlier_v4_demo_hashes(rank)

    require(
        demo_sha not in prior_hashes,
        "Earlier V4 Demo SHA256 overlap: "
        + repr(prior_hashes.get(demo_sha)),
    )

    print(
        "Historical Demo SHA256 overlap: NONE "
        f"({len(historical)} pinned historical demos)"
    )

    print(
        "Available historical Match ID overlap: NONE "
        f"({len(historical_ids)} exact IDs checked)"
    )

    print("Earlier V4 Demo SHA256 overlap: NONE")
    print(
        "Legacy Development Match-ID coverage: INCOMPLETE "
        "(preserved limitation)"
    )

    print()
    print("Measuring actual raw tick/game_time clock...")

    clock = measure_demo_raw_clock(
        context["demo"]
    )

    require(
        clock["within_v3_precedent_tolerance"] is True
        and clock["valid_clock_intervals"] > 0,
        "Raw tick clock failed frozen requirement.",
    )

    print(
        "Measured raw ticks/second:",
        clock["measured_raw_ticks_per_second"],
    )

    print(
        "Valid clock intervals:",
        clock["valid_clock_intervals"],
    )

    print()
    print(
        "Running frozen V4-A target / motion / "
        "team-context / matrix pipeline..."
    )

    inputs = parse_demo_inputs(
        context["demo"],
        demo_filename=context["member"],
        expected_sha256=demo_sha,
    )

    require(
        inputs.demo.header.get("map_name") == "de_mirage",
        "Parsed Demo map changed.",
    )

    result = extract_features(inputs)

    targets = result["targets"]
    motion = result["motion"]
    team_context = result["team_context"]
    joined = result["joined"]
    matrices = result["matrices"]

    target_keys = keys(
        targets,
        TARGET_KEY,
    )

    motion_keys = keys(
        motion,
        MOTION_KEY,
    )

    context_keys = keys(
        team_context,
        TEAM_KEY,
    )

    require(
        target_keys == keys(joined, TARGET_KEY)
        and targets.height == joined.height,
        "Feature Join changed Target rows.",
    )

    expected_motion_keys = {
        tuple(row[column] for column in MOTION_KEY)
        for row in targets.select(MOTION_KEY).iter_rows(
            named=True
        )
    }

    require(
        motion_keys == expected_motion_keys,
        "Motion keys differ from frozen Target population.",
    )

    expected_context_keys = {
        tuple(row[column] for column in TEAM_KEY)
        for row in targets.select(TEAM_KEY).iter_rows(
            named=True
        )
    }

    require(
        context_keys == expected_context_keys,
        "Team Context keys differ from frozen Target population.",
    )

    require(
        all(
            status == "RESOLVED"
            for status in motion["status"].to_list()
        ),
        "Unresolved causal motion row.",
    )

    counts = Counter(
        int(value)
        for value in targets["horizon_sec"].to_list()
    )

    require(
        set(counts) == {5, 10}
        and counts[5] > 0
        and counts[10] > 0,
        "Frozen +5/+10 horizon population missing.",
    )

    snapshot_keys = snapshot_keys_from_team_keys(
        context_keys,
        context["member"],
    )

    snapshots = inputs.materialize_snapshots(
        snapshot_keys
    )

    require(
        set(snapshots) == snapshot_keys,
        "Required current-time snapshots unavailable.",
    )

    production_vectors = {
        (
            int(row["round_num"]),
            int(row["current_tick"]),
        ): [
            int(row[column])
            for column in team.TEAM_COLUMNS
        ]
        for row in team_context.iter_rows(named=True)
    }

    require(
        len(production_vectors) == team_context.height,
        "Duplicate production Team Context keys.",
    )

    unknown_t = 0
    unknown_ct = 0
    audited = 0

    for snapshot_key in sorted(snapshot_keys):
        independent = independent_occupancy(
            snapshots[snapshot_key]
        )

        production = team.build_team_context(
            snapshots[snapshot_key]
        )

        require(
            independent["vector"]
            == production["vector"]
            == production_vectors[snapshot_key],
            f"Independent occupancy mismatch: {snapshot_key}",
        )

        require(
            independent["living_t"]
            == production["living_t"]
            and independent["living_ct"]
            == production["living_ct"],
            f"Living-player identity/count mismatch: {snapshot_key}",
        )

        unknown_t += independent["unknown_t"]
        unknown_ct += independent["unknown_ct"]
        audited += 1

    require(
        audited == team_context.height,
        "Independent snapshot population mismatch.",
    )

    require(
        set(matrices) == {5, 10},
        "Unexpected matrix horizons.",
    )

    matrix_observation = {}

    for horizon in (5, 10):
        bundle = matrices[horizon]

        control = bundle["control"]
        candidate = bundle["candidate"]

        expected_rows = counts[horizon]

        horizon_keys = {
            key
            for key in target_keys
            if key[3] == horizon
        }

        require(
            bundle["rows"].height == expected_rows
            and keys(bundle["rows"], TARGET_KEY)
            == horizon_keys,
            f"+{horizon}s matrix row population mismatch.",
        )

        require(
            control.shape == (expected_rows, 24)
            and candidate.shape == (expected_rows, 56),
            f"+{horizon}s feature matrix shape mismatch.",
        )

        require(
            control.dtype == np.float32
            and candidate.dtype == np.float32,
            f"+{horizon}s matrix dtype mismatch.",
        )

        require(
            np.isfinite(control).all()
            and np.isfinite(candidate).all(),
            f"+{horizon}s matrix contains non-finite values.",
        )

        require(
            np.array_equal(
                candidate[:, :24],
                control,
            ),
            f"+{horizon}s Candidate first 24 dimensions "
            "differ from Control.",
        )

        matrix_observation[str(horizon)] = {
            "rows": expected_rows,
            "control_shape": [expected_rows, 24],
            "candidate_shape": [expected_rows, 56],
            "dtype": "float32",
            "all_values_finite": True,
            "candidate_first24_equal_control": True,
        }

    observation = {
        "raw_clock": clock,
        "parsed_map": inputs.demo.header.get("map_name"),
        "target_rows": targets.height,
        "plus5_rows": counts[5],
        "plus10_rows": counts[10],
        "motion_rows": motion.height,
        "team_context_rows": team_context.height,
        "independently_audited_snapshots": audited,
        "joined_rows": joined.height,
        "unknown_place_t_observations": unknown_t,
        "unknown_place_ct_observations": unknown_ct,
        "target_level_exclusion_entries": sum(
            result["exclusions"].values()
        ),
        "round_level_exclusion_entries": sum(
            result["round_exclusions"].values()
        ),
        "historical_demo_sha256_overlap": False,
        "earlier_v4_demo_sha256_overlap": False,
        "available_historical_match_id_overlap": False,
        "matrices": matrix_observation,
    }

    validate_observation(observation)

    payload = {
        "version": VERSION,
        "record_type": "DATA_INTEGRITY_EVIDENCE_ONLY",
        "candidate_rank": rank,
        "source_match_id": match_id,
        "audit_source_commit": git("rev-parse", "HEAD"),
        "file_sha256": current_fingerprints(
            rank,
            match_id,
        ),
        "archive_sha256": context["archive_sha"],
        "demo_sha256": demo_sha,
        "date_evidence": {
            "completed_match_date_utc": context["source_date"],
            "source_snapshot_sha256": sha256_file(
                context["source"]
            ),
            "exact_gameplay_start_independently_measured": False,
        },
        "observation": observation,
        "legacy_development_match_id_coverage": "INCOMPLETE",
        "technical_eligibility_decision": "NOT_RECORDED",
        "final_manifest_created": False,
        "model_scoring_performed": False,
    }

    check_saved(payload)

    print()
    print("=== TECHNICAL AUDIT OBSERVATION PASSED ===")
    print(
        "Retained +5 rows:",
        counts[5],
    )
    print(
        "Retained +10 rows:",
        counts[10],
    )
    print(
        "Target rows:",
        targets.height,
    )
    print(
        "Motion rows:",
        motion.height,
    )
    print(
        "Team Context rows:",
        team_context.height,
    )
    print(
        "Independently audited snapshots:",
        audited,
    )
    print(
        "Unknown T / CT place observations:",
        unknown_t,
        "/",
        unknown_ct,
    )
    print(
        "+5 Control/Candidate:",
        list(matrices[5]["control"].shape),
        "/",
        list(matrices[5]["candidate"].shape),
    )
    print(
        "+10 Control/Candidate:",
        list(matrices[10]["control"].shape),
        "/",
        list(matrices[10]["candidate"].shape),
    )

    return context, payload


def write_once(path, payload):
    require(
        path.parent.is_dir()
        and not path.parent.is_symlink(),
        "Unsafe Technical Evidence directory.",
    )

    require(
        not path.exists() and not path.is_symlink(),
        "Technical Evidence already exists.",
    )

    data = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    require(
        not temporary.exists() and not temporary.is_symlink(),
        "Temporary Technical Evidence path already exists.",
    )

    created = False

    try:
        with temporary.open("xb") as handle:
            created = True

            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.link(
            temporary,
            path,
        )

    finally:
        if created and temporary.exists():
            temporary.unlink()

    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        required=True,
    )

    modes = parser.add_mutually_exclusive_group(
        required=True
    )

    modes.add_argument(
        "--audit",
        action="store_true",
        help="Run the complete Technical Audit read-only.",
    )

    modes.add_argument(
        "--record",
        action="store_true",
        help="Run and create immutable data-integrity Evidence.",
    )

    modes.add_argument(
        "--check",
        action="store_true",
        help="Verify existing Evidence without Demo reparse.",
    )

    args = parser.parse_args()

    require(
        1 <= args.rank <= 25,
        "Rank must belong to the frozen Queue.",
    )

    output = AUDITS / (
        f"rank_{args.rank:02d}_data_integrity_v1.json"
    )

    if args.check:
        require(
            output.is_file()
            and not output.is_symlink(),
            "Technical Evidence does not exist.",
        )

        payload = json.loads(
            output.read_text(encoding="utf-8")
        )

        require(
            payload.get("candidate_rank") == args.rank,
            "Technical Evidence Rank mismatch.",
        )

        check_saved(payload)

        print(
            "TECHNICAL EVIDENCE VERIFIED:",
            output.relative_to(ROOT),
        )
        print(
            "Evidence SHA256:",
            sha256_file(output),
        )
        print("Demo reparsed: NO")
        print("Technical eligibility decision: NOT RECORDED")
        print("Model scoring: NONE")
        return

    context, payload = run_audit(
        args.rank
    )

    if args.audit:
        print()
        print("MODE: READ-ONLY TECHNICAL AUDIT")
        print("Technical Evidence created: NO")
        print("Technical eligibility decision: NOT RECORDED")
        print("Model scoring: NONE")
        return

    record_sha = write_once(
        context["output"],
        payload,
    )

    print()
    print("TECHNICAL DATA-INTEGRITY EVIDENCE: CREATED")
    print(
        "Evidence:",
        context["output"].relative_to(ROOT),
    )
    print(
        "Evidence SHA256:",
        record_sha,
    )
    print("Technical eligibility decision: NOT RECORDED")
    print("Model scoring: NONE")


if __name__ == "__main__":
    try:
        main()

    except (
        AuditStop,
        AssertionError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:

        print(
            "GENERIC TECHNICAL AUDIT STOP:",
            repr(exc),
            file=sys.stderr,
        )

        raise SystemExit(1)
