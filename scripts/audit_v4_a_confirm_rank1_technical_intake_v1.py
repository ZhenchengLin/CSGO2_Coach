#!/usr/bin/env python3

"""
V4-A Confirmation — Rank 1 reproducible Technical Intake Audit.

This runner:
  - verifies the frozen Rank 1 acquisition and RAR extraction;
  - checks historical Demo SHA256 separation;
  - independently measures the raw clock;
  - executes the existing frozen V4-A extraction;
  - independently checks player identity and occupancy;
  - checks target, motion, team-context and matrix integrity.

It DOES NOT:
  - assign ELIGIBLE or EXCLUDED;
  - resolve unavailable historical Match IDs by assumption;
  - establish the actual match date from the scheduled date;
  - remove retained rows;
  - create the final confirmation manifest;
  - load or score a model;
  - write an evidence record in this version.
"""

from __future__ import annotations

import argparse
import json
import sys

from collections import Counter
from pathlib import Path

import numpy as np

from audit_v4_a_confirm_demo_identity_v1 import (
    load_historical_hashes,
)

from audit_v4_a_confirm_raw_tick_clock_v1 import (
    measure_demo_raw_clock,
)

from load_v4_a_confirm_verified_demos_v1 import (
    load_verified_demos,
)

from manage_v4_a_confirm_intake_state_v1 import (
    ROOT,
    load_initial_candidates,
    read_event,
    require,
    sha256_file,
)

from read_v4_a_confirm_rar_extraction_record_v1 import (
    read_verified_rank1_record,
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


RANK = 1
MATCH_ID = "2397691"

EXPECTED_DEMO_SHA256 = (
    "848a9e9d8b0593e3684a19f681d4f9c3902a9490ae530fe20667cdb7cc5843b1"
)

EXPECTED_RECORD_SHA256 = (
    "1c048712a2720c1556f81bddbed9523e91cf2bc37d7eb5d20157b7fd1ff80a4d"
)

PLAN_PATH = ROOT / (
    "docs/v4_a_confirm_rank1_technical_intake_plan_v1.json"
)

RECORD_PATH = ROOT / (
    "data/raw/v4_a_confirm_download_staging/"
    "rank_01_2397691/rar_selected_extraction_record_v1.json"
)

FINAL_MANIFEST = ROOT / (
    "docs/v4_a_confirm_manifest_v1.csv"
)


def keys(frame, columns):
    """Return all keys, rejecting duplicate rows."""

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


def snapshot_keys_from_team_keys(team_keys, demo_filename):
    """
    Convert full Team Context keys:

        (demo_filename, round_num, current_tick)

    into the snapshot keys required by materialize_snapshots():

        (round_num, current_tick)

    Reject unexpected Demo identities and key collisions.
    """

    snapshot_keys = set()

    for key in team_keys:
        require(
            isinstance(key, tuple)
            and len(key) == 3,
            "Unexpected Team Context key shape.",
        )

        name, round_num, current_tick = key

        require(
            name == demo_filename,
            "Team Context key belongs to another Demo.",
        )

        snapshot_keys.add((
            int(round_num),
            int(current_tick),
        ))

    require(
        len(snapshot_keys) == len(team_keys),
        "Multiple Team Context keys map to one snapshot.",
    )

    return snapshot_keys


def independent_occupancy(players):
    """
    Independently recompute one snapshot's identity and
    32D occupancy constraints.

    Does not call the production Team Context Builder.
    """

    require(
        bool(players),
        "Missing current-time player snapshot.",
    )

    counts = {
        "t": Counter(),
        "ct": Counter(),
    }

    unknown = {"t": 0, "ct": 0}
    living = {"t": 0, "ct": 0}
    seen_ids = set()

    for player in players:

        side = player["side"]
        health = player["health"]

        if side not in ("t", "ct"):
            continue

        if health is None or health <= 0:
            continue

        raw_id = player["steamid"]

        require(
            raw_id is not None,
            "Living player has missing Steam ID.",
        )

        text = str(raw_id).strip()

        require(
            text.isdecimal() and int(text) > 0,
            f"Living player has invalid Steam ID: {raw_id!r}",
        )

        steamid = int(text)

        require(
            steamid not in seen_ids,
            f"Duplicate living Steam ID: {steamid}",
        )

        seen_ids.add(steamid)
        living[side] += 1

        zone = team.PLACE_TO_ZONE.get(player["place"])

        if zone is None:
            unknown[side] += 1
        else:
            require(
                zone in team.ZONES,
                f"Unexpected macro-zone: {zone!r}",
            )

            counts[side][zone] += 1

    require(
        1 <= living["t"] <= 5
        and 1 <= living["ct"] <= 5,
        f"Invalid living team sizes: {living}",
    )

    vector = (
        [counts["t"][zone] for zone in team.ZONES]
        + [counts["ct"][zone] for zone in team.ZONES]
        + [unknown["t"], unknown["ct"]]
    )

    require(
        len(vector) == 32
        and all(isinstance(value, int) and value >= 0
                for value in vector),
        "Invalid independent occupancy vector.",
    )

    require(
        sum(vector[:15]) + vector[30] == living["t"],
        "Independent T occupancy conservation failed.",
    )

    require(
        sum(vector[15:30]) + vector[31] == living["ct"],
        "Independent CT occupancy conservation failed.",
    )

    require(
        len(seen_ids) == living["t"] + living["ct"],
        "Independent identity count mismatch.",
    )

    return {
        "vector": vector,
        "living_t": living["t"],
        "living_ct": living["ct"],
        "unknown_t": unknown["t"],
        "unknown_ct": unknown["ct"],
    }


def verify_plan():
    require(
        PLAN_PATH.is_file() and not PLAN_PATH.is_symlink(),
        "Rank 1 Technical Intake Plan is missing.",
    )

    plan = json.loads(
        PLAN_PATH.read_text(encoding="utf-8")
    )

    require(
        plan["version"]
        == "V4_A_CONFIRM_RANK1_TECHNICAL_INTAKE_PLAN_V1"
        and plan["candidate_rank"] == RANK
        and plan["source_match_id"] == MATCH_ID
        and plan["record_type"]
        == "AUDIT_PLAN_NOT_ELIGIBILITY_DECISION",
        "Unexpected Rank 1 audit plan.",
    )

    require(
        plan["verified_demo"]["sha256"]
        == EXPECTED_DEMO_SHA256
        and plan["verified_demo"]["extraction_record_sha256"]
        == EXPECTED_RECORD_SHA256,
        "Audit Plan Demo identity mismatch.",
    )

    require(
        plan["historical_identity_coverage"][
            "legacy_development_match_id_coverage"
        ] == "INCOMPLETE",
        "Historical identity limitation was removed from plan.",
    )

    require(
        plan["scientific_boundaries"][
            "technical_eligibility_evaluated"
        ] is False
        and plan["scientific_boundaries"][
            "confirmation_model_prediction_performed"
        ] is False,
        "Audit Plan crossed the scientific boundary.",
    )

    return plan


def audit_rank1():

    require(
        not FINAL_MANIFEST.exists(),
        "Final Confirmation Manifest already exists.",
    )

    plan = verify_plan()

    print("=== V4-A RANK 1 TECHNICAL INTAKE AUDIT ===")
    print("Audit Plan: VERIFIED")
    print("Audit Plan SHA256:", sha256_file(PLAN_PATH))

    print()
    print("=== ACQUISITION AND EXTRACTION PROVENANCE ===")

    row = load_initial_candidates()[0]

    require(
        row["candidate_rank"] == RANK
        and row["source_match_id"] == MATCH_ID,
        "Frozen candidate mismatch.",
    )

    event = read_event(row)

    require(
        event is not None
        and event["status"] == "ARCHIVE_ACQUIRED",
        "Verified acquisition event is required.",
    )

    require(
        RECORD_PATH.is_file()
        and not RECORD_PATH.is_symlink()
        and sha256_file(RECORD_PATH) == EXPECTED_RECORD_SHA256,
        "RAR Extraction Record identity mismatch.",
    )

    rar = read_verified_rank1_record()

    verified = load_verified_demos(row, event)

    require(
        len(verified) == 1
        and verified[0]["is_demo_filename"] is True,
        "Unexpected verified Demo population.",
    )

    item = verified[0]

    require(
        item["sha256"] == EXPECTED_DEMO_SHA256
        and item["sha256"] == rar["demo_sha256"]
        and item["absolute_path"].is_file(),
        "RAR Demo identity mismatch.",
    )

    require(
        rar["source_archive_sha256"] == event["archive_sha256"],
        "Archive provenance mismatch.",
    )

    print("Candidate Rank:", RANK)
    print("Match ID:", MATCH_ID)
    print("Verified Demo:", item["path"])
    print("Demo SHA256:", item["sha256"])
    print("RAR Extraction Record: VERIFIED")

    print()
    print("=== HISTORICAL SHA256 SEPARATION ===")

    historical = load_historical_hashes()

    overlap = historical.get(item["sha256"], [])

    require(
        not overlap,
        f"Historical Demo SHA256 overlap: {overlap}",
    )

    print("Historical Demo SHA256 overlap: NONE")
    print(
        "Legacy Development Match ID coverage:",
        "INCOMPLETE — not silently treated as verified",
    )

    print()
    print("=== INDEPENDENT RAW CLOCK ===")

    clock = measure_demo_raw_clock(item["absolute_path"])

    require(
        clock["within_v3_precedent_tolerance"] is True,
        "Measured raw tick clock is outside frozen tolerance.",
    )

    require(
        clock["valid_clock_intervals"] > 0,
        "No usable raw clock observations.",
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
    print("=== FROZEN V4-A EXTRACTION ===")

    inputs = parse_demo_inputs(
        item["absolute_path"],
        demo_filename=item["path"],
        expected_sha256=item["sha256"],
    )

    require(
        inputs.demo.header.get("map_name") == "de_mirage",
        "Demo is not Mirage.",
    )

    result = extract_features(inputs)

    targets = result["targets"]
    motion = result["motion"]
    context = result["team_context"]
    joined = result["joined"]
    matrices = result["matrices"]

    target_keys = keys(targets, TARGET_KEY)
    motion_keys = keys(motion, MOTION_KEY)
    context_keys = keys(context, TEAM_KEY)

    require(
        target_keys == keys(joined, TARGET_KEY)
        and joined.height == targets.height,
        "Feature Join changed retained Target Rows.",
    )

    require(
        motion_keys == {
            tuple(row[column] for column in MOTION_KEY)
            for row in targets.select(MOTION_KEY).iter_rows(named=True)
        },
        "Motion keys do not match retained observations.",
    )

    require(
        context_keys == {
            tuple(row[column] for column in TEAM_KEY)
            for row in targets.select(TEAM_KEY).iter_rows(named=True)
        },
        "Team Context keys do not match retained snapshots.",
    )

    require(
        all(status == "RESOLVED"
            for status in motion["status"].to_list()),
        "Unresolved causal motion in retained rows.",
    )

    counts = Counter(
        int(horizon)
        for horizon in targets["horizon_sec"].to_list()
    )

    require(
        set(counts) == {5, 10}
        and counts[5] > 0
        and counts[10] > 0,
        "One or more required horizons have no retained rows.",
    )

    print("Target Rows:", targets.height)
    print("+5s Rows:", counts[5])
    print("+10s Rows:", counts[10])
    print("Motion Rows:", motion.height)
    print("Team Context Rows:", context.height)
    print("Joined Rows:", joined.height)

    print()
    print("=== INDEPENDENT PLAYER IDENTITY / OCCUPANCY ===")

    snapshot_keys = snapshot_keys_from_team_keys(
        context_keys,
        item["path"],
    )

    snapshots = inputs.materialize_snapshots(snapshot_keys)

    require(
        set(snapshots) == snapshot_keys,
        "Required current-time snapshots are missing.",
    )

    audited = 0
    unknown_t = 0
    unknown_ct = 0

    production_vectors = {
        (
            int(record["round_num"]),
            int(record["current_tick"]),
        ): [
            int(record[column])
            for column in team.TEAM_COLUMNS
        ]
        for record in context.iter_rows(named=True)
    }

    require(
        len(production_vectors) == context.height,
        "Duplicate production Team Context keys.",
    )

    for key in sorted(snapshot_keys):

        independently_computed = independent_occupancy(
            snapshots[key]
        )

        production = team.build_team_context(
            snapshots[key]
        )

        require(
            independently_computed["vector"]
            == production["vector"]
            == production_vectors[key],
            f"Occupancy mismatch at snapshot {key}.",
        )

        require(
            independently_computed["living_t"]
            == production["living_t"]
            and independently_computed["living_ct"]
            == production["living_ct"],
            f"Living team-size mismatch at snapshot {key}.",
        )

        unknown_t += independently_computed["unknown_t"]
        unknown_ct += independently_computed["unknown_ct"]

        audited += 1

    require(
        audited == context.height,
        "Not all Team Context snapshots were audited.",
    )

    print("Current-time Snapshots independently audited:", audited)
    print("Identity / occupancy reconciliation: PASS")
    print("T UNKNOWN_PLACE observations:", unknown_t)
    print("CT UNKNOWN_PLACE observations:", unknown_ct)

    print()
    print("=== 24D / 56D MATRIX INTEGRITY ===")

    require(
        set(matrices) == {5, 10},
        "Unexpected matrix horizons.",
    )

    for horizon in (5, 10):

        frame = matrices[horizon]["rows"]
        control = matrices[horizon]["control"]
        candidate = matrices[horizon]["candidate"]

        require(
            frame.height == counts[horizon]
            and keys(frame, TARGET_KEY) == {
                key for key in target_keys if key[3] == horizon
            },
            f"+{horizon}s matrix target population mismatch.",
        )

        require(
            control.shape == (counts[horizon], 24)
            and candidate.shape == (counts[horizon], 56),
            f"+{horizon}s matrix shape mismatch.",
        )

        require(
            control.dtype == np.float32
            and candidate.dtype == np.float32
            and bool(np.isfinite(control).all())
            and bool(np.isfinite(candidate).all())
            and bool(np.array_equal(candidate[:, :24], control)),
            f"+{horizon}s matrix integrity mismatch.",
        )

        print(
            f"+{horizon}s: "
            f"{counts[horizon]} rows | "
            f"Control {control.shape} | "
            f"Candidate {candidate.shape} | PASS"
        )

    print()
    print("=== AUDIT SUMMARY ===")

    print(
        "Target-level exclusion entries:",
        sum(result["exclusions"].values()),
    )

    print(
        "Round-level exclusion entries:",
        sum(result["round_exclusions"].values()),
    )

    print("Rows removed by Feature Join:", 0)
    print("Legacy Development Match ID coverage: INCOMPLETE")
    print("Actual match date from independent source: NOT VERIFIED")
    print("Formal technical eligibility decision: NOT RECORDED")
    print("Final Confirmation Manifest: NOT CREATED")
    print("Model scoring: NONE")

    print()
    print("STATUS: RANK1_DATA_INTEGRITY_AUDIT_PASSED")
    print("8B-7H-2: READ-ONLY TECHNICAL INTAKE AUDIT COMPLETE")

    # Returned only after every preceding audit assertion passes.
    # No target labels, model predictions or model metrics are included.
    return {
        "version": "V4_A_RANK1_DATA_INTEGRITY_OBSERVATION_V1",
        "record_type": "DATA_INTEGRITY_NOT_ELIGIBILITY_DECISION",
        "status": "RANK1_DATA_INTEGRITY_AUDIT_PASSED",
        "candidate_rank": RANK,
        "source_match_id": MATCH_ID,
        "frozen_scheduled_match_date": row["frozen_match_date"],
        "archive_sha256": event["archive_sha256"],
        "extraction_record_sha256": sha256_file(RECORD_PATH),
        "demo_filename": item["path"],
        "demo_sha256": item["sha256"],
        "parsed_map": inputs.demo.header.get("map_name"),
        "raw_clock": clock,
        "target_rows": targets.height,
        "plus5_rows": counts[5],
        "plus10_rows": counts[10],
        "causal_motion_rows": motion.height,
        "team_context_rows": context.height,
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
        "matrices": {
            str(horizon): {
                "rows": matrices[horizon]["rows"].height,
                "control_shape": list(
                    matrices[horizon]["control"].shape
                ),
                "candidate_shape": list(
                    matrices[horizon]["candidate"].shape
                ),
                "dtype": "float32",
                "all_values_finite": True,
                "candidate_first24_equal_control": True,
            }
            for horizon in (5, 10)
        },
        "legacy_development_match_id_coverage": "INCOMPLETE",
        "actual_match_date_independently_verified": False,
        "technical_eligibility_evaluated": False,
        "model_scoring_performed": False,
        "final_manifest_created": False,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    require(
        args.rank == RANK,
        "Only frozen Rank 1 is supported by this runner.",
    )

    audit_rank1()


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
    ) as exc:
        print(
            "TECHNICAL INTAKE AUDIT STOP:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
