#!/usr/bin/env python3
"""Read-only Rank 8 V4-A technical audit and one-time structured evidence.

Never creates an eligibility decision, final Manifest or predictions. It reuses
frozen Rank 1 audit helpers but does not invoke the Rank 1-specific runner.
"""
from __future__ import annotations

import argparse
import csv
import re
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_v4_a_confirm_rank1_technical_intake_v1 import (  # noqa: E402
    independent_occupancy, keys, snapshot_keys_from_team_keys,
)
from audit_v4_a_confirm_demo_identity_v1 import load_historical_hashes  # noqa: E402
from audit_v4_a_confirm_raw_tick_clock_v1 import measure_demo_raw_clock  # noqa: E402
from guard_v4_a_confirm_next_rank_v2 import verify_rank1  # noqa: E402
from manage_v4_a_confirm_intake_state_v1 import (  # noqa: E402
    load_initial_candidates, read_event, sha256_file,
)
from probe_v4_a_confirm_demo_source_v1 import read_frozen_queue  # noqa: E402
from cs2_tactical_intelligence.v4_a_confirm_demo_inputs_v1 import parse_demo_inputs  # noqa: E402
from cs2_tactical_intelligence.v4_a_confirm_extract_features_v1 import (  # noqa: E402
    extract_features, TARGET_KEY, MOTION_KEY, TEAM_KEY,
)
from cs2_tactical_intelligence import v4_a_confirm_team_context_v1 as team  # noqa: E402

AUDIT_VERSION = 'V4_A_CONFIRM_RANK8_TECHNICAL_EVIDENCE_V1'
RANK = 8
MATCH_ID = '2397990'
DEMO_SHA = 'b9559f1a4ebef72304e120154a4c6c51b4173975b4aaf981c8cf3a1b87982ebd'
ARCHIVE_SHA = '822eae4a040db1f3705bae411996c8e57cafd97efb26d1bf2533f3b26dc4ce93'
EXTRACTION_SHA = '5d36136cc218cd8183f524c167512593d5bedee1ef41d9f7dc50d08973087bca'
RANK2_SHA = '874496b96943e1cdac691494c9821fe5d4389ba4e173df0f048c30254aca0ead'
SOURCE_SHA = 'd9bf64ced7a35cec9a621f46180780a263c84683d281ca500101ff7b94d9bbd9'
MEMBER = 'upgrade-vs-illyrians-m1-mirage.dem'
STAGING = ROOT / 'data/raw/v4_a_confirm_download_staging/rank_08_2397990'
DEMO = STAGING / MEMBER
EXTRACTION = ROOT / 'docs/v4_a_confirm_intake_audits/rank_08_selected_demo_extraction_v1.json'
RANK2 = ROOT / 'docs/v4_a_confirm_intake_audits/rank_02_technical_eligibility_v1.json'
SOURCE = ROOT / 'docs/v4_a_confirm_intake_audits/rank_08_match_page_source_v1.html'
OUTPUT = ROOT / 'docs/v4_a_confirm_intake_audits/rank_08_data_integrity_v1.json'
FINAL = ROOT / 'docs/v4_a_confirm_manifest_v1.csv'
SOURCE_FILES = (
    'docs/v4_a_confirm_acquisition_protocol_v1_frozen.json',
    'docs/v4_a_confirm_feature_extraction_contract_v1_frozen.json',
    'docs/v4_a_confirm_scoring_protocol_v1_frozen.json',
    'docs/v4_a_confirm_acquisition_queue_v1.csv',
    'docs/v4_a_confirm_intake_initial_state_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_01_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_02_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_03_data_integrity_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_03_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_04_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_05_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_06_data_integrity_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_06_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_07_technical_eligibility_v1.json',
    'docs/v4_a_confirm_intake_events_v1/rank_08_2397990_archive_acquired.json',
    'docs/v4_a_confirm_intake_audits/rank_08_selected_demo_extraction_v1.json',
    'docs/v4_a_confirm_intake_audits/rank_08_match_page_source_v1.html',
    'docs/v3_dev_manifest.csv',
    'docs/v2_confirm_manifest.csv',
    'docs/v3_confirm_manifest.csv',
    'docs/v2_acquisition_queue.csv',
    'docs/v3_fresh_acquisition_queue_v2.csv',
    'docs/v3_confirm_acquisition_queue_v1.csv',
    'scripts/audit_v4_a_confirm_rank8_v1.py',
    'tests/test_v4_a_confirm_rank8_contract_v1.py',
    'scripts/audit_v4_a_confirm_rank1_technical_intake_v1.py',
    'scripts/audit_v4_a_confirm_raw_tick_clock_v1.py',
    'scripts/audit_v4_a_confirm_demo_identity_v1.py',
    'scripts/manage_v4_a_confirm_intake_state_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_demo_inputs_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_extract_features_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_target_rows_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_target_semantics_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_motion_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_team_context_v1.py',
    'src/cs2_tactical_intelligence/v4_a_confirm_feature_matrix_v1.py',
)


def require(value, message):
    if not value:
        raise RuntimeError('RANK8_AUDIT_STOP: ' + message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe file: {path}')
    return sha256_file(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_fingerprints():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def check_saved(payload):
    require(payload['version'] == AUDIT_VERSION and payload['record_type'] == 'DATA_INTEGRITY_EVIDENCE_ONLY', 'Evidence type mismatch')
    require(payload['candidate_rank'] == RANK and payload['source_match_id'] == MATCH_ID, 'Evidence rank mismatch')
    require(payload['demo_sha256'] == DEMO_SHA and payload['archive_sha256'] == ARCHIVE_SHA, 'Evidence archive/demo identity mismatch')
    require(payload['file_sha256'] == source_fingerprints(), 'Source file fingerprint mismatch')
    require(payload['technical_eligibility_decision'] == 'NOT_RECORDED' and payload['model_scoring_performed'] is False, 'Evidence crossed decision/scoring boundary')
    require(payload['final_manifest_created'] is False, 'Premature Manifest')
    require(payload['legacy_development_match_id_coverage'] == 'INCOMPLETE', 'Lost legacy coverage limitation')
    require(payload['date_evidence']['exact_gameplay_start_independently_measured'] is False, 'Unverified gameplay start claim')
    obs = payload['observation']
    validate_observation(obs)
    subprocess.run(['git', 'merge-base', '--is-ancestor', payload['audit_source_commit'], 'HEAD'], check=True, cwd=ROOT)


def validate_observation(obs):
    """Pure independent invariants; testable without processing a Demo."""
    require(obs['raw_clock']['within_v3_precedent_tolerance'] is True and obs['raw_clock']['valid_clock_intervals'] > 0, 'Raw clock not confirmed')
    a, b = obs['plus5_rows'], obs['plus10_rows']
    require(isinstance(a, int) and isinstance(b, int) and a > 0 and b > 0, 'Missing +5/+10 observations')
    require(obs['target_rows'] == obs['joined_rows'] == a + b, 'Target row accounting mismatch')
    require(obs['motion_rows'] > 0 and obs['team_context_rows'] == obs['independently_audited_snapshots'], 'Snapshot accounting mismatch')
    require(obs['historical_demo_sha256_overlap'] is False and obs['earlier_v4_demo_sha256_overlap'] is False, 'Demo duplicate detected')
    require(obs['available_historical_match_id_overlap'] is False, 'Historical Match ID overlap detected')
    require(obs['matrices'].keys() == {'5', '10'}, 'Unexpected matrix horizons')
    for horizon, count in (('5', a), ('10', b)):
        m = obs['matrices'][horizon]
        require(m == {'rows': count, 'control_shape': [count, 24], 'candidate_shape': [count, 56], 'dtype': 'float32', 'all_values_finite': True, 'candidate_first24_equal_control': True}, f'Matrix invariant failed for +{horizon}')


def verify_inputs():
    require(not FINAL.exists(), 'Final Manifest already exists')
    require(git('branch', '--show-current') != '', 'Detached HEAD is unsupported')
    require(git('rev-parse', 'HEAD') == git('rev-parse', 'origin/' + git('branch', '--show-current')), 'Current branch not synchronized with its remote')
    require(subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT).returncode == 0, 'Tracked worktree modified')
    require(subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode == 0, 'Staged changes present')
    require(not (STAGING / (MEMBER + '.partial')).exists(), 'Unresolved partial extraction')
    require(digest(EXTRACTION) == EXTRACTION_SHA and digest(SOURCE) == SOURCE_SHA, 'Pinned Rank 8 source or extraction evidence changed')
    require(digest(RANK2) == RANK2_SHA, 'Rank 2 decision changed')
    require(verify_rank1()['decision'] == 'ELIGIBLE', 'Rank 1 decision mismatch')
    rank2 = json.loads(RANK2.read_text())
    require(rank2['decision'] == 'EXCLUDED' and rank2['exclusion_reason'] == 'WRONG_MAP', 'Rank 2 prior decision mismatch')

    earlier_decisions = {}
    for prior_rank, prior_id, expected_decision in (
        (3, '2397693', 'ELIGIBLE'),
        (4, '2397832', 'EXCLUDED'),
        (5, '2397833', 'EXCLUDED'),
        (6, '2397834', 'ELIGIBLE'),
        (7, '2397989', 'EXCLUDED'),
    ):
        prior_path = ROOT / (
            'docs/v4_a_confirm_intake_audits/'
            f'rank_{prior_rank:02d}_technical_eligibility_v1.json'
        )
        require(
            prior_path.is_file() and not prior_path.is_symlink(),
            f'Rank {prior_rank} formal decision missing',
        )
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        require(
            prior['candidate_rank'] == prior_rank
            and prior['source_match_id'] == prior_id
            and prior['decision'] == expected_decision,
            f'Rank {prior_rank} formal decision mismatch',
        )
        if expected_decision == 'EXCLUDED':
            require(
                prior['exclusion_reason'] == 'WRONG_MAP',
                f'Rank {prior_rank} exclusion reason mismatch',
            )
        earlier_decisions[prior_rank] = prior

    require(
        DEMO_SHA != earlier_decisions[3]['verified_observations']['demo_sha256'],
        'Rank 8 duplicates the earlier eligible Rank 3 Demo',
    )
    require(
        DEMO_SHA != earlier_decisions[6]['verified_observations']['demo_sha256'],
        'Rank 8 duplicates the earlier eligible Rank 6 Demo',
    )
    rows, _ = read_frozen_queue()
    initial = load_initial_candidates()
    require(len(rows) == len(initial) == 25, 'Unexpected candidate count')
    require([int(r['candidate_rank']) for r in rows] == list(range(1, 26)), 'Queue order changed')
    require(rows[7]['source_match_id'] == MATCH_ID and initial[7]['source_match_id'] == MATCH_ID, 'Rank 8 queue identity mismatch')
    event = read_event(initial[7])  # full archive hash and acquisition record verification
    require(event and event['archive_sha256'] == ARCHIVE_SHA and event['status'] == 'ARCHIVE_ACQUIRED', 'Rank 8 Archive Event mismatch')
    evidence = json.loads(EXTRACTION.read_text())
    require(evidence['candidate_rank'] == RANK and evidence['source_match_id'] == MATCH_ID and evidence['extracted_demo_sha256'] == DEMO_SHA, 'Selected-member evidence mismatch')
    require(evidence['source_archive_sha256'] == ARCHIVE_SHA and evidence['actual_demo_header_map'] == 'de_mirage' and evidence['demo_header_map_verified'] is True, 'Selected-member map identity mismatch')
    require(evidence['raw_tick_clock_independently_measured'] is False and evidence['technical_eligibility_evaluated'] is False and evidence['model_scoring_performed'] is False, 'Selected-member evidence crossed boundary')
    require(evidence['archive_event_sha256'] == digest(ROOT / evidence['archive_event_path']), 'Extraction/Archive Event link mismatch')
    require(digest(DEMO) == DEMO_SHA and DEMO.stat().st_size == 268307845, 'Actual selected Demo bytes changed')
    parsed = BeautifulSoup(SOURCE.read_bytes(), 'html.parser')
    maps = [h.select_one('.mapname').get_text(' ', strip=True) for h in parsed.select('.mapholder')]
    require(maps == ['Mirage', 'Anubis', 'Inferno'] and 'Match over' in parsed.get_text(' ', strip=True), 'Match source map/completion differs')
    holders = parsed.select('.mapholder')
    require(
        len(holders) == 3
        and 'STATS' in holders[0].get_text(' ', strip=True)
        and 'STATS' in holders[1].get_text(' ', strip=True)
        and 'STATS' not in holders[2].get_text(' ', strip=True)
        and 'Inferno was left over' in parsed.get_text(' ', strip=True),
        'Played-Mirage and unused-decider evidence mismatch',
    )
    dates = parsed.select('.timeAndEvent .date')
    require(len(dates) == 1 and str(dates[0].get('data-unix', '')).isdecimal(), 'Match date is missing')
    date = datetime.fromtimestamp(int(dates[0]['data-unix']) / 1000, timezone.utc).date().isoformat()
    require(date == rows[7]['match_date'] == '2026-09-21', 'Match date differs from frozen queue')
    require(shutil.disk_usage(ROOT).free >= 12 * 1024**3, 'STORAGE_BLOCK: below frozen 12 GiB floor')
    print('Rank 1/2 decisions, frozen Rank 8 identity, acquisition and extraction SHA256: VERIFIED')
    print('HLTV completed-match calendar date:', date, '(not independently measured gameplay start)')
    print('Archive and Demo SHA256: VERIFIED')
    return initial[7], event, date


def known_historical_match_ids():
    """Use available exact IDs; never fabricate missing legacy Development IDs."""
    references = {}
    sources = (
        ('V2_CONFIRM_MANIFEST', 'docs/v2_confirm_manifest.csv', 'source_url'),
        ('V2_ACQUISITION_QUEUE', 'docs/v2_acquisition_queue.csv', 'source_url'),
        ('V3_CONFIRM_MANIFEST', 'docs/v3_confirm_manifest.csv', 'source_match_id'),
        ('V3_CONFIRM_ACQUISITION_QUEUE', 'docs/v3_confirm_acquisition_queue_v1.csv', 'match_id'),
        ('V3_FRESH_DEVELOPMENT_QUEUE', 'docs/v3_fresh_acquisition_queue_v2.csv', 'match_id'),
    )
    for label, relative, column in sources:
        with (ROOT / relative).open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            require(reader.fieldnames is not None and column in reader.fieldnames, f'Historical ID column missing: {relative}')
            for row in reader:
                value = str(row[column]).strip()
                if column == 'source_url':
                    match = re.search(r'/matches/(\d+)(?:/|$)', value)
                    require(match is not None, f'Invalid historical source URL: {relative}')
                    value = match.group(1)
                require(value.isdecimal(), f'Invalid historical Match ID: {relative}')
                references.setdefault(value, set()).add(label)
    return {key: sorted(values) for key, values in references.items()}


def audit():
    row, event, source_date = verify_inputs()
    historical = load_historical_hashes()
    require(DEMO_SHA not in historical, 'Historical Demo SHA256 overlap')
    known_ids = known_historical_match_ids()
    require(MATCH_ID not in known_ids, f'Available historical Match ID overlap: {known_ids.get(MATCH_ID)}')
    rank1 = json.loads((ROOT / 'docs/v4_a_confirm_intake_audits/rank_01_technical_eligibility_v1.json').read_text())
    require(DEMO_SHA != rank1['verified_observations']['demo_sha256'], 'Earlier V4 Demo SHA256 overlap')
    print('Historical Demo SHA256 overlap: NONE (103 pinned manifest entries)')
    print('Available historical Match ID references checked:', len(known_ids), '| overlap: NONE; legacy Development ID coverage remains INCOMPLETE')
    print('Measuring actual raw tick/game_time clock; this can take time...')
    clock = measure_demo_raw_clock(DEMO)
    require(clock['within_v3_precedent_tolerance'] is True and clock['valid_clock_intervals'] > 0, 'Raw tick clock failed frozen 64-tick requirement')
    print('Raw clock:', clock['measured_raw_ticks_per_second'], '| intervals:', clock['valid_clock_intervals'])
    print('Running frozen V4-A Demo parser, target/motion/team-context extraction...')
    inputs = parse_demo_inputs(DEMO, demo_filename=MEMBER, expected_sha256=DEMO_SHA)
    require(inputs.demo.header.get('map_name') == 'de_mirage', 'Parsed map mismatch')
    result = extract_features(inputs)
    targets, motion, context = result['targets'], result['motion'], result['team_context']
    joined, matrices = result['joined'], result['matrices']
    target_keys, motion_keys, context_keys = keys(targets, TARGET_KEY), keys(motion, MOTION_KEY), keys(context, TEAM_KEY)
    require(target_keys == keys(joined, TARGET_KEY) and targets.height == joined.height, 'Feature Join changed Target rows')
    require(motion_keys == {tuple(x[k] for k in MOTION_KEY) for x in targets.select(MOTION_KEY).iter_rows(named=True)}, 'Motion keys differ')
    require(context_keys == {tuple(x[k] for k in TEAM_KEY) for x in targets.select(TEAM_KEY).iter_rows(named=True)}, 'Team Context keys differ')
    require(all(x == 'RESOLVED' for x in motion['status'].to_list()), 'Unresolved causal motion')
    counts = Counter(int(h) for h in targets['horizon_sec'].to_list())
    require(set(counts) == {5, 10} and counts[5] > 0 and counts[10] > 0, 'Frozen horizon rows missing')
    snapshot_keys = snapshot_keys_from_team_keys(context_keys, MEMBER)
    snapshots = inputs.materialize_snapshots(snapshot_keys)
    require(set(snapshots) == snapshot_keys, 'Required current snapshots missing')
    production_vectors = {(int(x['round_num']), int(x['current_tick'])): [int(x[col]) for col in team.TEAM_COLUMNS] for x in context.iter_rows(named=True)}
    require(len(production_vectors) == context.height, 'Duplicate production Team Context keys')
    unknown_t = unknown_ct = audited = 0
    for key in sorted(snapshot_keys):
        checked = independent_occupancy(snapshots[key])
        actual = team.build_team_context(snapshots[key])
        require(checked['vector'] == actual['vector'] == production_vectors[key], f'Independent occupancy mismatch: {key}')
        require(checked['living_t'] == actual['living_t'] and checked['living_ct'] == actual['living_ct'], f'Living team-size mismatch: {key}')
        unknown_t += checked['unknown_t']
        unknown_ct += checked['unknown_ct']
        audited += 1
    require(audited == context.height and motion.height > 0, 'Audited snapshot counts mismatch')
    require(set(matrices) == {5, 10}, 'Unexpected matrix horizon set')
    matrix_obs = {}
    for h in (5, 10):
        m = matrices[h]
        c, v = m['control'], m['candidate']
        require(m['rows'].height == counts[h] and keys(m['rows'], TARGET_KEY) == {x for x in target_keys if x[3] == h}, f'+{h}s target matrix population mismatch')
        require(c.shape == (counts[h], 24) and v.shape == (counts[h], 56), f'+{h}s matrix shape mismatch')
        require(c.dtype == v.dtype == np.float32 and np.isfinite(c).all() and np.isfinite(v).all(), f'+{h}s matrix dtype/finite failure')
        require(np.array_equal(v[:, :24], c), f'+{h}s Candidate prefix differs from Control')
        matrix_obs[str(h)] = {'rows': counts[h], 'control_shape': [counts[h], 24], 'candidate_shape': [counts[h], 56], 'dtype': 'float32', 'all_values_finite': True, 'candidate_first24_equal_control': True}
    obs = {
        'raw_clock': clock, 'parsed_map': inputs.demo.header.get('map_name'),
        'target_rows': targets.height, 'plus5_rows': counts[5], 'plus10_rows': counts[10],
        'motion_rows': motion.height, 'team_context_rows': context.height,
        'independently_audited_snapshots': audited, 'joined_rows': joined.height,
        'unknown_place_t_observations': unknown_t, 'unknown_place_ct_observations': unknown_ct,
        'target_level_exclusion_entries': sum(result['exclusions'].values()),
        'round_level_exclusion_entries': sum(result['round_exclusions'].values()),
        'historical_demo_sha256_overlap': False, 'earlier_v4_demo_sha256_overlap': False,
        'available_historical_match_id_overlap': False,
        'matrices': matrix_obs,
    }
    validate_observation(obs)
    payload = {
        'version': AUDIT_VERSION, 'record_type': 'DATA_INTEGRITY_EVIDENCE_ONLY',
        'candidate_rank': RANK, 'source_match_id': MATCH_ID,
        'audit_source_commit': git('rev-parse', 'HEAD'),
        'file_sha256': source_fingerprints(),
        'archive_sha256': ARCHIVE_SHA, 'demo_sha256': DEMO_SHA,
        'date_evidence': {'completed_match_date_utc': source_date, 'source_snapshot_sha256': SOURCE_SHA, 'exact_gameplay_start_independently_measured': False},
        'observation': obs,
        'legacy_development_match_id_coverage': 'INCOMPLETE',
        'technical_eligibility_decision': 'NOT_RECORDED',
        'final_manifest_created': False, 'model_scoring_performed': False,
    }
    check_saved(payload)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Existing evidence: use --check; refusing overwrite')
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode()
    temporary = OUTPUT.with_name(OUTPUT.name + f'.tmp.{os.getpid()}')
    try:
        with temporary.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    print('RANK8_DATA_INTEGRITY_AUDIT_PASSED')
    print('Target +5/+10:', counts[5], counts[10], '| independently audited snapshots:', audited)
    print('Evidence file:', OUTPUT.relative_to(ROOT), '| SHA256:', digest(OUTPUT))
    print('Technical eligibility decision: NOT RECORDED | Model scoring: NONE')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audit', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        require(not FINAL.exists(), 'Final Manifest already exists')
        payload = json.loads(OUTPUT.read_text(encoding='utf-8'))
        check_saved(payload)
        require(digest(DEMO) == DEMO_SHA, 'Selected Demo SHA256 mismatch')
        print('RANK8_DATA_INTEGRITY_EVIDENCE_VERIFIED | no Demo reparse | no model scoring')
    elif OUTPUT.exists():
        require(OUTPUT.is_file() and not OUTPUT.is_symlink(), 'Existing evidence path invalid')
        check_saved(json.loads(OUTPUT.read_text()))
        print('RANK8_DATA_INTEGRITY_EVIDENCE_VERIFIED | existing record preserved | no reparse')
    else:
        audit()


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, RuntimeError, ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print('RANK 8 AUDIT STOP:', repr(exc), file=sys.stderr)
        raise SystemExit(1)
