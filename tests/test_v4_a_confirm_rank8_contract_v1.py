"""Offline behavioral tests for audit invariants. No real Demo parsing.

These tests deliberately build examples independently of the production
extraction function and then corrupt one contract element at a time.
"""
import ast
import copy
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_v4_a_confirm_rank8_v1.py'
TREE = ast.parse(SCRIPT.read_text(encoding='utf-8'))
FUNCTIONS = {'require', 'validate_observation'}
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
assert len(NODES) == len(FUNCTIONS)
NAMESPACE = {}
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SCRIPT), 'exec'), NAMESPACE)
validate_observation = NAMESPACE['validate_observation']


def observation():
    return {
        'raw_clock': {'within_v3_precedent_tolerance': True, 'valid_clock_intervals': 20},
        'plus5_rows': 3, 'plus10_rows': 2, 'target_rows': 5,
        'joined_rows': 5, 'motion_rows': 3, 'team_context_rows': 3,
        'independently_audited_snapshots': 3,
        'historical_demo_sha256_overlap': False,
        'earlier_v4_demo_sha256_overlap': False,
        'available_historical_match_id_overlap': False,
        'matrices': {
            '5': {'rows': 3, 'control_shape': [3, 24], 'candidate_shape': [3, 56],
                  'dtype': 'float32', 'all_values_finite': True, 'candidate_first24_equal_control': True},
            '10': {'rows': 2, 'control_shape': [2, 24], 'candidate_shape': [2, 56],
                   'dtype': 'float32', 'all_values_finite': True, 'candidate_first24_equal_control': True},
        },
    }


class Rank8InvariantTests(unittest.TestCase):
    def test_independently_constructed_valid_example(self):
        validate_observation(observation())

    def test_zero_required_horizon_rejected(self):
        value = observation()
        value['plus10_rows'] = 0
        with self.assertRaisesRegex(RuntimeError, 'Missing \\+5/\\+10'):
            validate_observation(value)

    def test_target_rows_changed_by_join_rejected(self):
        value = observation()
        value['joined_rows'] -= 1
        with self.assertRaisesRegex(RuntimeError, 'Target row accounting'):
            validate_observation(value)

    def test_distinct_motion_and_team_snapshot_cardinalities_are_permitted(self):
        value = observation()
        value['motion_rows'] = 4  # two nominal current ticks can resolve to one snapshot
        validate_observation(value)

    def test_missing_snapshot_audit_rejected(self):
        value = observation()
        value['independently_audited_snapshots'] -= 1
        with self.assertRaisesRegex(RuntimeError, 'Snapshot accounting'):
            validate_observation(value)

    def test_wrong_feature_shape_rejected(self):
        value = observation()
        value['matrices']['5']['candidate_shape'] = [3, 55]
        with self.assertRaisesRegex(RuntimeError, 'Matrix invariant'):
            validate_observation(value)

    def test_historical_match_overlap_rejected(self):
        value = observation()
        value['available_historical_match_id_overlap'] = True
        with self.assertRaisesRegex(RuntimeError, 'Historical Match ID overlap'):
            validate_observation(value)

    def test_raw_clock_not_measured_rejected(self):
        value = observation()
        value['raw_clock']['valid_clock_intervals'] = 0
        with self.assertRaisesRegex(RuntimeError, 'Raw clock'):
            validate_observation(value)

    def test_missing_prefix_equality_rejected(self):
        value = observation()
        value['matrices']['10']['candidate_first24_equal_control'] = False
        with self.assertRaisesRegex(RuntimeError, 'Matrix invariant'):
            validate_observation(value)


if __name__ == '__main__':
    unittest.main()
