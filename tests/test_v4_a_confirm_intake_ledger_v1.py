import copy
import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import init_v4_a_confirm_intake_ledger_v1 as ledger


class InitialLedgerTests(unittest.TestCase):

    def test_initial_state_matches_frozen_queue(self):
        state = ledger.build_initial_state()

        ledger.validate_initial_state(state)

        self.assertEqual(
            state["candidate_count"],
            25,
        )

        self.assertEqual(
            state["required_eligible_matches"],
            20,
        )

        self.assertEqual(
            len(state["candidates"]),
            25,
        )

    def test_all_candidates_begin_pending(self):
        state = ledger.build_initial_state()

        self.assertEqual(
            {
                entry["initial_status"]
                for entry in state["candidates"]
            },
            {"PENDING"},
        )

    def test_modified_match_identity_is_rejected(self):
        state = ledger.build_initial_state()

        modified = copy.deepcopy(state)

        modified["candidates"][0][
            "source_match_id"
        ] = "9999999"

        with self.assertRaises(RuntimeError):
            ledger.validate_initial_state(modified)

    def test_premature_eligibility_is_rejected(self):
        state = ledger.build_initial_state()

        modified = copy.deepcopy(state)

        modified["candidates"][0][
            "initial_status"
        ] = "ELIGIBLE"

        with self.assertRaises(RuntimeError):
            ledger.validate_initial_state(modified)

    def test_initialization_is_single_use(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "initial_state.json"

            with patch.object(
                ledger,
                "LEDGER",
                path,
            ):
                ledger.initialize()
                ledger.check()

                with self.assertRaises(RuntimeError):
                    ledger.initialize()


if __name__ == "__main__":
    unittest.main()
