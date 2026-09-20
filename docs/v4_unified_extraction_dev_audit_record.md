# V4 Unified Extraction — Development Numerical Audit

## 1. Experiment Identity

Project: CS2 Tactical Intelligence

Experiment: V4 Unified Extraction

Implementation: 7B

Dataset: Frozen Development Dataset

Map: Mirage

Development matches: 53

Prediction horizons: +5s and +10s

Feature configurations:

- Control: 24D
- Candidate: 56D

Audit status: PASS

Audit completion date: 2026-09-20

This record documents the completed Development numerical
verification of the Unified Raw Demo Extraction pipeline.

It does not constitute a Confirmation Dataset evaluation.


## 2. Implementation Scope

The Unified Extraction pipeline integrates:

1. Shared raw-demo input extraction.
2. Frozen target-row reconstruction.
3. Causal motion feature reconstruction.
4. Current-time Team Context reconstruction.
5. Frozen 24D Control feature construction.
6. Frozen 56D Candidate feature construction.

The implementation is intended to reconstruct the frozen
Development features directly from the original demo inputs,
without using the frozen Development feature cache as the
source of the newly extracted feature values.


## 3. Development Audit Results

Development demos:

53 / 53 PASS

Total target rows:

29,065

Total causal motion rows:

14,869

Total Team Context snapshots:

14,869

UNKNOWN_PLACE snapshots:

1

Target, exclusions, motion and Team Context:

PASS

24D Control:

EXACT PASS

56D Candidate:

EXACT PASS

Final audit marker:

V4_UNIFIED_FULL_DEV_NUMERICAL_AUDIT_PASS


## 4. Verification Scope

The completed Development audit compares the reconstructed
Unified Extraction outputs against frozen Development
artifacts.

The audit covers:

- Target rows.
- Observation identities.
- Target exclusions.
- Causal motion features.
- Current-time Team Context features.
- 24D Control feature matrices.
- 56D Candidate feature matrices.

The complete Development replay passed across all 53 demos.

The numerical audit reported exact parity for both feature
configurations.


## 5. Experimental Boundaries

This implementation records Development verification only.

The following activities were not performed as part of
this Development audit:

- Confirmation Dataset evaluation.
- Model fitting.
- Model selection.
- Frozen model replacement.
- Frozen dataset modification.

No claim about Confirmation performance is made here.

Successful Development parity does not by itself establish
Confirmation Dataset performance or generalization.


## 6. Source Files

Unified demo inputs:

src/cs2_tactical_intelligence/v4_a_confirm_demo_inputs_v1.py

Unified target extraction:

src/cs2_tactical_intelligence/v4_a_confirm_target_rows_v1.py

Unified feature extraction:

src/cs2_tactical_intelligence/v4_a_confirm_extract_features_v1.py

Full Development numerical audit:

scripts/audit_v4_confirm_unified_dev_replay_v1.py


## 7. Implementation Decision

The Unified Extraction implementation passed the completed
53-demo Development numerical audit.

The verified implementation and its audit script are retained
as the next reproducible version of the V4 extraction pipeline.

This record does not alter any previously frozen experimental
contract.

Future Confirmation execution must continue to follow the
existing frozen Confirmation protocol.


## 8. Verification Evidence

The following SHA256 values identify the local source files
and audit log used for this implementation commit.


```text
6edbd2e09f678a7ec3222ad48dbf23e8f7c4efdc51580ff3a5816520fd5334b1  src/cs2_tactical_intelligence/v4_a_confirm_demo_inputs_v1.py
82440a3f662135f5bf959b78deb343989680e60584c5f50b08a437d179b967f4  src/cs2_tactical_intelligence/v4_a_confirm_target_rows_v1.py
856a499c3dd232ef27931efa6558631459e4223eb9677b5416d0062a8de32cbb  src/cs2_tactical_intelligence/v4_a_confirm_extract_features_v1.py
6a55d965191ad135d9b6a61e1ef6d4c691c42d2e9f95b68870ab7f7015a90f77  scripts/audit_v4_confirm_unified_dev_replay_v1.py
05b344b4ad21d758a8db6034da1340576f7d68097dbd7add412918005ba81eff  /tmp/csgo2_v4_7b1_full_dev_audit.log
```
