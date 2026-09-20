# V4-A Confirmation Metadata — Discovery Provenance V1

## 1. Research identity

Project: CS2 Tactical Intelligence

Experiment: V4-A Confirmation

Dataset: D_V4_A_CONFIRM

Frozen acquisition protocol:

docs/v4_a_confirm_acquisition_protocol_v1_frozen.json

This record preserves the initial metadata discovery
artifacts before final Confirmation Acquisition Queue
construction.

It is not a frozen candidate queue or final manifest.


## 2. Discovery population

The initial metadata discovery consists of two batches.

Batch 1:

11 records

Batch 2:

14 records

Total:

25 distinct source match IDs

All 25 records have staged dates of 2026-09-20.

The discovery batches passed local structural validation
and Match ID uniqueness checks.

These results do not establish independent source-page
verification, temporal eligibility, or technical eligibility.


## 3. Discovery methodology

The first 11 records were collected from an initial
inspection of HLTV match listings and pages.

A second discovery pass added 14 further match records.

The two batches were not produced by a documented
exhaustive enumeration of a predeclared source population.

The initial selection was not proven to be the first
25 matches in any complete HLTV listing.

Some match results, map information, and technical
availability information were inspected during the
metadata discovery and source-review process.

Therefore, the current 25-record set must not be
represented as a fully outcome-blind prospective sample.

This limitation must remain visible in any later
confirmation dataset provenance.

No model predictions or model performance metrics
were used in the recorded metadata collection process.


## 4. Outstanding source verification

Independent verification of all 25 source records
has not been completed.

The following issues remain relevant:

- Consistent source match-date convention.
- Match ID and source-page metadata consistency.
- Match participant identities for scheduled matches.
- Documentation of the candidate sampling frame.
- Consistency between the discovered corpus and
  the frozen acquisition protocol.

The staged BESTIA vs ShindeN record has a known
date discrepancy between HLTV source views.

No candidate may be silently removed or replaced
to improve expected model performance or obtain
a preferred target distribution.


## 5. Technical eligibility

Technical eligibility has not been established.

In particular, the following have not yet been
verified for all candidates:

- Availability of an eligible Mirage raw demo.
- Required raw tick rate and player telemetry.
- Frozen player identity and occupancy checks.
- Frozen target-row eligibility.
- Historical demo and match-identity non-overlap.

Technical eligibility must follow the frozen
acquisition protocol.

The current discovery records are not confirmation
training data, scored observations, or an eligible
confirmation sample.


## 6. Experimental boundary

The final 25-candidate acquisition queue has
not been created or committed.

The final confirmation manifest has not been created.

No new confirmation raw demos have been downloaded
by this implementation.

No confirmation model fitting or scoring has been
performed by this implementation.

All previously frozen feature contracts, model
artifacts, and Development datasets remain unchanged.


## 7. Next action

Complete independent source verification and document
the governing date convention and sampling-frame
implementation.

If these requirements cannot be reconciled with the
existing frozen protocol, record and commit an explicit
versioned protocol amendment before acquiring raw demos.

Only create and freeze the final candidate queue
after those requirements have been resolved.

Do not describe the present provenance commit as
the final acquisition queue freeze.


## 8. Artifact identities

The following SHA256 values identify the preserved
discovery artifacts.

```text
31bce9d778e50df22ed3a12725a95108dcc50451a9357833876b3d5f3fccb15c  scripts/prepare_v4_a_confirm_queue_v1.py
2780ead25154b8adc1b475abc0643a7933912045cc9fdf68a44eaf904939d03d  scripts/audit_v4_a_confirm_metadata_discovery_v1.py
efb93bcc0dbdf5359225e81cb55af77db15139e2804d11ed071168b5b21ee32d  docs/v4_a_confirm_metadata_staging_snapshot_v1.csv
83efe9c2581badfca3f61e649ab60fe2ceac3036046255ff012720249cdc4b49  docs/v4_a_confirm_metadata_discovery_batch2_v1.csv
28117918ed327f4bf4609ab059ff0dd3536f712af113e5785a4ff3cd0ad2ddaf  docs/v4_a_confirm_metadata_review_v1.md
bd06024e2e99268fab80e8f971a2ac9548f6a38e1e71ebac9122e54a65a9dcb1  docs/v4_a_confirm_metadata_source_review_v1.json
5e0e9b536776153ac2182e755ff5e762a42525b3f6e4a7ba0a4f6057200a4f98  docs/v4_a_confirm_metadata_verification_ledger_v1.csv
a12e2999878d6ed9851fd663e5130ea75f2293f91852ab167ee5ce15803fae2d  docs/v4_a_confirm_metadata_freeze_readiness_v1.json
```
