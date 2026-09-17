# V3 — Near-Future Tactical State Forecasting

## Status

DESIGN FROZEN FOR GATE 0

## Research question

Given the legitimate observer/replay state of a Mirage round at time t,
can we predict the bomb's tactical map region at t+5 seconds and t+10
seconds better than simple persistence and motion baselines?

## Primary targets

- bomb macro-zone at t + 5 s
- bomb macro-zone at t + 10 s

## Secondary targets

- T-side centroid macro-zone at t + 5 s
- T-side centroid macro-zone at t + 10 s

## Why V3

V0/V1/V2 answered the first question:

Can early-round state forecast eventual A Plant, B Plant, or No Plant?

V2 independent confirmation is complete.

V0 remains the primary model.

V3 therefore introduces a new capability rather than tuning the V2
confirmation experiment further.

The long-term project requires models of tactical movement, repeated
behavior, routes, rotations, and opponent tendencies.

Near-future spatial forecasting is an objective intermediate problem
because future positions are directly available from historical demos
without manually labeling concepts such as default, fake, execute, or
rotation.

## Scientific boundary

The V2 D_CONFIRM corpus is permanently sealed from:

- V3 training
- feature selection
- hyperparameter tuning
- model selection
- threshold selection

V3 will create new development and confirmation datasets.

## Map representation

V3 will use the Mirage navigation mesh.

A NavArea is treated as a graph node.

The initial V3 representation will create approximately 10-16 frozen
Mirage macro-zones from the nav graph.

Candidate tactical regions include:

- T Spawn
- A Ramp
- Palace
- A Site
- Top Mid
- Mid
- Connector / Jungle
- Underpass
- Short / Catwalk
- B Apartments
- B Site
- Market
- CT Spawn

These names are candidates only.

The final zone mapping must be visually audited and frozen before model
evaluation.

## Dependency rule

Do not silently upgrade Awpy.

The V0/V1/V2 parser contract used Awpy 2.0.2.

V3 Gate 0 will inspect and use the existing nav capabilities first.

## Baseline ladder

B0 — persistence

Predict future bomb zone = current bomb zone.

B1 — constant motion

Project current bomb/carrier motion forward and resolve the projected
position to a tactical zone.

B2 — zone transition baseline

Estimate:

P(zone_future | zone_current, horizon)

from development matches only.

B3 — tabular map-aware model

Use the frozen V0 state plus new map-semantic features.

Possible features:

- current bomb zone
- current T centroid zone
- T player counts per macro-zone
- CT player counts per macro-zone
- nav distance from bomb to tactical anchors
- nav distance from T centroid to tactical anchors
- current-to-previous zone transitions
- existing V0 geometry/combat features

Sequence models require evidence after B3.

## Evaluation

Primary:

- multiclass log loss

Secondary:

- accuracy
- macro F1
- per-zone recall
- top-2 accuracy
- calibration
- match-level log loss
- graph-distance error

Report +5 s and +10 s separately.

## Data strategy

D_V3_DEV:

- new chronological development corpus
- grouped by match
- used for baseline/model development

D_V3_CONFIRM:

- frozen only after V3 development decisions are finished
- evaluated once

The unused V2 reserve matches are not automatically a valid V3
confirmation set because they were already inspected during V2 intake.

## Gate sequence

Gate 0
Environment and Mirage nav audit.

Gate 1
Build and visually audit frozen Mirage macro-zones.

Gate 2
Build +5s / +10s target integrity audit.

Gate 3
Acquire and freeze D_V3_DEV.

Gate 4
Persistence / motion / transition baselines.

Gate 5
First learned map-aware model.

Gate 6
Error analysis and sequence-model decision.

Gate 7
Freeze and evaluate D_V3_CONFIRM once.

## Core invariant

V3 learns a new temporal/spatial prediction problem from new
development evidence.

V2 D_CONFIRM never becomes another tuning set.
