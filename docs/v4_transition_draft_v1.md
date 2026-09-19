# V3 → V4 Research Transition

**Project:** CS2 Tactical Intelligence Lab

**Status:** PROPOSAL — NOT FROZEN

**Proposed first investigation:** V4-A Team-Context-Aware
Spatial Forecasting

**Research scope:** Mirage, observer/replay information.

This document records the transition from the completed V3
experiment to a proposed V4 investigation.

It does not authorize new model training, confirmation scoring,
dataset replacement, or modification of frozen V3 artifacts.


## 1. Completed V3 evidence

V3 established a near-future bomb macro-zone forecasting task.

At an observation time t, models predicted the bomb's macro-zone
at t + 5 seconds and t + 10 seconds.

The frozen target space contained 15 Mirage macro-zones.

The development corpus contained 53 matches.

The independent confirmation corpus contained 20 later-period
matches selected using the frozen technical intake protocol.

The final comparison was B3_TABULAR_MAP_AWARE_V1 against
B2_ZONE_MARKOV_V1.

### Independent confirmation

| Metric | +5s | +10s |
|---|---:|---:|
| Eligible observations | 5861 | 5622 |
| B2 Log Loss | 0.754550 | 1.064823 |
| B3 Log Loss | 0.491641 | 0.839236 |
| B3 - B2 Log Loss | -0.262909 | -0.225587 |
| B3 Accuracy | 0.823580 | 0.705443 |
| B3 Macro F1 | 0.602756 | 0.452091 |

The paired match-bootstrap 95% percentile intervals for
B3 minus B2 were:

+5s: [-0.284017, -0.245411]

+10s: [-0.240819, -0.210696]

These measurements describe the frozen 20-match confirmation
sample. They do not establish generalization to other maps,
other populations, player-view input, or live deployment.


## 2. Limitations motivating further investigation

The V3 B3 feature representation contained 24 inputs:

- 15-dimensional one-hot current bomb macro-zone
- 2-dimensional current bomb state
- 3-dimensional current bomb XYZ
- 3-dimensional causal bomb velocity
- 1-dimensional bomb speed

It did not directly include the current spatial distribution
of all T-side and CT-side players.

V3's results also show uneven hard-classification performance
across macro-zones.

For B3 at +10 seconds:

- A_RAMP: support 215; recall approximately 0.070
- CONNECTOR: support 70; recall approximately 0.029
- MID_WINDOW: support 21; recall 0
- MARKET: support 3; recall 0
- CT_SPAWN: support 5; recall 0

The very low support for some zones prevents strong conclusions
about their population-level performance.

V3 confirmation data alone does not establish why these
specific errors occurred.

Possible explanatory factors, including missing team context,
class imbalance, route ambiguity, label granularity, and
insufficient examples, remain hypotheses to investigate.

V4 model selection must not use the exposed V3 confirmation
sample to optimize these weaknesses.


## 3. Proposed V4-A research question

Does current-time team spatial context provide additional
predictive information for future bomb macro-zone forecasting
beyond the frozen V3 B3 feature representation?

The initial target would remain:

P(bomb macro-zone at t + h | information available at t)

where h is separately 5 seconds or 10 seconds.

Maintaining the target permits a controlled comparison between
the existing V3 B3 baseline and a candidate model that includes
additional current-time team information.

This does not imply that V4 is intended to replace V3.


## 4. Candidate additional information

Potential candidate features include:

- Number of living T-side players in each current macro-zone
- Number of living CT-side players in each current macro-zone
- Current T-side spatial distribution relative to the bomb
- Current CT-side spatial distribution relative to the bomb
- Current teammate-to-bomb distances
- Current CT-to-bomb distances

These are candidates, not a frozen feature set.

Each feature requires a documented definition, information
availability check, missingness audit, and causal timestamp rule.

Player information observed after prediction time t is forbidden.

Future player positions, future bomb events, future plant sites,
future outcomes, and retrospective tactical labels are forbidden
as model inputs.

No feature may encode match identity or a future target label.


## 5. Experimental design proposal

### Baseline

The existing frozen V3 B3 prediction system will serve as
the reference model for the same +5s and +10s target.

Its trained artifacts, preprocessing, feature order, model
configuration, and frozen research records must remain unchanged.

### Candidate

A new V4 model may combine the existing 24 B3 features
with a separately defined team-context feature block.

A controlled first comparison should keep the modeling
family and training protocol as comparable as practical,
so that additional features are not confounded with a
simultaneous major model-architecture change.

### Development

The 53 V3 development matches may be examined as historical
development evidence for feasibility and exploratory analysis.

Any reuse must be documented explicitly.

Development evaluation must keep matches grouped and use
the same target rows for paired model comparisons.

If historical V3 out-of-fold predictions are used as a
reference, their provenance and compatibility with the
V4 development evaluation must be verified first.

### Independent confirmation

The V3 confirmation corpus has already been exposed by
formal evaluation.

It must not be presented as a fresh independent confirmation
set for a V4 model developed after inspecting V3 results.

A future formal V4 confirmation requires an independently
defined and frozen eligible corpus, disjoint from the
V2 and V3 confirmation corpora.

Acquisition and scoring protocols must be frozen before
accessing that corpus's outcomes.


## 6. Proposed evaluation framework

Primary candidate metric:

- Multiclass Log Loss, separately for +5s and +10s

Secondary candidate metrics:

- Accuracy
- Macro F1
- Multiclass Brier score
- Equal-match mean Log Loss
- Per-zone recall and support
- Match-level paired differences

An uncertainty analysis should account for observations
clustered within matches.

The exact metric implementation, resampling method,
seed, success criteria, exclusions, and model-selection rule
must be decided and frozen before the formal V4 experiment.

Confirmation results must not be used for further tuning.


## 7. Gate 0 — data and scientific feasibility

Before implementing V4 models:

1. Verify which current-time player fields are available
   from the existing parser and development demos.

2. Verify player positions, team membership, living status,
   timestamps, and macro-zone mapping compatibility.

3. Establish that every proposed feature can be computed
   using information available at or before t.

4. Audit missingness and ambiguity on development data.

5. Verify that candidate feature extraction does not change
   the existing frozen V3 target definitions.

6. Estimate memory, storage, and runtime requirements.

7. Record unsupported features instead of inventing
   fallback values or silently dropping observations.

Gate 0 is a feasibility audit, not a model evaluation.


## 8. Proposed implementation sequence

Gate 0:
Current-time team-state schema and causal availability audit.

Gate 1:
Draft and review the V4 target, feature, and data-role contracts.

Gate 2:
Freeze the exact first V4-A experiment and development
evaluation protocol.

Gate 3:
Implement a reproducible team-context feature extractor.

Gate 4:
Run grouped development comparisons on identical target rows.

Gate 5:
Perform error analysis and document what additional context
does and does not explain.

Gate 6:
Decide whether V4-A merits a separately acquired, frozen
independent confirmation experiment.

Gate 7:
If authorized by the frozen design, perform one formal
independent confirmation and close the research version.


## 9. Long-term direction

The long-term goal is observer/replay tactical intelligence.

Future investigations may consider:

- Multiple-player near-future spatial forecasting
- Team movement and rotation patterns
- Objective descriptions of tactical state transitions
- Displaying uncertainty in observer/replay interfaces
- Partial-observation/player-view inputs as a separate
  future research branch

These are not validated V4 capabilities.

Predicting a bomb macro-zone is not equivalent to predicting
an opponent's intentions or recommending a tactical action.


## 10. Non-negotiable research boundaries

V0, V1, V2, and V3 frozen records remain unchanged.

V2 D_CONFIRM and D_V3_CONFIRM are not new V4 confirmation data.

Do not regenerate or retune the frozen V3 B3 model.

Do not silently modify the frozen 15-zone mapping.

Do not run V4 confirmation scoring during exploratory development.

Do not create new raw demo copies simply to perform Gate 0.

Do not automatically delete existing demos or research artifacts.

Preserve the existing 12 GiB disk-space safety threshold.

Any new hypothesis, feature set, model, or target version must
have explicit provenance.

The next immediate task is Gate 0 feasibility, not training.


## 11. Evidence provenance

The transition is based on the following completed V3 records:

- docs/v3_research_design.md
- docs/v3_development_closure_v1.json
- docs/v3_confirm_results_v1.json
- docs/v3_closure_v1.md
- docs/v3_b3_feature_protocol_v1.json
- docs/v3_future_target_contract.json
- docs/v3_macro_zone_mapping_v1_frozen.json
- docs/v3_confirm_scoring_protocol_v1.json

This proposal must not be represented as a frozen experiment.
