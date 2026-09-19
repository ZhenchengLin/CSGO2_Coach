# V3 independent confirmation — closure

**Status:** Formal independent confirmation completed; V3 research version closed.

## Fixed experiment

- First 20 technically eligible Mirage matches in the frozen queue.
- Development: 53 separate matches. Confirmation: 20 later-period matches.
- Frozen B0/B1/B2/B3 and 15-zone target; +5s and +10s separately.
- Primary comparison: B3 minus B2 pooled multiclass Log Loss.
- 10,000 paired match-bootstrap replicates per horizon, seed 42.

## Formal confirmation measurements

| Horizon | Rows | B0 LL | B1 LL | B2 LL | B3 LL | B3 − B2 | 95% match-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---|
| +5s | 5861 | 1.202183 | 1.413096 | 0.754550 | 0.491641 | -0.262909 | [-0.284017, -0.245411] |
| +10s | 5622 | 1.720789 | 1.967531 | 1.064823 | 0.839236 | -0.225587 | [-0.240819, -0.210696] |

## Interpretation

- +5s: B3 had lower observed Log Loss, with the percentile interval below zero.
- +10s: B3 had lower observed Log Loss, with the percentile interval below zero.

## Boundaries and limitations

- Repeated observations within a match are correlated; uncertainty resamples matches, not individual rows.
- Findings are limited to the frozen Mirage task and this later-period 20-match sample.
- No confirmation-based tuning, refitting or candidate replacement.
- Methodological changes belong to a new research version.

## Immutable provenance

- Manifest SHA256: `68411147301c9e37a4649ad0b4ceac61f7b87a01493f1cff3a692b16dc8de3b7`
- Model freeze SHA256: `65bf09ec3e3cbeaacfb5ff4bb28a504d927fbcc55898b40e1413153ef11dba85`
- Scoring protocol SHA256: `362a5c598cde2bcb91724994330cc72751adaf1f7084f1e7bb5d793c8996a203`
- Evaluator SHA256: `70153c6174857a7f9ce6b885c50addeb032e41affe04c257c80ae5e443c84ba6`
- Predictions SHA256: `fea88e0ed0e6012d133b7dbaaa25f6bec92e0f6c8acf3ab1f4a05bdc57f69ea9`
- Causal motion SHA256: `d5be08d72373d8e124cfc13c4d358af08041ef44407f569cc8987f15b88d759a`

Detailed secondary metrics and per-zone support/recall are stored in `docs/v3_confirm_results_v1.json`.
