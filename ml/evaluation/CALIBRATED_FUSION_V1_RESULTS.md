# Calibrated Raw-Score + Agreement Fusion V1 Results

Date: 2026-09-16

Decision: **C — CALIBRATION DOES NOT GENERALIZE**

Production: **unchanged; uncapped RRF60**

## Protocol and integrity

The experiment uses 72 frozen calibration queries from three sources, then evaluates score
calibration once on 32 queries from two source-disjoint development-validation sources. The
validation set has 28 positive and four negative queries. The checksum-frozen Hybrid Fusion
Holdout V1 contains 34 queries from two further sources; zero were used for tuning or candidate
evaluation. Acceptance V1/V2 contributed no training examples. All source hashes are disjoint.

Features and the gate were fixed before validation. Each modality compares raw score, reciprocal
rank, top and next-result margins, robust query-local normalization, and query length through
deterministic L2 logistic calibration. Leave-one-source-out selection uses calibration data only.
The predeclared gate requires the selected model to be no worse than rank-only on both AUC and
Brier for both modalities.

## Raw-score evidence

| Modality | Relevant raw score, median (IQR) | Irrelevant raw score, median (IQR) | Raw AUC | Rank AUC |
| --- | ---: | ---: | ---: | ---: |
| Visual | 0.2725 (0.2561–0.2992) | 0.2550 (0.2412–0.2708) | 0.6863 | 0.7012 |
| Speech | 4.5467 (3.5930–6.7403) | 2.7928 (1.9042–3.7443) | 0.7788 | 0.7644 |

Visual raw scores do not add ranking information beyond rank on this validation set. Their AUC
is lower and varies sharply by source: 0.5624 on `design-free-software-talk` and 0.7912 on
`human-software-extensions-talk`. Speech raw scores show modest univariate separation beyond
rank, but the fitted query-relative calibration fails across sources. Speech raw-score AUC also
varies by source, from 0.7603 to 0.8054. There were no 1–6-token validation queries, so no claim
is made for short-query calibration.

Relevant candidates occur earlier than irrelevant candidates: Visual median rank is 11 versus
27, and Speech median rank is 6 versus 19. Relevant next-result margins are larger in both paths
(Visual 0.00110 versus 0.00037; Speech 0.2584 versus 0.0480), but these signals did not produce a
source-stable probability model.

## Calibration generalization

| Modality | Calibration-selected method | Calibration LOSO AUC / Brier | Validation selected AUC / Brier | Validation rank-only AUC / Brier | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Visual | raw-score logistic | 0.8085 / 0.0729 | 0.6863 / 0.1002 | 0.7012 / 0.0901 | FAIL |
| Speech | query-relative logistic | 0.7334 / 0.1780 | 0.6454 / 0.1721 | 0.7644 / 0.0873 | FAIL |

Visual loses 0.0149 AUC and worsens Brier by 0.0101 relative to rank-only. Speech loses 0.1190
AUC and worsens Brier by 0.0848. This is a failure of source-disjoint generalization, despite the
raw Speech score's univariate AUC. The diagnostic artifact loads round-trip and is 3,168 bytes,
but it is marked rejected and contributes zero production bytes or memory. Batched diagnostic
prediction over every validation candidate measured 0.0227/0.0233 ms median/p95 for Visual and
0.0280/0.0293 ms for Speech; these timings are not candidate-fusion latency.

## Production baseline on validation

| Metric | Count | Rate |
| --- | ---: | ---: |
| Useful Top-1 | 10/28 | 35.71% |
| Useful Top-3 | 16/28 | 57.14% |
| Useful Top-5 | 20/28 | 71.43% |
| MRR@5 | — | 0.4827 |

Category Top-5 is 7/8 Visual, 6/11 Speech, and 7/9 Multimodal. These figures reproduce the
uncapped RRF60 development baseline and are not a new production claim.

## Candidate and decision

No fusion candidate was constructed. Therefore a candidate formula, Top-K metrics, MRR, fusion
latency, memory impact, recovered failures, and regressions are **not applicable**. The measured
failure counts remain five irrelevant-consensus failures, zero strong-single-modality recoveries,
zero recoveries, zero regressions, and zero net gain because the experiment stopped before
candidate evaluation.

The exact next milestone is to collect at least three additional source-disjoint calibration
sources with complete Top-50 Visual and Speech traces, then freeze a new validation split before
reconsidering confidence-aware fusion. A second-stage reranker is not justified by this result.
The protected holdout remains untouched for a future fully frozen candidate.
