# Human-grounded English AUTO router results

Decision **C — ROUTER FAILS HUMAN-GROUNDED TEST**. The fixed character router passes validation at 90.00% but reaches only 60.00% on two unseen real-video sources. It misses overall, Visual-recall, and Speech-recall gates. Production remains on the current 54-parameter router and no formal second frozen run was performed.

## Data and protocol

Ten independent real English videos span explainers, education, ordinary cooking instruction, software UX, and technical presentations. Their source pages, open licenses, attribution, sizes, durations, and SHA-256 checksums are recorded in `sources_v1.json`. Review covered 1,669 actual five-second frames and 1,755 local Whisper segments. No media, frames, transcripts, weights, or experimental artifact are tracked.

The 360 annotations are exactly balanced: 120 Visual, 120 Speech, and 120 Hybrid. Train/validation/frozen-test contain 180/120/60 queries from 4/4/2 disjoint videos. All intervals and rationales were written from inspected frames and transcript evidence. Twenty annotations were flagged as ambiguous before evaluation. The protected failed-acceptance evidence was never opened or used; its named source is absent.

The first draft test exposed a query-form confound and was retired into validation before the final candidate evaluation. Two newly acquired, never-evaluated videos then became the frozen test. The final test deliberately crosses query form with route: it contains questions in every class, commands in Visual and Hybrid, short noun-phrase Speech searches, indirect phrasing, technical vocabulary, and strict multimodal constraints.

## Current production baseline

| Metric | Result |
|---|---:|
| Accuracy | 48.33% |
| Visual precision / recall / F1 | 41.67% / 100.00% / 58.82% |
| Speech precision / recall / F1 | 0.00% / 0.00% / 0.00% |
| Hybrid precision / recall / F1 | 100.00% / 45.00% / 62.07% |
| Median / p95 latency | 0.0322 / 0.0373 ms |

Confusion matrix, rows actual and columns predicted:

| Actual | Visual | Speech | Hybrid |
|---|---:|---:|---:|
| Visual | 20 | 0 | 0 |
| Speech | 20 | 0 | 0 |
| Hybrid | 8 | 3 | 9 |

The production router treats all 20 unseen Speech searches as Visual and misses 11 of 20 Hybrid requests. Its high softmax scores do not signal correctness.

## Fixed character candidate

The primary model uses character 3–5 gram TF-IDF and deterministic linear softmax with the previously justified configuration. Training uses only the four train videos. Validation makes no frozen-test-driven choice and leaves confidence fallback disabled. The optional word TF-IDF sanity baseline reaches 95.83% validation accuracy; the fixed character candidate reaches 90.00% validation accuracy with 85.00%/87.50%/97.50% Visual/Speech/Hybrid recall.

| Metric | Frozen result | Gate |
|---|---:|---:|
| Accuracy | 60.00% | ≥90% — FAIL |
| Visual precision / recall / F1 | 41.67% / 50.00% / 45.45% | recall ≥85% — FAIL |
| Speech precision / recall / F1 | 66.67% / 40.00% / 50.00% | recall ≥85% — FAIL |
| Hybrid precision / recall / F1 | 75.00% / 90.00% / 81.82% | recall ≥85% — PASS |
| Median latency | 1.6260 ms | <5 ms — PASS |
| p95 latency | 1.9283 ms | <10 ms — PASS |
| Artifact size | 402,292 bytes | reported |
| Estimated model/vocabulary memory | 159,207 bytes | reported |

Confusion matrix, rows actual and columns predicted:

| Actual | Visual | Speech | Hybrid |
|---|---:|---:|---:|
| Visual | 10 | 4 | 6 |
| Speech | 12 | 8 | 0 |
| Hybrid | 2 | 0 | 18 |

The frozen artifact SHA-256 is `4c1f3febee73275bcb25c110ff160375e77f9a27cde069079f5252ef7585e038`. It has 13,230 weights including class intercepts, uses local NumPy inference, and enables no confidence fallback. Because quality gates fail, it remains under ignored experiment data and is not packaged.

## Required final report

| # | Item | Result |
|---:|---|---|
| 1 | Real video source count | 10 |
| 2 | Human-grounded queries | 360 |
| 3 | Visual count | 120 |
| 4 | Speech count | 120 |
| 5 | Hybrid count | 120 |
| 6 | Train / validation / test | 180 / 120 / 60 |
| 7 | Frozen-test videos | 2 |
| 8 | Current accuracy | 48.33% |
| 9 | Current Visual recall | 100.00% |
| 10 | Current Speech recall | 0.00% |
| 11 | Current Hybrid recall | 45.00% |
| 12 | Character accuracy | 60.00% |
| 13 | Character Visual recall | 50.00% |
| 14 | Character Speech recall | 40.00% |
| 15 | Character Hybrid recall | 90.00% |
| 16 | Character precision/F1 | V 41.67/45.45%; S 66.67/50.00%; H 75.00/81.82% |
| 17 | Confusion matrix | V 10/4/6; S 12/8/0; H 2/0/18 |
| 18 | Median latency | 1.6260 ms |
| 19 | p95 latency | 1.9283 ms |
| 20 | Artifact size | 402,292 bytes |
| 21 | Ambiguous annotations | 20/360; 4/60 test |
| 22 | Source-disjoint verified | Yes |
| 23 | Acceptance leakage detected | No; protected evidence was not accessed |
| 24 | Second frozen run | No; gates failed |
| 25 | Deterministic | Training/checksum test passes; no formal second frozen metrics |
| 26 | Router promoted | No |
| 27 | Explicit modes preserved | Yes |
| 28 | Retrieval stack modified | No |
| 29 | Fallback available | Existing production router remains active; candidate fallback not needed |
| 30 | Decision | C |
| 31 | Exact reason | 60% accuracy, 50% Visual recall, and 40% Speech recall miss gates |
| 32 | Exact next milestone | Report this evidence; do not start another model experiment automatically |
| 33 | New Final Acceptance V2 video required | No, because no router was promoted |

The result rejects both production promotion and further tuning on these frozen sources. SceneMind should keep explicit Visual/Speech/Hybrid controls prominent and treat AUTO as unreliable for natural, indirect, or short English searches until a separately authorized milestone uses new training evidence and another untouched test.
