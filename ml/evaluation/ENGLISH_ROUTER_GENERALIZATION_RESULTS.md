# English AUTO router generalization results

Decision: **E — DATASET QUALITY/QUANTITY INSUFFICIENT**. The classical character n-gram router passes every internal numeric gate, but the new queries are grounded in independently authored source scenarios rather than human-reviewed real videos. That evidence is useful for diagnosis and implementation testing but is insufficient to replace production routing.

## Data

| Measure | Result |
|---|---:|
| Independent source scenarios | 15 |
| Total queries | 450 |
| Train | 270 queries / 9 sources |
| Validation | 90 queries / 3 sources |
| Frozen test | 90 queries / 3 sources |
| Visual / Speech / Hybrid | 150 / 150 / 150 |
| Frozen manifest SHA-256 | `ce1f2b0e0665c037fb4f914878e5209052160687775cfdfdafa5d406810380da` |
| Real-video grounded | No |

Splits are source-disjoint and the test was frozen before comparison. The final acceptance video, manifest, transcript, frames, timestamps, result lists, and failure examples were excluded. An automated normalized exact-overlap guard found no acceptance-query leakage. The source-card limitation is why the experiment cannot support production promotion.

## Current production router

| Metric | Frozen test |
|---|---:|
| Accuracy | 50.00% (45/90) |
| Visual precision / recall / F1 | 42.25% / 100.00% / 59.41% |
| Speech precision / recall / F1 | 50.00% / 13.33% / 21.05% |
| Hybrid precision / recall / F1 | 100.00% / 36.67% / 53.66% |
| Median / p95 latency | 0.0272 / 0.0414 ms |
| Confidence min / median / p95 / max | 0.4841 / 0.8439 / 0.9940 / 0.9988 |

Confusion matrix, rows actual and columns predicted:

| Actual \ Predicted | Visual | Speech | Hybrid |
|---|---:|---:|---:|
| Visual | 30 | 0 | 0 |
| Speech | 26 | 4 | 0 |
| Hybrid | 15 | 4 | 11 |

The baseline is confidently wrong: median confidence is 0.8439 while half the routes fail. Its weakest slices are indirect Speech intent at 3.85% accuracy, mixed modality at 36.67%, ambiguous wording at 41.67%, natural questions and long queries at 50.00%, and technical terms at 56.67%. Indirect Visual intent is 100%. On this controlled set, the 54-parameter router underfits phrasing beyond its small original vocabulary.

## Best classical candidate

Validation selected character 3–5 gram TF-IDF plus a three-class linear softmax classifier. Word and combined candidates were also evaluated; no sentence embedding model was needed.

| Metric | Frozen test |
|---|---:|
| Accuracy with validation-selected fallback | 95.56% (86/90) |
| Visual precision / recall / F1 | 96.77% / 100.00% / 98.36% |
| Speech precision / recall / F1 | 100.00% / 90.00% / 94.74% |
| Hybrid precision / recall / F1 | 90.63% / 96.67% / 93.55% |
| Classifier parameters | 23,919 |
| Serialized artifact | 731,122 bytes |
| Estimated model/vocabulary memory | 288,922 bytes |
| Median / p95 latency | 2.7706 / 3.3677 ms |

Confusion matrix:

| Actual \ Predicted | Visual | Speech | Hybrid |
|---|---:|---:|---:|
| Visual | 30 | 0 | 0 |
| Speech | 0 | 27 | 3 |
| Hybrid | 1 | 0 | 29 |

The selected validation threshold is `0.348785`. It sends 3.33% of frozen-test queries to Hybrid. Raw frozen-test accuracy without fallback is 97.78%; fallback lowers it to 95.56%, so confidence fallback does not generalize beneficially to this test and should not be promoted.

The candidate passes the internal ≥90% accuracy, ≥85% per-route recall, <5 ms median, and <10 ms p95 gates. A second evaluation of the same frozen test is exactly deterministic. Passing these gates does not override the missing real-video grounding.

## Integrity and production

| Check | Result |
|---|---|
| Acceptance-video leakage detected | No |
| Source-disjoint split verified | Yes |
| Test used for model/fallback selection | No |
| Second frozen run completed | Yes |
| Second-run metrics deterministic | Yes |
| Router promoted | No |
| Retrieval stack modified | No |
| Explicit modes preserved | Yes |

The exact next milestone is to collect independently reviewed route annotations from new real English videos across several domains, freeze source-disjoint train/validation/test splits, and rerun this same classical comparison. If real-video evidence passes, persist and promote a reproducible local artifact with the existing safe fallback. Final English Acceptance V2 will still require a completely new, previously unseen 30–60 minute video and a newly frozen manifest.
