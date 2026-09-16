# Hybrid Fusion Holdout Validation V1

**HOLDOUT - DO NOT TUNE ON THIS DATA.**

Date: 2026-09-16

Decision: **REJECTED**
Production promotion supported: **No**

The frozen Cap 1.50x candidate improves Top-1 and MRR@5, but it reduces ALL-positive Top-5 from 21/28 to 20/28, reduces Speech Top-5 from 8/11 to 7/11, breaks one Jimmy Wales query, rescues none, and moves one unsupported query toward a more plausible but still wrong school scene. It fails the predeclared promotion rule. Production Hybrid remains baseline RRF60.

## Frozen evidence

The manifest is [`holdout/hybrid_fusion_holdout_v1.json`](holdout/hybrid_fusion_holdout_v1.json). Its separately recorded SHA-256 is:

`dcddffb0495cb7c397249ccd13d5818967127b6c62d01173944d0235030b64b5`

Freeze timestamp: `2026-09-16T13:25:47+03:00`. The checksum was recorded before the first retrieval call and still matches after the comparison. No query text, expected interval, category, positive/negative label, or source assignment changed after freeze. Final English Acceptance V2 evidence was not used.

| Source | Role | Bytes | Duration | SHA-256 | License |
| --- | --- | ---: | ---: | --- | --- |
| Jimmy Wales interview | Speech-heavy | 136,352,207 | 1,208.32 s | `dfba01c662cb7736642092e2311eaa46306a7023cd5af4dd231ab3918e7e568b` | CC BY-SA 4.0 |
| RUN | Visual/Multimodal | 356,013,405 | 1,515.36 s | `d35021584e3859bab68a465a8d341c2f813abe145f506e8edede5713022e61db` | CC BY-SA 4.0 |

Both are Wikimedia Commons sources and were previously verified source-disjoint from Final English Acceptance V2, the Hybrid Fusion development sources, and each other.

## Normal product ingestion

Both sources went through streamed HTTP upload, durable SQLite jobs, the normal five-second FFmpeg sampler, production Whisper-tiny, production CLIP indexing, and persistent product storage.

| Measurement | Jimmy Wales | RUN |
| --- | ---: | ---: |
| Upload | 0.465 s | 1.183 s |
| Ingest | 20.212 s | 20.812 s |
| Speech | 61.959 s | 94.405 s |
| Visual index | 27.348 s | 22.586 s |
| Total processing | 109.518 s | 137.803 s |
| Frames | 242 | 304 |
| Whisper segments | 233 | 498 |
| Production language | `en` | `hi` |
| Peak worker-tree RSS | 1,418,129,408 bytes | 1,340,321,792 bytes |
| Persistent run storage | 142,086,141 bytes | 364,555,363 bytes |
| Temporary bytes after completion | 0 | 0 |

Every durable ingest, Speech, and Visual job in the final measured runs reached `ready` on attempt 1, with no retry or error. RUN's `hi` metadata is recorded as produced. It agrees with the prior eligibility warning that one global tiny-model label is unreliable for this accented, mixed, music-containing documentary.

An initial Jimmy Speech run exposed a real boundary defect: the final Whisper segment ended at 1,210.29 seconds while OpenCV reported a playable video duration of 1,208.32 seconds. Strict validation failed twice after ingest had succeeded. The bounded fix validates starts, clips a segment end to the playable duration, and removes segments beginning after playback ends. The final stored segment is 1,206.69-1,208.32 seconds. A regression test covers clipping and removal while retaining failure for invalid negative timestamps. Models, sampling, retrieval, API shape, and fusion are unchanged.

## Query composition

The 34 queries were authored from the media, transcript, and direct frame review without retrieval output.

| Source | Speech | Visual | Multimodal | Negative | Positive | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jimmy Wales | 8 | 2 | 2 | 2 | 12 | 14 |
| RUN | 3 | 6 | 7 | 4 | 16 | 20 |
| **Total** | **11** | **8** | **9** | **6** | **28** | **34** |

Review found no duplicate or near-duplicate query, exact text overlap with the development or V2 manifests, filename leakage, invalid interval, or mislabeled category. The unusually broad documentary events were narrowed before freeze. No retrieval output was available during annotation or manifest review.

## Comparison method

The runner requested explicit Speech and Visual Top-50 candidate lists and production Hybrid Top-5. The evaluation tracer called `app.hybrid.fuse` and reproduced all 34 production results exactly. The offline baseline then reproduced the same IDs, timestamps, scores, ordering, and Top-5 membership before Cap was applied.

Baseline:

`c(r) = 1 / (60 + r)`

Frozen candidate:

`score = min(sum(retriever contributions), 1.50 * max(retriever contribution))`

Alpha remained exactly 1.50. No other alpha, normalization, percentile rule, quota, weight, threshold, or reranker was run.

## Aggregate results

| Variant | Top-1 | Top-3 | Top-5 | MRR@5 | Candidate recall | Strong retained | Shared Top-5 slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline RRF60 | 12/28 (42.9%) | 19/28 (67.9%) | 21/28 (75.0%) | 0.5560 | 28/28 | 18/32 (56.3%) | 127/140 (90.7%) |
| Cap 1.50x | 15/28 (53.6%) | 19/28 (67.9%) | 20/28 (71.4%) | 0.6083 | 28/28 | 18/32 (56.3%) | 126/140 (90.0%) |

Cap improves three Top-1 placements and MRR, but the benchmark's more important Top-5 protection fails. Every positive has the required modality evidence in its Top-50 input list; the remaining failures are ranking/grouping failures.

## Category results

| Category | Variant | Top-1 | Top-3 | Top-5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Speech | Baseline | 4/11 | 7/11 | 8/11 | 0.4879 |
| Speech | Cap 1.50x | 4/11 | 7/11 | 7/11 | 0.5000 |
| Visual | Baseline | 4/8 | 4/8 | 5/8 | 0.5250 |
| Visual | Cap 1.50x | 4/8 | 4/8 | 5/8 | 0.5250 |
| Multimodal | Baseline | 4/9 | 8/9 | 8/9 | 0.6667 |
| Multimodal | Cap 1.50x | 7/9 | 8/9 | 8/9 | 0.8148 |

The candidate is favorable to Multimodal rank ordering in RUN, but it does not preserve Speech Top-5 on the speech-heavy source.

## Source robustness

| Source | Variant | Top-1 | Top-3 | Top-5 | MRR@5 | Rescued | Broken |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jimmy Wales | Baseline | 2/12 | 5/12 | 6/12 | 0.2944 | - | - |
| Jimmy Wales | Cap 1.50x | 3/12 | 5/12 | 5/12 | 0.3333 | 0 | 1 |
| RUN | Baseline | 10/16 | 14/16 | 15/16 | 0.7521 | - | - |
| RUN | Cap 1.50x | 12/16 | 14/16 | 15/16 | 0.8146 | 0 | 0 |

The apparent overall gain is concentrated in RUN's within-Top-5 ordering. Jimmy loses one Top-5 success. This violates the source-robustness rule.

## Query outcomes

Counts over 28 positives:

- `RESCUED`: 0
- `BROKEN`: 1
- `UNCHANGED_SUCCESS`: 20
- `UNCHANGED_FAILURE`: 7

The broken query is `hfh-j-s08`: “Why does he think Wikipedia can outlast changing technology companies and remain useful for centuries?” Its expected interval is 1,120-1,208.32 seconds. Baseline ranks the matching shared bucket at 5; Cap ranks it at 8.

The matching bucket has Visual contribution `0.010870` and Speech contribution `0.013889`. Its score changes from `0.024758` to `0.020833`. Cap makes stronger single contributions in competing shared buckets more decisive: a nonmatching bucket around 646 seconds, for example, is capped at `0.022727` and moves above it. Candidate generation and evidence remain present, so this is a **ranking failure** caused by the cap's mechanical preference among differently balanced shared buckets.

Unchanged failures are:

- `hfh-j-s03`, rank 16
- `hfh-j-s04`, rank 6 after Cap
- `hfh-j-s05`, rank 9 after Cap
- `hfh-j-v01`, rank 10 after Cap
- `hfh-j-v02`, rank 12
- `hfh-j-m01`, rank 22 after Cap
- `hfh-r-v02`, rank 8

The 20 unchanged successes are `hfh-j-s01`, `hfh-j-s02`, `hfh-j-s06`, `hfh-j-s07`, `hfh-j-m02`, `hfh-r-s01`, `hfh-r-s02`, `hfh-r-s03`, `hfh-r-v01`, `hfh-r-v03`, `hfh-r-v04`, `hfh-r-v05`, `hfh-r-v06`, and `hfh-r-m01` through `hfh-r-m07`.

There are zero candidate-recall failures and eight Cap Top-5 ranking failures. Relevant Speech or Visual evidence was available in the required Top-50 inputs for every positive.

## Strong candidates and overlap

Strong required-modality Top-5 candidate retention remains exactly 18/32 (56.3%). Cap therefore does not improve the predeclared retention measure on new evidence. Positive shared-thumbnail occupancy changes only from 127/140 (90.7%) to 126/140 (90.0%). The cap changes relative scores more than it changes the shared composition of the returned list.

## Negative queries

No no-match threshold or absence claim was introduced. Of six unsupported queries, one keeps identical order and two keep the same candidate set. Four change set membership.

Most changes are different wrong moments rather than a systematic improvement. One is clearly less safe: for the unsupported request for a teacher writing the legal marriage age on a classroom board, Cap moves an outdoor student/advocacy moment around 1,343 seconds to rank 1. It is visually and semantically plausible enough to look convincing while still lacking the requested teacher, board, law, and explanation. Negative behavior therefore does not pass the conservative predeclared criterion.

## Search latency

The comparison changes only offline ranking and adds no product latency. Observed production request latency was:

| Path | All median | All p95 | Warm median | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| Speech | 16.45 ms | 23.60 ms | 15.64 ms | 23.60 ms |
| Visual | 28.89 ms | 6,954.90 ms | 28.51 ms | 39.92 ms |
| Hybrid | 42.68 ms | 96.83 ms | 42.68 ms | 96.83 ms |

Visual all-query p95 includes one lazy model/index initialization for each isolated API process.

## Predeclared decision rule

| Criterion | Pass | Evidence |
| --- | :---: | --- |
| ALL Top-5 does not materially regress | No | 21/28 to 20/28; one net loss and no rescue |
| Overall MRR@5 does not materially regress | Yes | 0.5560 to 0.6083 |
| Speech improves or has no meaningful regression | No | Top-5 8/11 to 7/11 |
| Visual Top-5 does not materially regress | Yes | 5/8 to 5/8 |
| Multimodal Top-5 does not materially regress | Yes | 8/9 to 8/9 |
| Newly broken does not outweigh rescued | No | 1 broken, 0 rescued |
| No meaningful cross-source damage | No | Jimmy Top-5 6/12 to 5/12 |
| Strong-candidate retention stable or improved | Yes | 18/32 to 18/32 |
| No severe ranking pathology | Yes | Deterministic valid rankings; no crash or broad candidate loss |
| Negatives do not become clearly more misleading | No | Unsupported classroom-law request gets a more plausible wrong Top-1 |

All criteria did not pass. The holdout decision is **REJECTED**. Cap 1.50x must not be promoted, and the failed holdout must not be tuned. Because the candidate failed its quality gates, no second frozen run was performed.

## Product decision

Production remains the existing baseline in `app.hybrid.fuse`: unweighted RRF with constant 60 and no overlap cap. Smart Search, explicit routes, frontend, models, sampling, candidate generation, grouping, and API response shape remain unchanged. The only supporting production change is safe transcript-tail normalization required for the approved Jimmy source to complete the normal ingestion path.

The machine-readable evidence is [`reports/hybrid-fusion-holdout-v1.json`](reports/hybrid-fusion-holdout-v1.json). It contains both input candidate lists, exact production traces, Baseline/Cap scores and ranks, every positive outcome, negative ordering, ingestion measurements, and the decision-rule audit.

No production promotion, Final Acceptance V3, parameter search, or second holdout run occurred.
