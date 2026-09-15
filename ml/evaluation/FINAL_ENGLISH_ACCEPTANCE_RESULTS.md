# Final English long-video acceptance results

Status: **C — FINAL ACCEPTANCE FAILED** (2026-09-15).

SceneMind processed one new, real, continuous 54-minute English technical presentation through the normal HTTP upload, durable job, FFmpeg, Whisper, CLIP, persistence, and AUTO search path. The full audio/transcript and 217 timeline frames were reviewed independently before any SceneMind search. Thirty natural queries were then frozen in an ignored private manifest with SHA-256 `075d792d05f08e443b3f638e69f93a3f1c24d54dbd472052c1075bb478da11f8`. No model, threshold, router, sampler, query label, or ranking behavior changed after results were observed.

## Media and ingest

| Measurement | Result |
| --- | ---: |
| Filename | `gpn24-673-eng-Evaluating_and_developing_machine_learning_models_aeN_introduction_av1-hd.webm` |
| Media SHA-256 | `9b99a4eb2729772e71baa153b7f38f050a0291993c9eed513a72b6bb4e0b11de` |
| Size | 285,562,260 bytes |
| Duration | 3,251.28 s (54:11.28) |
| Streams | 2 × 1920×1080 AV1 video; English stereo Opus audio |
| Rights | CC BY 4.0, recorded in file metadata and on the media.ccc.de source page |
| Upload | Success in 0.995 s |
| End-to-end product processing | Success in 579.258 s (9:39.26) |
| Production frames | 651 at the unchanged 5-second sampling interval |
| Whisper segments | 684; detected language `en` |
| Peak API process-tree RSS | 582,352,896 bytes (555.4 MiB) |
| Peak worker process-tree RSS | 3,005,247,488 bytes (2.80 GiB) |
| Final worker process-tree RSS | 1,008,967,680 bytes (962.2 MiB) |
| Persistent application data | 299,480,644 bytes (285.6 MiB) |
| Temporary residue | 0 bytes |
| Retries / failures / final state | 0 / 0 / Ready |

The three durable jobs each completed on their first attempt. The observed job spans were 326.423 seconds for ingest, 164.656 seconds for Whisper, and 57.060 seconds for CLIP indexing. End-to-end time includes queue polling and the normal API requests that start the later stages.

## Frozen gates

Primary Top-k metrics use only the 26 positive queries. The four confirmed negatives evaluate whether the conservative UI remains honest.

| Metric | Result | Frozen gate | Pass? |
| --- | ---: | ---: | :---: |
| AUTO routing accuracy | 56.67% (17/30) | ≥90% | No |
| Useful Top-1 | 50.00% (13/26) | preferred ≥70% | No |
| Useful Top-3 | 65.38% (17/26) | ≥85% | No |
| Useful Top-5 | 65.38% (17/26) | ≥90% | No |
| MRR@5 | 0.5705 | descriptive | — |
| Positive PASS / PARTIAL / FAIL | 12 / 5 / 9 | descriptive | — |
| Negative conservative behavior | 100% (4/4) | no false certainty | Yes |
| Negative misleading behavior | 0% (0/4) | descriptive | — |
| Search latency median / p95 | 27.76 / 62.96 ms | interactive | Yes |
| First cold search / maximum | 7,366.61 ms | descriptive | — |
| Catastrophic ingest/process failure | None | none | Yes |

The first query paid the in-process CLIP cold-load cost. The remaining search requests were interactive; nearest-rank p95 excludes that single maximum in this 30-query run. The cold-start delay is retained as product evidence rather than hidden by a warm-up.

The interactive browser connector exposed no Chrome or in-app browser surface during this run, so the acceptance media could not receive an additional live-window click-through. Actual AUTO responses, thumbnails, nearby audio, and transcript evidence were still reviewed result by result. The current frontend contract was reverified separately by six passing desktop/mobile Playwright flows covering AUTO default, conservative wording, transcript context, and click-to-seek; two opt-in real-model browser cases were skipped. This is a UI-observation limitation, not a substitution for the measured production API results.

## Category results

| Category | Queries | Top-1 | Top-3 | Top-5 | PASS / PARTIAL / FAIL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speech | 10 | 20.00% | 30.00% | 30.00% | 1 / 2 / 7 |
| Visual | 6 | 83.33% | 83.33% | 83.33% | 5 / 0 / 1 |
| Hybrid | 6 | 50.00% | 83.33% | 83.33% | 3 / 2 / 1 |
| Compositional | 4 | 75.00% | 100.00% | 100.00% | 3 / 1 / 0 |

The presentation is visually favorable: large, stable slides occupy most of the frame. Clear slide-state search is useful, including code, repository, colored-grid, and graph queries. The one Visual failure confused a three-figure FLOP-counter diagram with the visually similar state-tracking grid.

Speech performance is the decisive weakness in AUTO. Nine of ten Speech queries were routed; only one reached Speech. Explicit diagnostic Speech mode found the target at rank 1 or 4 for S01, S04, S05, S07, S08, and S09. Two remaining questions, about experiment metadata and private-compute options, also failed explicit BM25 Top-5 despite usable transcript text. Hybrid generally recovered mixed evidence, although fusion placed three useful speech-supported moments at rank 2 or 3.

## Per-query judgment

`Route` is whether AUTO matched the route frozen before retrieval. Top-k is human usefulness after inspecting the clicked frame, nearby speech, and transcript context; interval overlap alone did not qualify.

| ID | Category | Expected | AUTO | Route | T1 | T3 | T5 | Outcome |
| --- | --- | --- | --- | :---: | :---: | :---: | :---: | --- |
| S01 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S02 | Speech | SPEECH | SPEECH | Yes | Yes | Yes | Yes | PASS |
| S03 | Speech | SPEECH | VISUAL | No | Yes | Yes | Yes | PARTIAL |
| S04 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S05 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S06 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S07 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S08 | Speech | SPEECH | VISUAL | No | No | Yes | Yes | PARTIAL |
| S09 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| S10 | Speech | SPEECH | VISUAL | No | No | No | No | FAIL |
| V01 | Visual | VISUAL | HYBRID | No | Yes | Yes | Yes | PASS |
| V02 | Visual | VISUAL | VISUAL | Yes | Yes | Yes | Yes | PASS |
| V03 | Visual | VISUAL | VISUAL | Yes | Yes | Yes | Yes | PASS |
| V04 | Visual | VISUAL | VISUAL | Yes | No | No | No | FAIL |
| V05 | Visual | VISUAL | VISUAL | Yes | Yes | Yes | Yes | PASS |
| V06 | Visual | VISUAL | VISUAL | Yes | Yes | Yes | Yes | PASS |
| H01 | Hybrid | HYBRID | HYBRID | Yes | Yes | Yes | Yes | PASS |
| H02 | Hybrid | HYBRID | HYBRID | Yes | Yes | Yes | Yes | PASS |
| H03 | Hybrid | HYBRID | HYBRID | Yes | No | Yes | Yes | PARTIAL |
| H04 | Hybrid | HYBRID | HYBRID | Yes | Yes | Yes | Yes | PASS |
| H05 | Hybrid | HYBRID | VISUAL | No | No | No | No | FAIL |
| H06 | Hybrid | HYBRID | HYBRID | Yes | No | Yes | Yes | PARTIAL |
| C01 | Compositional | HYBRID | HYBRID | Yes | Yes | Yes | Yes | PASS |
| C02 | Compositional | HYBRID | SPEECH | No | Yes | Yes | Yes | PASS |
| C03 | Compositional | HYBRID | HYBRID | Yes | Yes | Yes | Yes | PASS |
| C04 | Compositional | HYBRID | HYBRID | Yes | No | Yes | Yes | PARTIAL |
| N01 | Negative | HYBRID | VISUAL | No | — | — | — | ACCEPTABLE |
| N02 | Negative | VISUAL | VISUAL | Yes | — | — | — | ACCEPTABLE |
| N03 | Negative | HYBRID | HYBRID | Yes | — | — | — | ACCEPTABLE |
| N04 | Negative | VISUAL | VISUAL | Yes | — | — | — | ACCEPTABLE |

Negative candidates were visibly or textually unrelated, and the current UI says “Most relevant moments” with a single relevance caveat. It never claims the confusion matrix, Docker/Kubernetes diagram, Jupyter demonstration, or multi-head-attention diagram exists. This is acceptable ranked-candidate behavior; it is not no-match detection.

## Decision

Choose **C — FINAL ACCEPTANCE FAILED**. The single dominant blocker is **AUTO routing generalization on natural English interrogative technical queries**. Routing reached 56.67%, 33.33 percentage points below its frozen gate, and six otherwise weak positive searches were rescued by their frozen expected explicit mode. Useful Top-3 and Top-5 remain far below their gates, so this is not a narrow conditional acceptance.

SceneMind may retain its existing component and infrastructure claims, including local 54-minute processing, CLIP visual retrieval, Whisper indexing, BM25/RRF paths, and manual search modes. It may not claim that the v1.0 search core is accepted or that AUTO is safe as the default for this use. Do not tune on this video or rerun it as fresh acceptance after a change.

ML experimentation should not broaden. The exact next milestone is one bounded, source-disjoint English AUTO-router generalization milestone. If a candidate passes its own held-out gate, final acceptance must use a new independently reviewed 30–60 minute English video and a newly frozen manifest.

The machine-readable aggregate and per-query judgments are in [reports/final-english-acceptance-v1.json](reports/final-english-acceptance-v1.json). Private media, the frozen manifest, raw responses, observations, frames, transcripts, indexes, and databases remain ignored.
