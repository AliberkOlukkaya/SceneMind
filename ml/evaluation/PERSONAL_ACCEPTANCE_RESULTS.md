# Personal acceptance results

Status: **C — NOT YET ACCEPTED** (2026-09-15).

SceneMind was exercised through the production HTTP pipeline on the three files supplied by Aliberk. The 54-query manifest was frozen before retrieval, contains 27 English and 27 natural Turkish queries, and was not used for tuning. Every query ran in AUTO, Visual, Speech, and Hybrid. Human judgment required a result to be useful after a normal click; timestamp overlap alone did not count.

## Acceptance result

All five numerical quality gates failed. Search itself stayed interactive, but Turkish behavior, software-demo retrieval, false confidence on unsupported queries, and the lack of a valid long-form acceptance sample prevent a v1.0 acceptance claim.

| Metric | Result | Gate | Pass? |
| --- | ---: | ---: | :---: |
| AUTO routing accuracy | 77.78% | >=90% | No |
| Useful Top-1, positive queries | 52.78% | — | — |
| Useful Top-3, positive queries | 66.67% | >=85% | No |
| Useful Top-5, positive queries | 83.33% | >=90% | No |
| English useful Top-5 | 88.89% | >=90% | No |
| Turkish useful Top-5 | 77.78% | preferred >=80% | No |
| Negative queries with a non-misleading response | 22.22% | — | — |
| AUTO search latency, median / p95 | 23.25 / 32.02 ms | interactive | Yes |

Overall human labels were **21 PASS, 11 PARTIAL, and 22 FAIL**. Top-k rates use the 36 positive queries; negative behavior is reported separately across 18 queries.

## Language and scenario slices

| Slice | AUTO route | Top-1 | Top-3 | Top-5 | PASS / PARTIAL / FAIL |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 100.00% | 66.67% | 83.33% | 88.89% | 14 / 4 / 9 |
| Turkish | 55.56% | 38.89% | 50.00% | 77.78% | 7 / 7 / 13 |
| Lecture/tutorial | 85.00% | 41.67% | 58.33% | 91.67% | 5 / 6 / 9 |
| Software demo | 65.00% | 55.56% | 61.11% | 72.22% | 10 / 3 / 7 |
| Street visual smoke | 85.71% | 66.67% | 100.00% | 100.00% | 6 / 2 / 6 |

The street result covers only six positive visual queries in a 35-second clip. It is not evidence for long real-world videos. The apparent lecture Top-5 strength is visual evidence from a silent 22:49 tutorial; it does not validate lecture speech search.

AUTO chose the expected route for every English query and only 55.56% of Turkish queries. All 12 route mismatches were Turkish. Explicitly forcing the expected route raised Turkish Top-3 from 50.00% to 61.11%, but Top-5 remained 77.78%, so routing is only part of the multilingual failure. English terminology also matters: the status-bar query failed because the transcript used “display.”

## Category and mode diagnosis

| Category | Positive queries | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Visual object | 10 | 70.00% | 80.00% | 90.00% |
| Visual scene | 8 | 62.50% | 62.50% | 100.00% |
| Compositional visual | 8 | 37.50% | 87.50% | 87.50% |
| Spoken topic | 4 | 25.00% | 25.00% | 50.00% |
| Quoted/mentioned phrase | 2 | 50.00% | 50.00% | 100.00% |
| Mixed visual + spoken | 4 | 50.00% | 50.00% | 50.00% |

Across only the positive queries assigned to each route, explicit Visual reached 53.85%/76.92%/92.31% at Top-1/3/5, Speech reached 66.67% at all three cutoffs, and Hybrid reached 50.00% at all three. The samples are small, especially Speech (6) and Hybrid (4), but they show that ordinary visual object/scene search is the most dependable use today. Mixed queries and software UI distinctions remain unreliable.

## Media and processing

| Video | Verified source | Production outcome | Frames / segments | Pipeline time |
| --- | --- | --- | ---: | ---: |
| Cup-of-tea tutorial | 22:49.63, 1920x1080, 60 fps, VP8, no audio, 439,295,727 bytes | Original upload returned 413 at the 250 MiB limit. A meaning-preserving H.264 CRF 27 derivative was required. | 274 / 0 | 106.30 s after transcode |
| Blender interface demo | 11:12.56, 1920x1080, 24 fps, VP9 + Opus | Ingested unchanged | 135 / 103 | 55.48 s |
| Street traffic | 35.00 s, 1920x1080, 30 fps, VP8 + Vorbis | Ingested unchanged | 7 / 0 | 3.70 s |

The lecture derivative took about 309 seconds to create outside SceneMind. Its product pipeline then took 67.09 seconds for ingestion, 36.93 seconds for CLIP indexing, and 2.28 seconds for the failed audio-less ASR attempt. The demo took 10.90 seconds for ingestion, 7.09 seconds for indexing, and 37.49 seconds for ASR. These in-limit runs are tolerable relative to video length, but the original lecture rejection and manual transcode are serious product friction.

Neither the 22:49 lecture nor the 11:12 demo meets the requested 30–60 and 15–60 minute scenario ranges. Consequently this run cannot establish smooth 30–60 minute use or 60-minute support. Production still defaults to a 1,800-second duration cap.

## Transcript and timestamp observations

The cup-of-tea source has no audio, so it cannot support spoken-topic evaluation. Whisper-tiny produced 103 English segments for the demo. A chapter-derived technical-term sample retained 19 of 20 terms; observed errors included “prime menu” for “pie menu,” “double-cooking” for “double-clicking,” “properties added” for “Properties editor,” and “Blunder” for “Blender.” Timestamps followed the spoken sections closely. The street clip produced no speech segments and did not hallucinate narration.

Of 30 rated useful timestamps, 19 were within one second of a frozen interval, 2 were over 1 and at most 5 seconds away, 1 was over 5 and at most 10 seconds away, and 8 were over 30 seconds away. The stored median is 0 seconds and p95 is 310.73 seconds. Some long errors come from frozen intervals that were too narrow for repeated static UI or scene states, so the p95 is retained for audit but is not a clean seek-precision estimate. Human labels already account for whether the click remained useful.

## Product questions

1. **Would Aliberk benefit today?** Yes for bounded English visual lookup and some exact speech terms, but not reliably enough to call the whole product v1.0-ready.
2. **What is trustworthy?** Clear English visual objects/scenes and explicit Visual search; visual-scene Top-5 was 100% and visual-object Top-5 was 90% on this small suite.
3. **What is unreliable?** Turkish speech/mixed queries, fine software-interface distinctions, paraphrased lexical speech, unsupported negatives, and temporal/action requests.
4. **How well did Turkish work?** Below target: 55.56% route accuracy and 38.89%/50.00%/77.78% useful Top-1/3/5.
5. **How well did English work?** Substantially better: 100% route accuracy and 66.67%/83.33%/88.89% Top-1/3/5, still just below the frozen Top-5 gate.
6. **Is AUTO reliable?** No as a universal default for this bilingual use. Its English routing was reliable; Turkish lexical cues were not.
7. **Was speech useful?** Useful for exact English demo terms, but vulnerable to terminology and impossible on the silent tutorial.
8. **Was visual useful?** Yes for clear objects and broad scenes; less so for Turkish phrasing and fine UI/compositional distinctions.
9. **Was Hybrid useful?** Inconsistent. On its four intended positive queries it reached only 50% at Top-5, with two diagnosed fusion failures overall.
10. **Was ASR a blocker?** The demo ASR was usable. The silent lecture made speech unavailable and exposed an unhelpful retry message, but this is partly a source-selection failure.
11. **Was no-match a blocker?** Yes. Fourteen of 18 negative queries returned plausibly misleading moments; only 22.22% were non-misleading.
12. **Was timestamp precision useful?** Usually when retrieval found the right moment. The frozen-interval design prevents a strong aggregate precision claim.
13. **Was processing acceptable?** Search was fast and in-limit processing was reasonable. The 250 MiB rejection and five-minute manual transcode were not acceptable as a normal workflow.
14. **Is another ML experiment needed?** Yes: one bounded Turkish retrieval/routing compatibility milestone using the current models should precede release.
15. **Top three blockers?** Turkish routing/retrieval, false confidence on unsupported queries, and unvalidated plus size-constrained long-form ingestion.
16. **Should model experimentation stop?** Broad model exploration should stop; only the measured multilingual blocker justifies focused work.
17. **What remains for v1.0?** Fix and freeze Turkish routing/retrieval behavior, design an honest unsupported-query response, remove or clearly handle the long-video ingestion constraint, and repeat acceptance on valid-duration audio material.
18. **Safe CV/GitHub claims?** Local-first CLIP/Whisper/BM25/RRF video search; production AUTO and explicit modes; this frozen suite’s exact measured metrics; interactive local search latency; tested English visual usefulness.
19. **Claims to avoid?** v1.0 accepted, reliable Turkish search, reliable no-match, smooth 30–60/60-minute support, broad real-world accuracy, or robust temporal understanding.
20. **Next single milestone?** A source-disjoint Turkish query compatibility milestone for AUTO plus current CLIP/BM25/RRF, followed by a frozen repeat before promotion.

The complete auditable measurements are in [reports/personal-acceptance-v1.json](reports/personal-acceptance-v1.json). No production search behavior or model was changed.
