# Final English Acceptance V2 results

Status: **C — FINAL ACCEPTANCE V2 FAILED** (2026-09-16).

SceneMind processed a new, source-disjoint 30:29 English presentation through the unchanged normal durable pipeline. The full transcript and 366 five-second review frames were inspected before the 30-query manifest was frozen. Every primary query used **Smart Search → Hybrid**; AUTO was never used. No model, sampler, index, threshold, query, interval, label, or production retrieval behavior changed after freeze.

## Media and integrity

| Measurement | Result |
| --- | --- |
| File | `emf2026-9801-eng-So_-_how_does_the_Internet_really_work_hd.mp4` |
| Size | 167,172,049 bytes (159.43 MiB) |
| SHA-256 | `addb005d02cde7bc6b277e938bc10190709e278088caa23354e1323477f0000c` |
| Duration | 1,829.08 s (30:29.08) |
| Streams | H.264 High 1920×1080 at 25 fps; AAC-LC 48 kHz stereo English audio |
| Source / rights | media.ccc.de; CC BY-SA 4.0 |
| Independent review | Complete transcript/audio review plus all 366 five-second JPEGs before search |
| Manifest | 30 queries; SHA-256 `ce633b525af0cc3a347c51017c4019eb173e376246783b8153bd6dc8b4c60b7c` |
| Frozen mix | 10 Speech, 8 Visual, 8 Multimodal, 4 verified negatives |

The manifest is checked by a canonical checksum that excludes only its checksum field. The tracked regression rejects any post-freeze query, interval, category, or label change. The video was not used for training, tuning, calibration, threshold selection, vocabulary work, or model selection.

## Normal product ingestion

| Measurement | Result |
| --- | ---: |
| Upload | 0.994 s; HTTP 202 |
| End-to-end processing | 244.393 s (4:04.39) |
| Ingest / Speech / Visual stage spans | 57.815 / 132.821 / 53.757 s |
| Production frames | 366 at the unchanged five-second interval |
| Whisper segments | 576; language `en`; ordered timestamps valid |
| Upload API peak RSS | 112,427,008 bytes (107.22 MiB) |
| Search API peak process-tree RSS | 590,110,720 bytes (562.77 MiB) |
| Worker peak / steady process-tree RSS | 2,086,465,536 / 1,042,067,456 bytes (1.94 GiB / 993.79 MiB) |
| Persistent application data | 174,692,504 bytes (166.60 MiB) |
| Durable attempts | Ingest 1; Speech 1; Visual 1 |
| Retries / failures / temporary residue | 0 / 0 / 0 bytes |

The video reached Ready through HTTP upload, durable jobs, FFmpeg extraction, production Whisper, production CLIP, and persisted indexes. No special benchmark ingestion path was used.

## Frozen acceptance metrics

Only PASS-level results count as useful. The one PARTIAL result does not raise Top-k or MRR.

| Metric | Result | Frozen gate | Pass? |
| --- | ---: | ---: | :---: |
| Useful Top-1 | 76.92% (20/26) | preferred ≥70% | Yes |
| Useful Top-3 | 76.92% (20/26) | ≥85% | **No** |
| Useful Top-5 | 84.62% (22/26) | ≥90% | **No** |
| MRR@5 | 0.7865 | descriptive | — |
| Positive PASS / PARTIAL / FAIL | 22 / 1 / 3 | descriptive | — |
| Negative acceptable / misleading | 4 / 0 | no false certainty | Yes |
| Search latency median / nearest-rank p95 | 64.58 / 251.63 ms | interactive | Yes after load |
| First cold search / maximum | 10,416.62 ms | disclosed | Product friction |
| Catastrophic pipeline failure | None | none | Yes |

The full 30-request distribution is used for latency. The single 10.42-second maximum is the first in-process CLIP load; it is reported separately and remains a real cold-start limitation. The other 29 requests were 46–252 ms. Initial processing time is separate from interactive search latency.

All 22 PASS-level timestamps fall inside their frozen evidence interval, so interval-boundary error is 0 seconds. This does not mean an exact event-center error was measured: the frozen manifest defines valid intervals, not single ideal timestamps. S03 also overlaps its interval but remains PARTIAL because it opens after the causal explanation; overlap alone was not credited.

## Category results

| Evidence type | Queries | Top-1 | Top-3 | Top-5 | PASS / PARTIAL / FAIL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speech | 10 | 60.00% | 60.00% | 70.00% | 7 / 1 / 2 |
| Visual | 8 | 75.00% | 75.00% | 87.50% | 7 / 0 / 1 |
| Multimodal | 8 | 100.00% | 100.00% | 100.00% | 8 / 0 / 0 |

The presentation is favorable to combined evidence: its stable, large slides and synchronized narration let all eight multimodal requests succeed at rank 1. Clear visual states also work well. Speech-evidence queries are the weakest group because relevant BM25 moments can lose too much rank when fused with visually generic slides.

## Query-level judgment

| ID | Evidence | PASS rank | T1 | T3 | T5 | Outcome | Human judgment |
| --- | --- | ---: | :---: | :---: | :---: | --- | --- |
| S01 | Speech | 1 | Yes | Yes | Yes | PASS | Goals passage and beginner takeaway are direct. |
| S02 | Speech | 1 | Yes | Yes | Yes | PASS | Browser-received content is explicit. |
| S03 | Speech | — | No | No | No | PARTIAL | Rank 5 gives the conclusion after the causal explanation; user must search backward. |
| S04 | Speech | 1 | Yes | Yes | Yes | PASS | Switch learning is stated directly. |
| S05 | Speech | 1 | Yes | Yes | Yes | PASS | Wired/Wi-Fi equivalence is explicit. |
| S06 | Speech | 1 | Yes | Yes | Yes | PASS | Opens the continuous MAC-address passage leading to the uniqueness reason. |
| S07 | Speech | 5 | No | No | Yes | PASS | Rank 5 opens at the activity graph immediately before the selection advice. |
| S08 | Speech | — | No | No | No | FAIL | Lower-level-address explanation is absent from Hybrid Top-5. |
| S09 | Speech | — | No | No | No | FAIL | Earlier DNS material displaces the 9.9.9.9 diagnostic passage. |
| S10 | Speech | 1 | Yes | Yes | Yes | PASS | Closing experimentation advice opens directly. |
| V01 | Visual | 1 | Yes | Yes | Yes | PASS | Three orange goal bars are visible. |
| V02 | Visual | 1 | Yes | Yes | Yes | PASS | HTML source view is direct. |
| V03 | Visual | 1 | Yes | Yes | Yes | PASS | Hand-drawn postcards are visible. |
| V04 | Visual | — | No | No | No | FAIL | No returned frame shows the three home-network boxes. |
| V05 | Visual | 1 | Yes | Yes | Yes | PASS | Pneumatic-tube photograph is direct. |
| V06 | Visual | 4 | No | No | Yes | PASS | Rank 4 clearly shows Alice, Blair, arrows, and teacher. |
| V07 | Visual | 1 | Yes | Yes | Yes | PASS | IEEE 802 device comparison is direct. |
| V08 | Visual | 1 | Yes | Yes | Yes | PASS | Wireshark and yellow address annotation are direct. |
| M01–M08 | Multimodal | 1 each | Yes | Yes | Yes | PASS | Each rank-1 click contains the required visual state and matching speech context. |

All four negative requests were independently confirmed absent. The returned neighbors are visibly or textually distinguishable from the request, and the current “Most relevant moments” / “Possible matches” wording does not claim that unsupported content exists. This is acceptable conservative ranking behavior, not reliable no-match detection.

## Diagnostic evidence and decision

Explicit modes were run only after primary judgment and never substituted into metrics. Speech mode moved S03, S08, and S09 to rank 1 at 243.78, 1210.78, and 1489.16 seconds. S07 remained low in Speech (rank 2 there versus rank 5 in Hybrid). Visual mode still missed V04, returning 425–1100 second switch/router frames; it also placed V06's closest relevant frame at rank 2. S06 was checked while its long-passage judgment was unresolved and remained a primary PASS.

Choose **C — FINAL ACCEPTANCE V2 FAILED**. Top-3 misses its hard gate by 8.08 percentage points and Top-5 misses by 5.38 points. The dominant subsystem is **Hybrid fusion/ranking of strong speech evidence**: two primary FAILs and the sole PARTIAL have much stronger explicit Speech results. The secondary failure is one CLIP visual-retrieval miss on a distinctive three-device slide.

The English-first v1.0 search core is **not frozen**. Do not tune or rerun on this video. The exact next milestone is a bounded product-level investigation of Hybrid fusion using new source-disjoint development evidence, starting with why high-quality speech hits are displaced by visually generic candidates. No new model milestone is authorized automatically.

Machine-readable primary responses, human observations, resource measurements, and post-judgment diagnostics are preserved in [reports/final-english-acceptance-v2.json](reports/final-english-acceptance-v2.json). Final Acceptance V1 remains unchanged historical evidence.
