# Final English long-video acceptance failures

Status: **measured; decision C — FINAL ACCEPTANCE FAILED** (2026-09-15).

Counts are multi-label. Planned negative queries are listed separately because their `UNSUPPORTED_QUERY` label is expected and did not make the conservative UI misleading.

## Positive-query impact

| Rank | Failure category | Affected positives | Evidence |
| ---: | --- | ---: | --- |
| 1 | ROUTING | 12 | AUTO disagreed with the frozen expected route. Six weak/failed searches were recovered by explicit Speech or Hybrid diagnostics. |
| 2 | HYBRID_FUSION | 3 | H03, H06, and C04 had exact Speech evidence, but the combined list placed the useful result at rank 2 or 3. |
| 2 | TIMESTAMP_PRECISION | 3 | S03, S08, and C02 opened a broad section, a later repeat, or a point before the exact statement. The clicks remained useful for S03/C02 and useful by rank 2 for S08. |
| 4 | BM25/SPEECH_RETRIEVAL | 2 | S06 and S10 were absent from explicit Speech Top-5 despite usable Whisper text. |
| 4 | CLIP_VISUAL_RETRIEVAL | 2 | V04 confused similar diagrams; C04's correct colored-grid frame did not survive visual ranking, leaving Speech to recover rank 3. |

The dominant blocker is routing, not ASR, temporal sampling, small objects, OCR, or action recognition. Whisper produced coherent English timestamps across the full talk. Clear slide Visual search reached 83.33% Top-5, and Compositional reached 100% Top-3/5 on four queries. No new model family is justified by this acceptance run.

## Failed positives

| ID | AUTO diagnosis | Explicit-mode evidence | Human consequence |
| --- | --- | --- | --- |
| S01 | Speech motivation routed to Visual | Speech rank 2 begins “why did I get into this” | AUTO never reaches the motivation section. |
| S04 | FID warning routed to Visual | Speech rank 4 reaches the FID implementation warning | AUTO Top-5 is unrelated. |
| S05 | FLOP-counter explanation routed to Visual | Speech rank 1 is exact | AUTO Top-5 is unrelated. |
| S06 | Experiment-recording detail routed to Visual | Speech also misses the detailed answer | The circular loop mentions recording but does not answer what to store. |
| S07 | Positive/negative-results advice routed to Visual | Speech rank 1 is exact | AUTO shows premature-optimization slides. |
| S09 | Activation-selection advice routed to Visual | Speech rank 1 reaches activation functions | AUTO returns premature optimization and a different spline-customization answer. |
| S10 | Private-compute options routed to Visual | Speech also misses the compute Q&A | Neither AUTO nor the expected channel returns the usable transcript passage. |
| V04 | Visual route is correct | Explicit Visual repeats the failure | CLIP selects the one-person state-tracking grid instead of the three-figure FLOP diagram. |
| H05 | Combined state-tracking query routed to Visual | Hybrid rank 1 reaches the narrated person-and-directions example | AUTO shows a later slide reuse during a different Q&A answer, without the requested narration. |

## Partial positives

| ID | Best useful rank | Diagnosis |
| --- | ---: | --- |
| S03 | 1 | The Visual route lands in the correct paper-reading section, but the repository advice follows about one minute later. |
| S08 | 2 | Visual rank 2 reaches the later context-window explanation; explicit Hybrid rank 1 reaches the main degradation claim. |
| H03 | 2 | Hybrid ranks the preceding results plot ahead of the Obsidian graph; Speech is exact at rank 1. |
| H06 | 2 | General activation sensitivity outranks the requested spline/grid moment. |
| C04 | 3 | Speech finds the car/object example at rank 1, but visual retrieval and fusion delay the combined evidence to rank 3. |

## Negative behavior

All four confirmed-absent queries returned candidates, as expected from a product without no-match detection. None was judged misleading under the current UI. The returned frames visibly showed unrelated state-tracking, title, testing, code, experiment-loop, or activation slides; transcript excerpts helped distinguish related words from the absent request. The UI described them as ranked possible moments and made no existence claim.

This 100% rate is evidence for honest presentation on four cases. It does not establish reliable absence detection.

## Required next action

Do not modify the router, thresholds, labels, sampling, or ranking from this held-out video. Preserve the failed result. Scope exactly one source-disjoint English router-generalization milestone around natural interrogative queries and mixed evidence. A candidate must pass a separate frozen held-out gate before production consideration. Any later final acceptance must use new media and new ground truth.
