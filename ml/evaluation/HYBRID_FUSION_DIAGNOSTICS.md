# Hybrid fusion diagnostics

Date: 2026-09-16
Decision: **diagnosis complete; production unchanged**

## Scope and isolation

This is a development diagnostic of the existing production retrieval path. It does not tune or change RRF, candidate limits, modality weights, CLIP, Whisper, BM25, sampling, temporal grouping, API behavior, or the frontend. The Final English Acceptance V2 media, transcript, queries, labels, and failure rows were not used. An exact-query leakage check against its frozen manifest is enforced by the runner and tests.

Two existing, unrelated CC BY 4.0 English development sources were found in the repository, so no media was downloaded:

| Source | Duration | SHA-256 | Production processing |
| --- | ---: | --- | --- |
| Design Students Experimenting with Free Software | 22:23.66 | `f389e9911d1583b63b5286967675a189c152421f91dd5f0ff0a158270f5dcb74` | 269 frames, 109 Whisper segments, 98.750 s |
| Humans as Software Extensions | 26:29.28 | `2bc604a1ad04d1ca5537eb3348d691a3e1e2a845c114c49d97ccd923e8d880dc` | 318 frames, 273 Whisper segments, 100.963 s |

Both sources previously served only as human-grounded router development/validation material. Existing complete five-second frame and local transcript reviews were rechecked to define evidence intervals. The 32-query development manifest was frozen before retrieval: 11 Speech, 8 Visual, 9 Multimodal, and 4 negative queries. It includes lexical, semantic paraphrase, visual state/object/composition, genuinely multimodal, ambiguous, and unsupported cases.

The machine-readable report is [`reports/hybrid-fusion-diagnostics.json`](reports/hybrid-fusion-diagnostics.json). It contains the exact production responses plus every raw candidate, original rank/score, RRF contribution, exact-thumbnail overlap, grouping/deduplication effect, final score/rank, displacement, and Top-K membership. The report is deliberately verbose so an individual row can be audited.

## Method

For every query the runner called the normal HTTP search endpoint in explicit Speech, Visual, and Hybrid modes. Speech and Visual were requested at 50 candidates so the candidate pool could be inspected; Hybrid was requested at the product's five results. The evaluation-only tracer then called `app.hybrid.fuse` for the authoritative result and independently reconstructed the fusion trace. It aborts unless IDs, timestamps, scores, and order match production exactly.

A result matches a frozen development interval when its relevant contribution overlaps that interval. Speech queries require a Speech contribution, Visual queries a Visual contribution, and Multimodal queries require both. This is interval-based diagnostic evidence, not a new claim of user usefulness. Negative queries are excluded from positive recall and are retained to expose that current retrieval has no abstention mechanism.

## Quantitative results

Across the 28 positive queries:

| Path | Top-1 | Top-3 | Top-5 | Top-50 candidate recall |
| --- | ---: | ---: | ---: | ---: |
| Speech | 18/28 (64.3%) | 23/28 (82.1%) | 25/28 (89.3%) | 28/28 (100%) |
| Visual | 9/28 (32.1%) | 11/28 (39.3%) | 14/28 (50.0%) | 24/28 (85.7%) |
| Hybrid | 10/28 (35.7%) | 16/28 (57.1%) | 20/28 (71.4%) | 26/28 (92.9%) |

The modality-aware category slice is the more meaningful view:

| Frozen query category | Queries | Speech Top-5 | Visual Top-5 | Hybrid Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Speech | 11 | 10 (90.9%) | 4 (36.4%) | 6 (54.5%) |
| Visual | 8 | 7 (87.5%) | 6 (75.0%) | 7 (87.5%) |
| Multimodal | 9 | 8 (88.9%) | 4 (44.4%) | 7 (77.8%) |

The cross-path Speech/Visual columns can match an interval for incidental content at that time; they are diagnostic context, not evidence that the wrong modality answers the query semantically.

The 28 positive outcomes split into 20 Hybrid Top-5 successes, 7 ranking/exact-overlap failures where relevant evidence existed in an input list, and 1 candidate-recall failure. Thus 25.0% of positives are ranking/grouping failures and 3.6% are candidate recall failures under this protocol.

The trace found 13 cases where a relevant Speech Top-5 candidate fell below Hybrid Top-5 and 8 analogous Visual cases. Eight of the Speech demotions occur on Speech queries. The clearest losses were Speech rank 1 to Hybrid ranks 18, 12, 8, and 7; another Speech rank 5 became Hybrid rank 16. Visual rank 1 fell to Hybrid ranks 13, 10, 9, and 7 in four cases. Some demoted candidates coexist with a different relevant Hybrid result, so these counts measure displacement rather than query failure.

Of 160 final Hybrid slots, 135 (84.4%) were exact-thumbnail overlaps, 17 were Visual-only, and 8 were Speech-only. Twenty-four of 32 queries filled all five final slots with shared candidates. There were 224 shared buckets in the complete traced pools. Nine queries experienced within-thumbnail Speech deduplication, discarding 19 later transcript segments.

Observed request latency in milliseconds:

| Path | All-query median | All-query p95 | After first query/source median | After first query/source p95 |
| --- | ---: | ---: | ---: | ---: |
| Speech | 11.43 | 19.11 | 11.28 | 18.97 |
| Visual | 26.05 | 7257.17 | 25.82 | 35.45 |
| Hybrid | 37.54 | 47.72 | 37.54 | 47.72 |

The Visual all-query p95 includes one lazy model/index initialization per API process. Processing measurements include 587 sampled frames, 382 transcript segments, 199.713 seconds total processing, about 3.23 GiB maximum combined worker peak if the two independent peaks are summed, and zero temporary residue. The runs were sequential; summed peak memory is therefore not concurrent usage.

## Answers to the diagnostic questions

**A. Are strong Speech candidates commonly demoted?** Yes. Thirteen relevant Speech Top-5 candidates were displaced below Hybrid Top-5, including eight on the eleven Speech queries. Speech Top-5 drops from 90.9% in explicit Speech to 54.5% in Hybrid for that category.

**B. Are strong Visual candidates commonly demoted?** Yes, though less often in query-level outcomes. Eight relevant Visual Top-5 candidates were displaced. Hybrid still improves the Visual category from six to seven Top-5 successes because coincident Speech evidence rescues some moments.

**C. Which contributions cause displacement?** Exact-thumbnail dual contributions dominate. A shared bucket receives `1/(60 + visual_rank) + 1/(60 + speech_rank)`, while a single-modality bucket receives only one term. Raw CLIP and BM25 score magnitudes never participate after their separate rankers produce ordered lists.

**D. Is RRF or pre-RRF candidate generation the likely problem?** Both occur, but ranking is dominant in this set. Twenty-seven of 28 positives have the required evidence in at least one top-50 input list, while seven lose it from Hybrid Top-5. Only one is a candidate-recall failure. The current unweighted RRF plus exact-thumbnail grouping is therefore the main next test target; this result does not establish that RRF as a general method is defective.

**E. Does overlap create an unfair advantage?** The current formula gives it a deterministic structural advantage. With both lists capped at 50, the weakest possible shared score is `2/(60+50) = 0.01818`, greater than the strongest possible single score `1/(60+1) = 0.01639`. Every shared bucket therefore sorts above every single bucket, regardless of raw relevance scores. The observed 84.4% shared Top-5 occupancy matches that consequence. Whether each overlap is semantically fair requires human relevance judgment; exact proximity alone does not prove corroboration.

**F. Does one modality systematically dominate?** Neither modality's single candidates dominate: shared buckets do. Among the remaining 25 single-modality Top-5 slots, Visual has 17 and Speech 8. This is partly because Visual always supplies 50 candidates while BM25 may return fewer nonzero segments.

**G. Are lexical/BM25 candidates disproportionately affecting ranks?** BM25 candidates have strong influence whenever they map to a Visual thumbnail because their RRF term creates a shared bucket. BM25 magnitude is ignored after rank assignment. This can promote lexical evidence even when the image match is generic, but the run does not prove all such promotions are wrong.

**H. Do temporal grouping or deduplication change good ranks?** Yes mechanically. Speech segments are assigned to the nearest sampled thumbnail, making exact-thumbnail overlap possible. Only the first contribution per modality and thumbnail counts; 19 later Speech segments were discarded across nine queries. The trace exposes these effects, but this dataset does not isolate how many discarded segments would independently have improved relevance.

**I. Is the issue query-type dependent?** Yes. Speech is most affected: Hybrid Top-5 is 54.5%, versus 87.5% for Visual and 77.8% for Multimodal. The design is naturally favorable when both modalities truly corroborate a moment and risky for speech-only concepts paired with generic presentation frames.

**J. Recall or ranking?** Primarily ranking/grouping: 7 of 28 positives versus one candidate-recall failure. Visual candidate generation is still weaker than Speech, reaching the required evidence within its top 50 for 24 of 28 positives.

## Observations, hypotheses, and next tests

The measured facts are the fixed formula, 84.4% shared-slot occupancy, 21 strong-candidate demotions, and the 20/7/1 success/ranking/recall split. The leading hypothesis is that exact-thumbnail overlap is treated as stronger cross-modal agreement than the evidence warrants. A nearby transcript and a generic slide can coincide without both supporting the query.

Evidence supports testing a fusion change on development data, but it does not support a production change yet. The smallest next experiments are:

1. Preserve a small per-modality quota or union before final truncation so rank-1 Speech or Visual evidence cannot be removed solely by overlap buckets.
2. Compare exact-thumbnail overlap against a bounded temporal match with explicit contribution provenance, while preventing two weak ranks from automatically beating every rank-1 single.
3. Test a normalized rank contribution or a capped overlap bonus, selected only on development sources and evaluated once on a new frozen source-disjoint set.
4. Preserve or expose multiple Speech segments within a grouped thumbnail during evaluation so deduplication does not hide the best transcript excerpt.

No candidate is implemented here. A later implementation should define gates before fitting, keep V2 acceptance isolated, and require a new frozen evaluation source before promotion.

## Reproduction

```powershell
.venv/Scripts/python -m ml.evaluation.run_hybrid_fusion_diagnostics
.venv/Scripts/python -m pytest tests/test_hybrid_fusion_trace.py tests/test_hybrid_fusion_diagnostics.py
```

The ignored product roots under `data/hybrid-fusion-diagnostics/` contain local media-derived indexes and databases and must not be committed.
