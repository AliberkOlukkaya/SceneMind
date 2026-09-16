# Hybrid Fusion Refinement Experiment V1

Date: 2026-09-16

Decision: **A — a provisional development candidate exists; production remains unchanged and new holdout validation is required.**

## Scope and isolation

This offline experiment uses only the two videos and 32 frozen queries from Hybrid Fusion Diagnostics. Final English Acceptance V2 media, transcripts, queries, labels, reports, and failure rows were not evaluated or used for parameter selection. The runner rejects a diagnostic input containing the protected V2 media hash.

No production module, API contract, frontend behavior, candidate list, model, sampler, grouping rule, or deduplication rule changed. Experimental code lives under `ml/evaluation/`. The machine-readable report is [`reports/hybrid-fusion-refinement-v1.json`](reports/hybrid-fusion-refinement-v1.json).

## Baseline parity

The offline baseline independently implements the current production rule:

\[
c(r)=\frac{1}{60+r}, \qquad S(x)=\sum_{m \in M_x} c(r_{x,m})
\]

It preserves production's Visual-then-Speech insertion order, one contribution per modality per exact thumbnail, Speech timestamp/text overwrite, descending score sort, timestamp tie-break, and Top-K truncation.

Parity passed on all 32 frozen queries. Candidate IDs, timestamps, order, fusion scores, and Top-5 membership exactly equal the previously captured `app.hybrid.fuse` results. Unit tests also compare the independent baseline directly with the production function.

## Predeclared experiment families

Experiment A keeps the production rank contribution and caps a shared bucket relative to its largest individual contribution:

\[
S_\alpha(x)=
\begin{cases}
c(r), & |M_x|=1 \\
\min\left(\sum_m c(r_m),\ \alpha\max_m c(r_m)\right), & |M_x|=2
\end{cases}
\]

The predeclared grid was `α ∈ {1.75, 1.50, 1.25}`. A value of 1.50 permits at most a 50% agreement bonus over the stronger contribution. It does not force a modality quota or use raw scores.

Experiment B normalizes contributions within each retriever's returned list of length `L`. It was evaluated independently from Experiment A; no cap was applied.

Normalized RRF used `C ∈ {60, 10}`:

\[
n_C(r,L)=\frac{\frac{1}{C+r}-\frac{1}{C+L+1}}
{\frac{1}{C+1}-\frac{1}{C+L+1}}
\]

Percentile contribution used:

\[
p(r,L)=\frac{L-r+1}{L}
\]

Shared buckets sum normalized contributions. All formulas depend only on rank and generic list length. There are no learned weights, CLIP/BM25 score use, query-specific exceptions, thresholds, or acceptance-derived parameters.

## Complete comparison

Counts use 28 positives. Shared occupancy is calculated over their 140 Top-5 slots. Strong retention measures the best interval-matching Top-5 candidate from each modality required by the frozen query category; 28 such candidates exist.

| Variant | Top-1 | Top-3 | Top-5 | MRR@5 | Speech Top-5 | Visual Top-5 | Multimodal Top-5 | Rescued | Broken | Shared Top-5 | Strong retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production-equivalent RRF60 | 10 (35.7%) | 16 (57.1%) | 20 (71.4%) | 0.4827 | 6/11 | 7/8 | 7/9 | 0 | 0 | 125/140 (89.3%) | 13/28 (46.4%) |
| Cap 1.75× | 10 (35.7%) | 15 (53.6%) | 21 (75.0%) | 0.4887 | 7/11 | 7/8 | 7/9 | 1 | 0 | 125/140 (89.3%) | 14/28 (50.0%) |
| **Cap 1.50×** | **11 (39.3%)** | **16 (57.1%)** | **21 (75.0%)** | **0.5155** | **7/11** | **7/8** | **7/9** | **1** | **0** | **124/140 (88.6%)** | **14/28 (50.0%)** |
| Cap 1.25× | 11 (39.3%) | 17 (60.7%) | 20 (71.4%) | 0.5113 | 7/11 | 6/8 | 7/9 | 1 | 1 | 115/140 (82.1%) | 16/28 (57.1%) |
| Normalized RRF60 | 11 (39.3%) | 17 (60.7%) | 20 (71.4%) | 0.5071 | 8/11 | 5/8 | 7/9 | 2 | 2 | 94/140 (67.1%) | 19/28 (67.9%) |
| Normalized RRF10 | 10 (35.7%) | 18 (64.3%) | 20 (71.4%) | 0.5000 | 8/11 | 5/8 | 7/9 | 3 | 3 | 64/140 (45.7%) | 25/28 (89.3%) |
| Normalized percentile | 11 (39.3%) | 17 (60.7%) | 20 (71.4%) | 0.5071 | 7/11 | 6/8 | 7/9 | 1 | 1 | 105/140 (75.0%) | 17/28 (60.7%) |

The strong-candidate mean fusion rank improves from 6.82 at baseline to 6.18 with Cap 1.50×. Stronger normalization improves retention more, but it trades away Visual query success and creates as many new failures as rescues.

## Category effects

Cap 1.50× changes Speech Top-1/3/5 from 2/4/6 to 3/4/7 and Speech MRR@5 from 0.2833 to 0.3818. Visual remains exactly 3/5/7 with MRR@5 0.5500. Multimodal remains 5/7/7 at Top-1/3/5, while its MRR@5 moves from 0.6667 to 0.6481. The unchanged Multimodal Top-K results satisfy the primary protection rule, but the small within-Top-5 rank decline is a holdout warning and prevents any production claim.

Cap 1.25× and every normalized family reduce Visual Top-5. Normalized RRF60 and RRF10 improve Speech Top-5 to 8/11, but Visual falls from 7/8 to 5/8. They demonstrate that reducing overlap dominance can retain single-modality candidates, while also showing that aggressive correction merely moves errors between categories.

## Query-level ablation

Cap 1.50× rescues one query and breaks none:

| Outcome | Query | Category | Interval | Baseline rank | Experimental rank | Source | Shared slots before/after | Relevant bucket score before/after | Reason |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| RESCUED | `hfd-d-s01` | Speech | 318–376 s | 6 | 5 | Visual + Speech | 5 / 5 | 0.026420 / 0.022388 | The relevant shared bucket and competing shared buckets keep the same candidates; the cap reduces unequal overlap amplification enough to swap the boundary rank. |

For this bucket the unchanged rank contributions are Visual `0.011494` and Speech `0.014925`, totaling `0.026420`. The 1.50× cap limits it to `0.022388`. Its rank improves because higher competing shared buckets lose more overlap bonus. All 20 baseline successes remain successes; seven failures remain failures.

Cap 1.75× rescues the same query at rank 5 and breaks none, but lowers aggregate Top-3 and reduces MRR on the second source. Cap 1.25× also rescues `hfd-d-s01` but breaks Visual query `hfd-d-v03`, moving its relevant evidence from rank 5 to 9.

The normalization families make broader swaps:

- Normalized RRF60 rescues `hfd-d-a01` and `hfd-h-s03`, but breaks `hfd-d-v03` and `hfd-h-v01`.
- Normalized RRF10 rescues `hfd-d-s03`, `hfd-d-a01`, and `hfd-h-s03`, but breaks `hfd-d-v02`, `hfd-d-v03`, and `hfd-h-s05`.
- Percentile normalization rescues `hfd-h-s03` and breaks `hfd-d-v03`.

The JSON report contains all 28 positive query rows for every variant with `RESCUED`, `BROKEN`, `UNCHANGED_SUCCESS`, or `UNCHANGED_FAILURE`; expected intervals; before/after ranks and Top-5 state; candidate IDs; retriever sources; overlap counts; uncapped and final fusion scores; individual contributions; and cap reduction.

## Negative ordering

No no-match threshold or absence claim was introduced. With Cap 1.50×, all four negative queries retain the same Top-5 candidate set; two retain identical order and two reorder existing candidates. This is ordering evidence only and does not make unsupported searches safer.

## Per-source robustness

The two most promising clean variants were checked separately:

| Source | Variant | Top-1 | Top-3 | Top-5 | MRR@5 | Strong retained | Shared Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Design Free Software | Baseline | 4/14 | 6/14 | 7/14 | 0.3714 | 5/13 | 65/70 |
| Design Free Software | Cap 1.75× | 5/14 | 6/14 | 8/14 | 0.4214 | 5/13 | 65/70 |
| Design Free Software | **Cap 1.50×** | **5/14** | **6/14** | **8/14** | **0.4214** | **5/13** | **65/70** |
| Humans as Software Extensions | Baseline | 6/14 | 10/14 | 13/14 | 0.5940 | 8/15 | 60/70 |
| Humans as Software Extensions | Cap 1.75× | 5/14 | 9/14 | 13/14 | 0.5560 | 9/15 | 60/70 |
| Humans as Software Extensions | **Cap 1.50×** | **6/14** | **10/14** | **13/14** | **0.6095** | **9/15** | **59/70** |

Cap 1.50× gets its Top-5 gain from the first video, but it does not damage Top-1/3/5 on the second and improves its MRR and strong retention. Cap 1.75× regresses the second source's Top-1, Top-3, and MRR. The evidence therefore prefers 1.50× among the predeclared configurations without claiming that the decimal is universally optimal.

## Selection and next action

Cap 1.50× is the best **development candidate**. It improves aggregate Top-1, Top-5, MRR@5, Speech Top-1/5, strong-candidate retention, and mean strong-candidate rank; preserves Visual and Multimodal Top-1/3/5; produces no newly broken query; and avoids per-video Top-K regression. It is a two-line, rank-only rule with no model, raw-score coupling, learned parameter, modality exception, or query classifier.

Experiment C was not run. Experiment A already provides a clean improvement under the predeclared rule, whereas a modality floor would add a new mechanism and selection surface without evidence that it is needed.

Uncertainty remains substantial. There are only two related presentation-style development sources, one net rescued query, a small Multimodal MRR decline, and almost unchanged overlap occupancy. The cap can alter negative ordering and has not been evaluated on a new domain. These results justify one new source-disjoint holdout evaluation with metrics and gates frozen in advance. They do not justify changing `app.hybrid.fuse`, Smart Search, or declaring the search core frozen.

## Reproduction

```powershell
.venv/Scripts/python -m ml.evaluation.run_hybrid_fusion_refinement_v1
.venv/Scripts/python -m pytest tests/test_hybrid_fusion_refinement_v1.py
```

The experiment reuses only committed diagnostic candidate traces; it does not require or read ignored media during ranking.
