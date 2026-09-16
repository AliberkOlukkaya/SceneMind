# Hybrid Ranking Failure Analysis V1

## Decision

This diagnostic milestone analyzed **19 ranking failures** and **12 success controls** without changing retrieval or selecting parameters. Production remains uncapped RRF60. Cap 1.50x remains rejected on frozen Holdout V1, no further alpha tuning is authorized, and Final English Acceptance V3 has not started.

The dominant observed mechanism is **irrelevant consensus**: in 14/19 failures, four or five Hybrid Top-5 competitors receive both Visual and Speech rank contributions and displace evidence that is already present in a required modality. This is not a generic indictment of shared-thumbnail evidence. The winning candidate has two contributions in 18/19 failures and 12/12 controls. Shared evidence is normal and often useful. The failure condition is that fixed rank-only fusion treats weak or ambiguous agreement as more valuable than stronger query-relevant evidence.

The evidence indicates that a simple fixed rank-fusion formula is insufficient for the remaining failure set. It cannot use raw-score magnitude, distinguish meaningful from incidental cross-modal agreement, preserve longer causal speech context, or repair weak required-modality ranking. This finding supports a new development milestone, not an immediate production change. Holdout V1 and Acceptance V2 must remain evaluation evidence rather than design-selection data.

## Scope and governance

The analysis uses:

- seven production-baseline ranking failures from Hybrid Fusion development diagnostics;
- seven production-baseline failures from frozen Holdout V1;
- the one Cap-only Holdout regression, kept separate from production failures;
- four previously observed Final English Acceptance V2 PARTIAL/FAIL cases whose traces could be reconstructed safely.

V2 reconstruction queried the unchanged local product data for the same 50 Visual and 50 Speech candidates and passed them to the existing evaluation tracer. All **30/30 reconstructed production Hybrid Top-5 outputs** matched the historical acceptance artifact exactly by thumbnail, timestamp, score, and order. V2 queries, labels, and media were not changed. `v2-s03` is retained because human review labeled its interval hit unusable without seeking backward. `v2-v04` keeps its historical `visual retrieval` diagnosis; the deeper trace only establishes that the relevant frame was CLIP rank 13 and Hybrid rank 23.

Every trace records the expected interval, relevant candidate timestamp, raw modality ranks and scores, RRF contributions, Hybrid rank, five competitors, thumbnail key, contribution count, deduplication, rank displacement, and temporal distance. The complete machine-readable traces are in `reports/hybrid-ranking-failure-analysis-v1.json`.

## Aggregate findings

| Measure | Result |
|---|---:|
| Ranking failures | 19 |
| Success controls | 12 |
| Speech / Visual / Multimodal failures | 12 / 5 / 2 |
| Development / Holdout baseline / rejected Cap only / V2 | 7 / 7 / 1 / 4 |
| Explicit required-modality Top-5 evidence displaced outside Hybrid Top-5 | 16/19 (84.2%) |
| Winner has Visual + Speech contributions | 18/19 (94.7%) |
| Relevant candidate has a stronger same-retriever raw score than winner | 12/19 (63.2%) |
| Rank-only fusion reverses that raw-score advantage | 11/19 (57.9%) |
| Weak consensus beats strong single-modality evidence | 11/19 (57.9%) |
| Temporal attachment or context fragmentation is material | 2/19 (10.5%) |
| Confirmed duplicate-evidence inflation | 0/19 |

Primary root causes sum to the 19 distinct cases:

| Primary mechanism | Count | Interpretation |
|---|---:|---|
| `IRRELEVANT_CONSENSUS` | 14 | At least four Top-5 competitors have two-modality support and displace required evidence. |
| `WEAK_REQUIRED_MODALITY_RANK` | 2 | Relevant evidence exists in Top-50 but is already below rank 5 in every required modality. |
| `CROSS_MODAL_AGREEMENT_FAILURE` | 1 | Both modalities retrieve the interval, but adjacent thumbnail attachment prevents a jointly relevant bucket. |
| `TEMPORAL_EVIDENCE_FRAGMENTATION` | 1 | A segment at the end of a long explanation lacks the causal context needed at click time. |
| `CAP_COMPRESSION_REGRESSION` | 1 | The rejected cap moves a production success from rank 5 to rank 8. |

Secondary mechanisms are `RANK_ONLY_INFORMATION_LOSS` in 11 cases, `TEMPORAL_QUANTIZATION` in one, and `IRRELEVANT_CONSENSUS` in the Cap regression. All primary assignments are high confidence because they follow declared trace predicates. No case was assigned from query wording alone.

## Standardized failure index

The displayed modality rank is the contribution in the best jointly relevant bucket. A required modality can have an earlier interval hit attached to another thumbnail; the JSON retains both values.

| Query | Dataset | Category | Best relevant bucket contributions | Hybrid rank | Primary mechanism |
|---|---|---|---|---:|---|
| `hfd-d-s01` | Development | Speech | Visual r27, Speech r7 | 6 | Irrelevant consensus |
| `hfd-d-s03` | Development | Speech | Speech r1 | 18 | Irrelevant consensus |
| `hfd-d-s05` | Development | Speech | Speech r3 | 16 | Irrelevant consensus |
| `hfd-d-v04` | Development | Visual | Visual r27 | 29 | Weak required-modality rank |
| `hfd-d-m03` | Development | Multimodal | no jointly relevant bucket | — | Cross-modal agreement failure |
| `hfd-d-a01` | Development | Speech | Speech r1 | 7 | Irrelevant consensus |
| `hfd-h-s03` | Development | Speech | Speech r1 | 12 | Irrelevant consensus |
| `hfh-j-s03` | Holdout baseline | Speech | Speech r3 | 16 | Irrelevant consensus |
| `hfh-j-s04` | Holdout baseline | Speech | Visual r43, Speech r5 | 7 | Irrelevant consensus |
| `hfh-j-s05` | Holdout baseline | Speech | Speech r3 | 10 | Irrelevant consensus |
| `hfh-j-s08` | Holdout rejected Cap only | Speech | Visual r32, Speech r12 | 8 | Cap compression regression |
| `hfh-j-v01` | Holdout baseline | Visual | Visual r1 | 14 | Irrelevant consensus |
| `hfh-j-v02` | Holdout baseline | Visual | Visual r1 | 12 | Irrelevant consensus |
| `hfh-j-m01` | Holdout baseline | Multimodal | Visual r48, Speech r41 | 12 | Irrelevant consensus |
| `hfh-r-v02` | Holdout baseline | Visual | Visual r3 | 8 | Irrelevant consensus |
| `v2-s03` | Historical V2 | Speech | Speech r1 | 5, not useful | Temporal evidence fragmentation |
| `v2-s08` | Historical V2 | Speech | Visual r43, Speech r24 | 6 | Irrelevant consensus |
| `v2-s09` | Historical V2 | Speech | Speech r1 | 11 | Irrelevant consensus |
| `v2-v04` | Historical V2 | Visual | Visual r13 | 23 | Weak required-modality rank |

Source counts are Design Students talk 6, Human Software Extensions talk 1, Jimmy Wales interview 7, RUN documentary 1, and historical V2 talk 4.

## Hypothesis results

### Thumbnail accumulation

The hypothesized same-modality accumulation does **not** occur in production. `app.hybrid.fuse` keeps one contribution per modality per thumbnail. Later Speech segments attached to the same thumbnail are discarded. Top-5 buckets discarded one such segment across all 19 failures, compared with five across 12 controls. This is evidence against same-thumbnail Speech accumulation as a cause.

Two different modalities do add two RRF contributions. That overlap occurs in 18/19 failure winners and 12/12 control winners, so contribution count alone has no discriminating power. It becomes harmful when the agreement is incidental or only partially matches the query.

### Modality dominance and rank-only information loss

There is no fixed Visual or Speech weight that produces ordinary modality dominance; both retrievers contribute the same `1 / (60 + rank)` form. The observed dominance is structural: any two-modality bucket can beat a high-ranked single-modality bucket.

In 12/19 failures, the relevant bucket has a higher raw score than the winner within at least one shared retriever. In 11, that raw-score advantage is reversed outside Hybrid Top-5. Examples include:

- `hfd-d-a01`: relevant Speech rank 1/raw 15.6212 becomes Hybrid rank 7;
- `hfd-h-s03`: relevant Speech rank 1/raw 11.1956 becomes rank 12;
- `hfh-j-v01`: relevant Visual rank 1/raw 0.3338 becomes rank 14;
- `v2-s09`: relevant Speech rank 1/raw 11.6722 becomes rank 11 while five earlier DNS-related shared buckets occupy Top-5.

Raw scores are not calibrated across modalities, so these observations do not justify adding them directly. They do show that discarding all within-retriever confidence magnitude removes information that often separates the relevant candidate from an ambiguous shared winner.

### Temporal quantization and cross-modal agreement

Temporal representation is material in 2/19 cases:

- In `hfd-d-m03`, Speech rank 1 and Visual rank 20 both overlap the frozen interval, but nearest-frame attachment places them in adjacent thumbnail buckets. The multimodal relevance rule therefore finds no jointly relevant candidate. This is both cross-modal agreement failure and temporal quantization.
- In `v2-s03`, Speech rank 1 maps to Hybrid rank 5 at the conclusion of the packet explanation. It is inside the broad frozen interval but omits the causal size-limit context. Human acceptance review correctly marks it PARTIAL rather than useful.

These cases support better temporal evidence representation, but they are not the dominant pattern.

### Duplicate evidence inflation

No failure provides confirmed duplicate-evidence inflation. A deliberately broad 15-second proximity proxy finds a Top-5 time cluster in 9/19 failures (47.4%) and 7/12 controls (58.3%). Because the proxy is at least as common in controls, temporal proximity is not a sufficient explanation. `v2-s09` does contain several semantically related earlier DNS moments, but they occupy different thumbnails and receive independent evidence; that case is recorded as irrelevant consensus rather than proven duplication.

### Weak required-modality rank

`hfd-d-v04` (Visual rank 27) and `v2-v04` (Visual rank 13) are present within the 50-candidate diagnostic pool but are already too deep for fixed fusion to recover reliably. A fusion-only change cannot be expected to solve these two cases. The historical V2 label remains visual retrieval failure.

## Success controls

The 12 controls are balanced across Speech, Visual, and Multimodal queries (four each) and span development, Holdout, and historical V2. They were selected before computing aggregate properties.

| Property | Failures | Controls | Finding |
|---|---:|---:|---|
| Winner has two modality contributions | 18/19 (94.7%) | 12/12 (100%) | Shared evidence is normal, not sufficient as a root cause. |
| 15-second Top-5 cluster proxy | 9/19 (47.4%) | 7/12 (58.3%) | The proxy does not distinguish failures. |
| Discarded same-thumbnail Speech segments in Top-5 | 1 | 5 | Deduplication is working and does not inflate scores. |

The distinguishing failure property is semantic and ordinal: an irrelevant shared bucket receives the fixed two-vote benefit while strong required evidence receives one vote or weakly aligned second evidence. Successful queries often have shared winners too, but that agreement points to the correct interval.

## Unsupported negative query

For Holdout `hfh-r-n04` (“teacher writing the legal marriage age on a classroom chalkboard while explaining the law”), the plausible school/protest frame at 1343.37 s receives:

- Speech rank 5, raw BM25 4.2038, text “To allow it at a young age”;
- Visual rank 40, raw CLIP 0.2275;
- one RRF contribution from each modality in thumbnail `000270.jpg`.

Production baseline places it rank 4. The rejected Cap 1.50x places it rank 1 because the cap compresses stronger, more balanced overlap candidates more than this unbalanced rank-5/rank-40 pair. A second Speech segment at rank 26 maps to the same thumbnail but is discarded and contributes nothing. Thumbnail grouping creates the cross-modal pair; same-modality accumulation does not amplify it.

The raw scores contain useful uncertainty—the Visual support is rank 40 and low—but fixed rank-only fusion cannot interpret that uncertainty or infer that the result lacks teacher, chalkboard, and legal explanation. This diagnoses the convincing false result. It does not establish a no-match threshold.

## Evidence-ranked design classes

These are architecture directions only. No implementation, constant, or parameter was selected.

| Rank | Design class | Evidence coverage | Does not cover | Complexity | Regression risk | New development data |
|---:|---|---|---|---|---|---|
| 1 | Calibrated raw-score information plus explicit agreement features | Irrelevant consensus; rank-only information loss | temporal fragmentation; weak candidate rank | Medium | High | Required |
| 2 | Candidate generation followed by a second-stage reranker | consensus quality; raw-score use; cross-modal query semantics | missing required-modality candidates | High | High | Required |
| 3 | Improved temporal evidence representation | passage fragmentation; adjacent-bucket agreement | dominant consensus failures | Medium | Medium | Required |
| 4 | Grouping/dedup semantics only | adjacent-bucket agreement; temporal quantization | most consensus failures; raw-score loss; weak candidate rank | Low | Medium | Required |

The first direction has the broadest direct evidence coverage, but its scores are currently uncalibrated and the frozen evaluation sets cannot be used to design it. The second direction covers similar mechanisms with greater complexity and data demand. Temporal work is justified for a smaller, distinct subset. A grouping-only change has narrow coverage, and the rejected cap demonstrates that a simple overlap adjustment can create regressions and more convincing negative results.

## Architectural conclusion

Fixed RRF60 remains the safest production baseline because the only tested alternative failed frozen Holdout V1. It is nevertheless insufficient as a complete long-term ranking architecture for the observed tail: 16/19 analyzed cases contain explicit required-modality Top-5 evidence that does not survive usefully into Hybrid Top-5, and 11 show a measurable raw-score advantage reversed by rank-only fusion.

The next milestone should create new development data for confidence-aware agreement or reranking and define an untouched future holdout before implementation. It should also preserve a separate temporal-representation track. Do not change production retrieval, retune alpha, derive constants from Holdout V1, optimize against Final Acceptance V2, or start Final Acceptance V3 yet.

## Reproducibility

`development/analyze_hybrid_ranking_failures_v1.py` consumes the committed development and Holdout reports plus the locally reconstructed V2 trace. It produces the committed JSON summary and does not import into production. `hybrid_fusion_trace.py` still calls `app.hybrid.fuse` for authoritative ranking and asserts exact reconstruction parity.

The focused tests prove that tracing and summarization reproduce the production baseline, do not mutate candidates or traces, and preserve tie order. Full repository validation results are recorded in the project status after execution.
