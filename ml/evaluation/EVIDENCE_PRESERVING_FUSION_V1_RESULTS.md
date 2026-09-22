# Evidence-Preserving Hybrid Fusion V1 Results

Decision: **B — IMPROVES BUT MISSES ACCURACY GATES**

Production remains **Smart Search → uncapped RRF60**. No protected holdout was consumed.

## Development selection

Eight configurations were run on 28 positive and four negative queries from two explicit
development sources.

| Configuration | Top-1 | Top-3 | Top-5 | MRR@5 | Displacements | Irrelevant consensus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RRF60 | 10 | 16 | 20 | 0.4827 | 5 | 5 |
| quota 1 | 8 | 20 | 22 | 0.4982 | 3 | 1 |
| quota 2 | 8 | 17 | 20 | 0.4560 | 3 | 2 |
| interleave Visual first | 7 | 17 | 17 | 0.4286 | 5 | 0 |
| interleave Speech first | 13 | 17 | 19 | 0.5500 | 3 | 2 |
| preserve depth 1 | 8 | 20 | 22 | 0.4982 | 3 | 1 |
| preserve depth 2 | 8 | 17 | 20 | 0.4679 | 3 | 1 |
| preserve depth 3 | 8 | 17 | 18 | 0.4536 | 4 | 2 |

The predeclared selection order chose `quota_1`. It reserves the first unique bucket from each
retrieval lane, fills remaining Top-5 capacity in production RRF order, keeps exact-thumbnail
grouping and uses no raw scores. The frozen JSON specification SHA-256 is
`5e427a293a07968f5a650cf3aeef4ed974ad8a240974cf11a85de64d4d209b6f`.

## Source-disjoint validation

All 48 historical RRF60 Top-5 outputs reproduced before evaluation. Metrics below use the 24
positive queries; 24 confirmed negatives retain the same possible-match presentation.

| Metric | RRF60 | Frozen quota 1 | Delta |
| --- | ---: | ---: | ---: |
| Useful Top-1 | 19/24 = 79.17% | 20/24 = 83.33% | +1 |
| Useful Top-3 | 21/24 = 87.50% | 22/24 = 91.67% | +1 |
| Useful Top-5 | 21/24 = 87.50% | 22/24 = 91.67% | +1 |
| MRR@5 | 0.8333 | 0.8681 | +0.0347 |

The candidate recovers `talk-s-languages` from outside Top-5 to rank 1, introduces no Top-5
regression, and retains 21/21 baseline successes. Explicit-modality displacement falls from 1/19
(5.26%) to 0/19, and irrelevant-consensus failures fall from one to zero.

## Category behavior

| Category | RRF60 Top-1/3/5 | Candidate Top-1/3/5 | MRR change |
| --- | --- | --- | ---: |
| Speech | 3/4/4 of 6 | 4/5/5 of 6 | +0.1667 |
| Visual | 12/12/12 of 12 | 12/12/12 of 12 | 0 |
| Multimodal/Hybrid | 4/5/5 of 6 | 4/5/5 of 6 | **−0.0278** |

The candidate meets numeric Top-3/Top-5 and Top-1 gates, but the improvement is exclusively Speech.
Visual does not improve, and Multimodal MRR regresses because `talk-h-airport` moves from rank 1 to
rank 3. The predeclared rule forbids promotion when improvement comes from one category only.

## Query-level changes

| Query | Category | RRF60 rank | Candidate rank | Classification |
| --- | --- | ---: | ---: | --- |
| `talk-s-change` | Speech | 2 | 1 | Neutral reorder |
| `talk-s-bug` | Speech | 1 | 2 | Neutral reorder |
| `talk-s-languages` | Speech | outside 5 | 1 | **Recovery** |
| `talk-h-change` | Hybrid | 2 | 1 | Neutral reorder |
| `talk-h-airport` | Hybrid | 1 | 3 | Neutral reorder with severity cost |

Three negative queries changed their top candidate. Because the product continues to label these as
possible matches and the dataset has no comparative misleadingness judgment between absent-content
candidates, no claim of improved negative behavior is made.

## Efficiency and decision

Fusion overhead over all 48 validation queries was 0.592/0.824 ms median/p95, or
0.0123/0.0172 ms per query. The frozen experimental artifact is 18,357 bytes. Production memory and
artifact impact are zero.

Decision **B** is required: aggregate accuracy and displacement improve, but cross-category
generalization fails. Hybrid Holdout V1 remains unconsumed. Do not integrate the candidate and do
not start another fusion variant automatically.
