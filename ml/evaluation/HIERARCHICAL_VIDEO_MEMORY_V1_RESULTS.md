# Hierarchical Video Memory V1 results

## Decision

**C — LOCAL EVIDENCE / ANSWERING STILL FAILS.** Persistent hierarchical navigation found the relevant section in the Top 3 for 90.91% of supported questions and for 100% of the 40-minute video, but the final transcript package reached only 84.85% completeness and final Core User Success was 57.58%. BM25 remained materially better at 96.97% evidence completeness and 78.79% Core User Success. The 22-minute GM video fell to 27.27%, below the pre-registered 70% catastrophic threshold. Ask Video remains disabled and production is unchanged.

## Data and frozen configuration

Three new development videos total 75:07 and three source-disjoint validation videos total 77:25. Each split contains 45 questions: 33 answerable and 12 hard negatives. Development includes a 40:11 video; validation includes 40:11, 22:10 and 15:03 videos. The frozen manifest checksum is `7bf8dd40e06c41c5c478e4828bd394a9ba0fdff6a703a6c5f595e47c7dd8b9d1`.

Development selected the compact configuration from three pre-registered candidates. It creates an L2 boundary after at least 35 seconds when an adjacent pause is at least 3 seconds or pinned-MiniLM adjacent cosine falls below 0.40, with a hard 180-second cap. Section retrieval K is three. L3 uses a deterministic local extractive centroid summary of at most two original L1 units and 420 characters plus source-frequency topics. No preprocessing API was used.

Validation produced 46 sections, averaging 15.33 per video and 109.15 seconds. All 46 summaries were reconstructed from their recorded source L1 IDs; all topic tokens occurred in their source section. Factual consistency was 100%, with zero unsupported summary statements. This audit does not make summaries evidence: only original timestamped L1 text was sent to the answer generator or accepted for citation.

## Section navigation and evidence

| Metric | Result |
| --- | ---: |
| Relevant Section R@1 | 75.76% |
| Relevant Section R@3 | 90.91% |
| Wrong-section rate | 9.09% |
| 40-minute Section R@1 / R@3 | 90.91% / 100.00% |

| Evidence metric | BM25 | Flat semantic | Hierarchical |
| --- | ---: | ---: | ---: |
| Recall@1 | 81.82% | 72.73% | 72.73% |
| Recall@3 | 93.94% | 90.91% | 84.85% |
| Recall@5 | 96.97% | 93.94% | 84.85% |
| Minimum sufficient recall | 96.97% | 93.94% | 84.85% |
| Mean package completeness | 96.97% | 93.94% | 84.85% |
| Fully complete packages | 96.97% | 93.94% | 84.85% |

Hierarchy therefore did not improve the BM25 evidence package. Its section stage worked on the long source, but restricting local retrieval to three sections removed required evidence on five questions.

## Answer quality and safety

| Metric | BM25 | Flat semantic | Hierarchical |
| --- | ---: | ---: | ---: |
| Answer Correctness | 78.79% | 57.58% | 57.58% |
| Core User Success | 78.79% | 57.58% | 57.58% |
| Grounded Answer Rate | 100.00% | 100.00% | 100.00% |
| Citation Precision | 100.00% | 100.00% | 100.00% |
| Citation Recall | 81.82% | 75.76% | 69.70% |
| Unsupported Claim Rate | 0.00% | 0.00% | 0.00% |
| Correct Abstention | 100.00% | 100.00% | 100.00% |
| False Answer Rate | 0.00% | 0.00% | 0.00% |
| False Abstention Rate | 18.18% | 24.24% | 30.30% |

Hierarchical per-video Core User Success was 72.73% on the 40-minute ACIP opening, 27.27% on the 22-minute GM dedication and 72.73% on the 15-minute copyright lecture. The 40-minute breakdown passes the catastrophic threshold, but the GM source blocks promotion. Safety remained intact; coverage and usefulness did not.

## Performance and cost

The three validation memories took 5.07 seconds total after model load and occupy 663,568 bytes. Per-source section/memory/index persistence was 2.96 seconds for the 40-minute video, 1.24 seconds for the 22-minute video and 0.87 seconds for the 15-minute video. Reload after a simulated restart took 38–51 ms. The measured added peak RSS, including the local encoder and all comparison indexes, was 559,001,600 bytes.

Hierarchical section retrieval was 5.66/6.65 ms median/p95; local evidence selection was 0.51/0.83 ms; complete hierarchical retrieval was 6.24/7.27 ms. Generation was 1.661/3.597 seconds and total query time 1.667/3.603 seconds median/p95.

Local preprocessing used zero API tokens and cost $0. The 135 paired answer generations used 104,663 input and 13,835 output tokens, costing $0.140755 at the frozen price: $0.001043 per generated arm-query and $0.10910 per validation video-hour. Cached memory incurs no repeat preprocessing charge.

Repository validation passed: 67 focused tests; 268 full pytest tests with one opt-in skip; Ruff; ESLint; TypeScript; Next.js production build; 16 Playwright tests with two opt-in real-model skips; and 92 frozen retrieval/Q&A regressions. No historical expected metric changed.

## Historical replay

Replay occurred only after Decision C froze. The candidate recovered 3 of the 6 Final Ask Video Core Acceptance failures; all six had expected evidence in the package. On the previous 44-minute source it found expected evidence for 5/11 answerable questions and reached only 3/11 (27.27%) Answer Correctness/Core User Success. The hierarchy therefore did not repair the prior catastrophic long-video case.

## Recommendation

Do not integrate this branch, do not enable Ask Video, and do not start Hierarchical Memory V2 automatically. The evidence supports a separate product decision: either redesign multi-section local evidence assembly and answer completeness as a new architecture with new data, or postpone Ask Video. Do not tune this observed validation set. Find Moments and production BM25 remain unchanged.

The machine-readable report is [`reports/hierarchical-video-memory-v1.json`](reports/hierarchical-video-memory-v1.json), and individual failures are in [`HIERARCHICAL_VIDEO_MEMORY_V1_FAILURES.md`](HIERARCHICAL_VIDEO_MEMORY_V1_FAILURES.md).
