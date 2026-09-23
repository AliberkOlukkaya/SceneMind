# Semantic Hierarchical Q&A V1 results

## Decision

**B — SEMANTIC RETRIEVAL HELPS, SELECTOR FAILS.** The candidate improves evidence coverage and halves false abstention, but it reduces answer correctness and user success. The 44-minute source falls to 36.36% success. Ask Video remains disabled and production is unchanged.

## Data and retrieval

Development and validation each use three source-disjoint licensed English videos and 45 questions: 33 answerable and 12 hard negatives. Validation includes a 44:18 conference recording. The frozen manifest checksum is `e99fecc0c0c941afe830eb1d8ff8072212b18226096ead0bcd0b54343d068ba1`.

| Retrieval | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| BM25 | 66.67% | 78.79% | 81.82% | 93.94% |
| Semantic | 54.55% | 81.82% | 87.88% | 93.94% |
| Hybrid candidate union | 57.58% | 75.76% | 90.91% | 96.97% |

At the selected package level, BM25 versus candidate Evidence Recall@5 is 81.82% versus 90.91%; minimum-sufficient evidence recall is 78.79% versus 87.88%; mean evidence completeness is 80.30% versus 89.39%; and fully complete evidence is 78.79% versus 87.88%. The candidate improves each package measure by about 9.09 percentage points but misses every fixed evidence gate.

## Paired Q&A

| Metric | BM25 | Candidate | Change |
|---|---:|---:|---:|
| Answer Correctness | 69.70% | 63.64% | -6.06 pp |
| Core User Success | 69.70% | 63.64% | -6.06 pp |
| Grounded Answer Rate | 100% | 100% | 0 pp |
| Citation Precision | 100% | 100% | 0 pp |
| Citation Recall | 87.88% | 93.94% | +6.06 pp |
| Unsupported Claim Rate | 0% | 0% | 0 pp |
| Correct Abstention | 100% | 100% | 0 pp |
| False Answer Rate | 0% | 0% | 0 pp |
| False Abstention | 12.12% | 6.06% | -6.06 pp |

Candidate per-video success is 63.64% on RAIDZ, 90.91% on 1Lib1Ref, and 36.36% on the long Wiki Education talk. The final figure is a catastrophic source failure under the frozen rule.

## Performance and cost

The three validation indexes contain 131 fine and 44 context units. Index construction took 2.807 seconds total and persisted 438,803 bytes. Encoder plus index peak RSS increased by 542,101,504 bytes (516.99 MiB). The 44-minute source alone built in 2.051 seconds and used a 307,082-byte index.

BM25 retrieval measured 1.196/4.323 ms median/p95. Semantic retrieval measured 11.996/15.607 ms, and selection 0.865/2.980 ms. Candidate generation measured 1.762/4.236 seconds median/p95; candidate end-to-end measured 1.776/4.248 seconds. Its mean package was five units and 2,201.8 characters.

Frozen validation consumed 84,451 input and 9,610 output tokens across both paired arms and cost $0.106583. Candidate cost was $0.050262, or $0.001117 per question, versus $0.056321 for BM25. Including the valid historical replay and one explicitly discarded diagnostic scripting error, the full measured milestone cost was $0.119351.

## Historical replay

All six previous failures still received an expected-interval hit. Generation recovered two of four previous false abstentions and one of two previous incomplete answers. Because the old BM25 packages already hit the expected interval in all six cases, the replay does not show a retrieval-only cure.

## Recommendation

Do not integrate this candidate. Semantic retrieval is promising as a discovery signal, but a fixed five-unit selector does not reliably identify the right local explanation inside long, repetitive talks. Stop here for product review. The next decision must choose among a different embedding architecture, multimodal evidence, hierarchical summarization, or postponing Ask Video; this milestone does not automatically authorize another retrieval experiment.

## Validation

- Backend pytest: 258 passed, 1 opt-in test skipped.
- Ruff, ESLint, TypeScript, and Next.js production build: passed.
- Playwright: 16 desktop/mobile flows passed, 2 opt-in real-model flows skipped.
- Frozen retrieval regression selection: 99 passed.
