# Grounded Video Q&A V1 results

## Outcome

**Decision D — abstention safety failed.** One checksum-frozen source-disjoint validation run was completed with OpenAI `gpt-5.4-mini-2026-03-17`. Ask Video remains disabled by default. The validation set is closed to tuning and will not be rerun as a holdout.

## Data and method

- Development: one CC BY 4.0 machine-learning presentation, 12 questions (9 answerable, 3 unanswerable).
- Validation: one source-disjoint CC BY-SA 4.0 networking presentation, 12 questions (9 answerable, 3 unanswerable), 576 Whisper segments and 43 deterministic Q&A chunks.
- Frozen manifest SHA-256: `ac103e3318f9f46664867c6619daf60d65d99cd5447db236e6efd6b29cadb3a4`.
- Human review compared each answer and cited transcript text with pre-frozen key facts and intervals. No LLM judge determined correctness.

## Results

| Metric | Result | Gate | Pass? |
| --- | ---: | ---: | --- |
| Evidence Recall@1 | 77.78% | diagnostic | — |
| Evidence Recall@3 | 100.00% | diagnostic | — |
| Evidence Recall@5 | 100.00% | >=90% | yes |
| Answer Correctness | 88.89% | >=90% | no |
| Grounded Answer Rate | 80.00% | >=95% | no |
| Citation Precision | 94.44% | >=95% | no |
| Citation Recall | 91.30% | reported | — |
| Unsupported Claim Rate | 8.70% | <=5% | no |
| Correct Abstention Rate | 66.67% | >=90% | no |
| False-answer Rate | 33.33% | <=10% | no |

Retrieval median/p95 was 1.741/2.096 ms. Generation median/p95 was 1490.698/2217.642 ms. Total median/p95 was 1492.469/2219.502 ms. The validation consumed 10,371 input and 825 output tokens. At the official prices checked on 2026-09-23 ($0.75/$4.50 per million input/output tokens), estimated cost was $0.01149075.

## Interpretation

Evidence retrieval is not the V1 bottleneck: all nine answerable questions had an interval-overlapping chunk in Top-3 and Top-5. Generation safety is the blocker. `val-09` correctly identified DNS after DHCP but added a browser sequence supported only by an earlier transcript passage. `val-12`, which was verified unanswerable, returned a partial Wi-Fi comparison instead of abstaining. These two answers account for the unsupported-claim and citation-completeness failures; `val-12` makes unanswerable false-answer rate 1/3.

All 12 provider calls completed; there were zero provider errors. Upload and URL provenance use the same Q&A endpoint. Deterministic browser tests cover answer, abstention, missing configuration and citation seek behavior, but those tests are not counted as accuracy evidence.
