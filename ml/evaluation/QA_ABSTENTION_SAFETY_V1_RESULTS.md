# Q&A Abstention Safety V1 results

**Decision D — temporal / structured question failure.** The new answerability and claim contract fixes the observed false-answer and unsupported-claim safety problem, but answer correctness and structured-category reliability miss frozen gates. Ask Video remains disabled and is not eligible for enablement.

## Data and run integrity

- Development: 2 licensed English sources, 30 questions, 18 answerable, 12 unanswerable, 10 hard negatives.
- Validation: 2 different licensed English sources, 30 questions, 18 answerable, 12 unanswerable, 10 hard negatives.
- Development, validation, previous Q&A validation, speakers/topics, and source hashes are disjoint.
- Frozen manifest: `26646d2bf977981950e68888898522e348e6019cc39acac417a8a28acca34255`.
- One validation run; no post-run tuning or second run.

## Validation metrics

| Metric | Result | Gate | Pass |
|---|---:|---:|:---:|
| Evidence Recall@5 | 94.44% (17/18) | >= 90% | yes |
| Answer Correctness | 77.78% (14/18) | >= 90% | no |
| Grounded Answer Rate | 100.00% (16/16) | >= 95% | yes |
| Citation Precision | 100.00% | >= 95% | yes |
| Citation Recall | 100.00% | >= 90% | yes |
| Unsupported Claim Rate | 0.00% | <= 5% | yes |
| Correct Abstention | 100.00% (12/12) | >= 90% | yes |
| False Answer Rate | 0.00% (0/12) | <= 10% | yes |
| False Abstention Rate | 11.11% (2/18) | reported | — |
| Hard-negative false-answer rate | 0.00% (0/10) | acceptable | yes |

All 16 substantive answers were fully supported at the claim level. Four answerable questions were not correct: two false abstentions, one incomplete three-item guideline answer, and one incomplete decision-rule answer.

## Category results

| Frozen category | Questions | Correctness / safety |
|---|---:|---:|
| Factual | 5 answerable | 100.00% |
| Explanation | 9 answerable | 88.89% |
| List/count | 2 answerable | 0.00% |
| Temporal | 2 answerable | 50.00% |
| Hard negative | 10 unanswerable | 100.00% correct abstention; 0.00% false answer |
| Ordinary unanswerable | 2 unanswerable | 100.00% correct abstention; 0.00% false answer |

The frozen list/count group is a catastrophic category failure. Temporal ordering is not systematically fabricated, but one of two valid temporal questions is refused despite relevant Top-5 evidence. A third count-shaped question was frozen under `FACTUAL`; it succeeded but is not moved between categories after validation.

## Performance and cost

Retrieval latency was 1.735 ms median and 2.113 ms p95. Generation latency was 1,505.082 ms median and 2,024.027 ms p95; total latency was 1,506.851 ms median and 2,025.899 ms p95. The validation used 33,287 input and 2,902 output tokens. At the recorded official price of $0.75/M input and $4.50/M output tokens, estimated validation cost was **$0.03802425**.

The selected development run used 20,735 input and 2,581 output tokens, with 1,294.274/1,836.456 ms generation median/p95 and an estimated cost of $0.02716575.

## Historical replay

After the new validation decision was frozen, the old `val-09` temporal pattern and `val-12` mobile-data pattern were replayed for diagnostics only. Both produced the standard safe abstention. They are excluded from every metric above and do not convert the decision to a pass.

## Engineering validation

- Backend pytest: **227 passed, 1 skipped**, with two known upstream deprecation warnings.
- Ruff lint: passed for `backend`, `ml/evaluation`, and `tests`.
- Ruff format: all five changed Python files passed.
- Frontend ESLint: passed.
- TypeScript `tsc --noEmit`: passed.
- Next.js production build: passed.
- Playwright: **14 passed, 2 opt-in real-model tests skipped**.
- Focused frozen Q&A/Natural V2/Acceptance V2/Hybrid Holdout/evidence-preserving retrieval regressions: **44 passed**.

## Decision

Decision D is required because Answer Correctness is below 90%, frozen list/count correctness is 0/2, temporal correctness is 1/2, and the no-catastrophic-category gate fails. The exact next milestone is **Q&A Structured-Question Evidence V1 on new development sources**, focused on representing count/list completeness and temporal anchors without changing the frozen retrieval baseline or tuning this validation. `SCENEMIND_QA_ENABLED=false` remains mandatory.
