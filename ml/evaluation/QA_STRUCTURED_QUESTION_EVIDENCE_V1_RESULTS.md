# Q&A Structured-Question Evidence V1 Results

## Decision

**D — SAFETY REGRESSION.** The bounded expansion is rejected and Ask Video remains disabled. The one frozen validation run failed every promotion gate except ordinary-question non-regression. It is not eligible for Final Ask Video Acceptance V1.

## Data and freeze

Development used 36 questions over two licensed English sources; validation used 36 questions over two different licensed English sources. Each split contains 10 ordinary answerable, 8 list/count, 8 temporal and 10 hard-negative questions. The four sources are disjoint from one another and from both prior Q&A suites.

The selected development candidate adds at most three list neighbors or two requested-direction temporal segments, with eight total evidence units, 6,000 characters and 90 seconds as hard bounds. The manifest checksum is `27e43fb83b7d039abe160828d3d0f19f9e2b56a2ef223a78e6bde7f4f8c85d9f`. Validation was executed once with `gpt-5.4-mini-2026-03-17`; no second run occurred because the first failed.

## Evidence

| Metric | Frozen validation |
| --- | ---: |
| Base evidence recall | 92.31% |
| Expanded evidence recall | 92.31% |
| Structured base all-interval recall | 93.75% |
| Structured expanded all-interval recall | 93.75% |
| List evidence completeness | 87.50% |
| Temporal anchor recall | 75.00% |
| Temporal target recall | 87.50% |
| Temporal pair recall | 75.00% |

Expansion did not improve frozen evidence recall. It correctly preserved ordinary evidence, but semantic anchor errors caused a wrong flour-related anchor in the pasta video and a wrong European-English anchor in the language talk.

## Answer quality

| Metric | Result | Gate | Pass? |
| --- | ---: | ---: | :---: |
| Overall answer correctness | 72.22% | >=90% | No |
| Grounded answer rate | 75.00% | >=95% | No |
| Citation precision | 85.45% | >=95% | No |
| Citation recall | 72.97% | >=90% | No |
| Unsupported claim rate | 20.75% | <=5% | No |
| Correct abstention | 70.00% | >=90% | No |
| False answer rate | 30.00% | <=10% | No |
| False abstention rate | 3.85% | measured | — |

The protected Safety V1 characteristics did not survive source-disjoint validation. Three of ten hard negatives received confident false answers: a nonexistent pasta-maker step, stock and peas mislabeled as meats, and locale names invented as benchmark results.

## Structured categories

| Metric | Result | Gate | Pass? |
| --- | ---: | ---: | :---: |
| List/count correctness | 62.50% | >=85% | No |
| Supported list-item precision | 81.25% | — | — |
| Supported list-item recall | 68.42% | — | — |
| Requested-count compliance | 75.00% | — | — |
| Temporal correctness | 62.50% | >=85% | No |
| Temporal direction accuracy | 75.00% | >=95% | No |
| Temporal citation correctness | 62.50% | — | — |
| Ordinary correctness | 90.00% | >=90% | Yes |
| Hard-negative false-answer rate | 30.00% | <=10% | No |

List failures came from non-distinct item splitting, one false abstention and treating an explanation of one reason as a second reason. Temporal failures came from semantic anchor mismatch, selecting the wrong local target and dedicated anchor/target citations that did not represent the answer's actual relation.

## Performance and budget

| Metric | Median | p95 |
| --- | ---: | ---: |
| Base retrieval | 0.637 ms | 0.918 ms |
| Structured expansion | 0.087 ms | 4.831 ms |
| Generation | 1,734.595 ms | 2,790.063 ms |
| Total | 1,735.795 ms | 2,794.748 ms |

Average evidence count grew from 5.00 to 6.78 overall (+35.56%). Structured questions grew from 5.00 to 7.94 (+58.75%) and added 835 characters on average. The run used 48,980 input and 3,832 output tokens, with an estimated API cost of $0.053979 at the documented $0.75/M input and $4.50/M output rates.

## Historical replay

The four old structured failures were replayed only after Decision D froze and are excluded from validation metrics. One of two old list failures and one of two old temporal failures were correct. The other list answer still omitted a required design guideline, and the other temporal answer returned “oscilloscope” instead of the timeline-node proposal. This diagnostic does not justify promotion.

## Product consequence

`SCENEMIND_QA_ENABLED=false` remains the required product state. BM25 Top-5, transcript chunking, Whisper, CLIP, FAISS, RRF60 and production search remain unchanged. A future attempt would need new development and validation sources and a bounded premise-validity plus semantic-anchor safety design; the current expansion must not be promoted or tuned on this now-observed validation set.

The machine-readable report is `ml/evaluation/reports/qa-structured-question-evidence-v1.json`.
