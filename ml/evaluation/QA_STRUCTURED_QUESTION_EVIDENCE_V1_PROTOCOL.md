# Q&A Structured-Question Evidence V1 Protocol

## Objective

Improve list/count and temporal Q&A usefulness without weakening the frozen abstention-safety characteristics. Ask Video remains disabled throughout this milestone. The experiment changes only the evidence supplied after the unchanged BM25 Top-5 retrieval step.

## Data and isolation

The development split contains 36 questions over two licensed English sources: *Using good sources on Wikipedia* and *Blended learning*. The validation split contains 36 questions over two different licensed English sources: *Cooking with Laura: Homemade Pasta* and *Let's change the default language of the Internet*.

Each split contains 10 ordinary answerable, 8 list/count answerable, 8 temporal answerable and 10 hard-negative questions. Questions, answerability, key facts, requested counts, temporal anchor/target intervals and expected evidence intervals were written from full local transcript review before any run on that split. Both prior Q&A validation suites and their sources remain historical diagnostics and are excluded from development and validation.

## Frozen baseline

- deterministic 45-second / 900-character overlapping transcript chunks
- BM25 Top-5 retrieval
- `gpt-5.4-mini-2026-03-17`
- Responses API defaults, strict JSON schema and `store=false`
- strict claim evidence, exact explicit list count and safe abstention contract
- Whisper, CLIP, FAISS, RRF60 and all search paths unchanged

## Predeclared development candidates

Two bounded configurations are compared only on development data:

| Candidate | List additions | Temporal neighbors |
| --- | ---: | ---: |
| A | 2 | 1 |
| B | 3 | 2 |

Both cap total evidence at eight chunks, structured context at 6,000 characters and temporal expansion at 90 seconds. If retrieval is tied, answer completeness and then the smaller context decide selection.

## Structured evidence

Ordinary questions retain the exact BM25 Top-5 evidence and ordering. List/count questions add deduplicated previous/next chunks around promising Top-5 evidence, subject to the frozen budget.

Temporal questions extract the requested BEFORE/AFTER relation and anchor phrase. Anchor localization uses the same BM25 lexical machinery plus deterministic normalized-token coverage over one- or two-segment transcript windows. Only the immediately requested-direction transcript segments are added. Anchor and target evidence receive separate roles and timestamps. The server rejects missing roles, unknown IDs, wrong direction, incomplete requested counts, duplicate list claims and unsupported provider contracts.

## Metrics

Retrieval metrics are base and expanded evidence recall, list evidence completeness, temporal anchor recall, temporal target recall and temporal pair recall. Human review records answer correctness, grounded answer rate, citation precision/recall, unsupported claims, abstention behavior, list item precision/recall and count compliance, temporal direction and temporal citation correctness.

Performance records retrieval, expansion, generation and total median/p95 latency, evidence counts, added characters, temporal span, input/output tokens and estimated API cost using the already documented $0.75/M input and $4.50/M output rates.

## Freeze and execution

After development selection, the manifest, prompt, rules, limits, model and answerability contract are checksum-frozen. The validation split is run exactly once. No validation question or algorithm is changed after output is observed. Historical failures are replayed only after the decision and are excluded from validation metrics.

## Gates and decision

The required gates are at least 90% overall correctness, 95% grounded answers, 95% citation precision, 90% citation recall, at most 5% unsupported claims, at least 90% correct abstention, at most 10% false answers, at least 85% list/count correctness, at least 85% temporal correctness, at least 95% temporal direction accuracy, at most 10% hard-negative false answers and no material ordinary regression.

The final result is exactly one of A (passes), B (list evidence failure), C (temporal evidence failure), D (safety regression) or E (no generalizable improvement). Even Decision A only makes the feature eligible for Final Ask Video Acceptance; it does not enable Ask Video.
