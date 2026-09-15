# Human-grounded English AUTO router failures

The fixed character candidate makes 24 errors in 60 frozen queries. Two errors occur on the four annotations flagged ambiguous before evaluation; the remaining 22 are model errors. Frozen labels were not changed after predictions were observed.

## Error pattern

| Slice | Queries | Errors | Accuracy |
|---|---:|---:|---:|
| Natural questions | 31 | 12 | 61.29% |
| Indirect intent | 23 | 14 | 39.13% |
| Pre-flagged ambiguity | 4 | 2 | 50.00% |
| Genuine multimodal requirement | 20 | 2 | 90.00% |
| Long queries (14+ tokens) | 4 | 0 | 100.00% |
| Short queries (≤6 tokens) | 12 | 10 | 16.67% |
| Unseen-vocabulary rows | 60 | 24 | 60.00% |

The dominant failure is cross-source query-form transfer. Training contains many bare Visual descriptions, Speech questions, and explicit Hybrid conjunctions. The clean test deliberately removes that shortcut. Short conceptual searches such as “multiple playheads at once,” “Clay as a layout engine,” and “final reason for selecting egui” are mostly sent to Visual. Conversely, Visual questions such as “What six web technology logos are lined up?” are often sent to Speech or Hybrid. Character fragments capture syntax more reliably than evidence modality.

## Confusions

- Twelve of 20 Speech rows become Visual. Examples include `traditional NLE editing is archaic`, `automatic ripple after changing a trim`, and `Tauri's frontend and backend languages`.
- Ten of 20 Visual rows are missed: four become Speech and six Hybrid. Visual questions and polite commands are especially weak.
- Two of 20 strict Hybrid rows are missed. Hybrid is the only class that clears its recall gate because explicit paired evidence remains lexically easier to recognize.
- The production baseline is worse: it predicts no frozen Speech rows correctly and reaches only 45% Hybrid recall.

The complete row-level errors, predicted labels, ambiguity flags, and model-versus-annotation classification are in `reports/human-grounded-router-v1.json`.

## Annotation ambiguity

Twenty of 360 rows were marked borderline before model evaluation, including four frozen rows. The two ambiguous frozen errors are the polite Visual request for the Clay poster and the compact Speech phrase “Clay as a layout engine.” These are counted as errors under the unchanged labels. Removing them would raise accuracy only to 61.67%, far below the 90% gate, so ambiguity does not explain the failure.

No confidence cutoff can repair this result safely: incorrect routes are not separated by a validated confidence region, and the prior confidence-to-Hybrid fallback had already reduced generalization. The run therefore uses raw argmax and threshold zero.

Decision C keeps production unchanged. This frozen test must not be relabeled, used for vocabulary, or used to add lexical rules. Another router study should begin only if explicitly authorized and must add independently authored training sources plus a completely new frozen test.
