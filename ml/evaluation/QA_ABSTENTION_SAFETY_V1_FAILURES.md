# Q&A Abstention Safety V1 failures

The safety objective succeeded: none of the 12 unanswerable questions, including ten hard negatives, received a substantive answer; all 38 material claims in 16 substantive answers were supported, and every displayed citation supported at least one claim. Remaining failures are answer completeness and false abstention.

| ID | Category | Failure | Diagnosis |
|---|---|---|---|
| `val-rewire-03` | list/count | Incorrect substantive answer | The model supplied three supported statements but split “familiar and intuitive” into two items and omitted the required node-graph/no-panel guideline. Structural claim count alone cannot prove list completeness. |
| `val-rewire-04` | temporal | False abstention | Top-5 included the reviewed transition from connection-based systems to the timeline-as-a-node proposal, but the answerability decision rejected it. |
| `val-ui-03` | list/count | False abstention | BM25 Top-5 missed the interval that states Rust-backend-to-UI and UI-to-Rust synchronization. Retrieval is frozen in this milestone, so the miss is recorded without repair. |
| `val-ui-09` | explanation | Incomplete answer | Every returned claim was supported, but the answer omitted the frozen key instruction to identify non-negotiable requirements and accept tradeoffs elsewhere. |

The result separates safety from usefulness. Claim-level citation validation prevents unsupported content, while a requested claim count does not establish that the correct items were selected. Temporal answerability remains overly conservative on one source. One list/count failure is a retrieval miss, so a generation-only change cannot solve the whole structured-question category.

Do not tune these four questions, rerun the frozen validation, relax the gates, or enable Ask Video. Any next experiment needs new development sources and must pre-register how temporal anchors and list completeness are represented. The previous Q&A validation and this validation remain historical evidence only.
