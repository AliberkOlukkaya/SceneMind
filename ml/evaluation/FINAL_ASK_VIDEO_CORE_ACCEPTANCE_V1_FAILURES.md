# Final Ask Video Core Acceptance V1 Failures

Six of 60 frozen questions failed. There were no scope, retrieval, transcript, unsupported-claim, citation, hard-negative, provider, or end-to-end failures.

| Question | Source | Taxonomy | Observed failure |
|---|---|---|---|
| `creative-commons-licenses-explained-s02` | Creative Commons | `FALSE_ABSTENTION` | Retrieved evidence directly said Kerry had already granted permission, but the resolver returned the safe abstention. |
| `creative-commons-licenses-explained-s05` | Creative Commons | `FALSE_ABSTENTION` | Retrieved evidence directly defined no derivatives and the need to ask before retouching, but the response abstained. |
| `webm-codec-explainer-s05` | WebM | `FALSE_ABSTENTION` | Retrieved evidence explicitly said a server must send and manage a thousand streams, but the response abstained. |
| `human-software-extensions-talk-s01` | Humans as Software Extensions | `GENERATION_ERROR` | The answer said only that people extend computational systems; it omitted bodies, senses, cognition, and the plug-in/discard property in the frozen definition. |
| `demo-video-production-tutorial-s03` | Demo Video Tutorial | `FALSE_ABSTENTION` | Retrieved evidence directly explained that schedule room is needed to resolve issues before posting, but the response abstained. |
| `demo-video-production-tutorial-s04` | Demo Video Tutorial | `GENERATION_ERROR` | The answer mentioned audience characteristics but omitted identifying the audience and intended outcome. |

## Diagnosis

BM25 evidence recall was 36/36, so these failures are not retrieval misses. All 32 returned answers were grounded, every citation was valid, no material unsupported claim appeared, all 12 hard negatives abstained, and all 12 out-of-scope questions were rejected without generation. The residual problem is conservative answerability judgment plus incomplete realization of requested facts even when the correct Top-5 evidence is present.

This pattern fails usefulness without failing hallucination safety. Another small prompt, neighbor-context, BM25, or rule change would repeat the incremental path already exhausted by the three Q&A milestones. Any future Q&A milestone must begin with an architecture decision and a new source-disjoint evaluation, with semantic transcript retrieval and hierarchical evidence as the first candidate family. No such architecture is implemented here.
