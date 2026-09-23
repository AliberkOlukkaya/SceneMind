# Transcript-grounded Video Q&A

Grounded Video Q&A V1 is an evaluation-gated Ask Video path. It answers only from a ready Whisper transcript and keeps Find Moments unchanged. It does not inspect frames, slides, OCR, objects, actions, or outside sources.

## Data flow

1. Ordered transcript segments become deterministic evidence chunks bounded by 45 seconds and 900 characters, with one-segment overlap.
2. The existing BM25 formula ranks these Q&A-specific chunks. Production Speech Search segments and ranking are unchanged.
3. A deterministic scope gate allows factual, definition, direct-explanation, localized-topic and localized-summary questions. It rejects explicit list/count, temporal-ordering, long-range compositional, visual-only and OCR-dependent structures before generation.
4. Allowed questions send only the unchanged Top-5 chunks. Failed structured neighbor and temporal expansion is not active in the candidate production path.
5. The OpenAI provider requests strict structured output: `answerable`, `answer`, claim-level evidence IDs, and any unsupported or missing requirement.
6. The backend rejects unknown IDs, missing claim citations, provider-declared evidence gaps, mismatched top-level/claim citations, and duplicate claims. It derives displayed citations from valid claims and resolves IDs to trusted transcript timestamps and text. Invalid contracts become the canonical abstention response.
7. The UI describes the bounded spoken-evidence scope and displays clickable sources that seek the local video player.

When BM25 finds no lexical evidence, SceneMind abstains without calling the provider. A model may also abstain. The canonical response is: “I couldn't find enough evidence in this video to answer that reliably.”

## Configuration and privacy

Ask Video is disabled by default because Final Ask Video Core Acceptance V1 failed answer-correctness, citation-recall and core-user-success gates. Enabling it for development requires both `SCENEMIND_QA_ENABLED=true` and `OPENAI_API_KEY` in the ignored local `.env.local` file, followed by a backend restart. Never put a key in source code, `.env.example`, logs, or Git.

The OpenAI request contains only the user's question, up to five selected transcript chunks, grounding instructions, and the response schema. It does not contain the video, full transcript, filenames, local paths, source URL, thumbnails, model caches, or user credentials. Requests set `store: false`. Provider-side policies and billing still apply.

The model is configurable with `SCENEMIND_QA_MODEL`; the safety validation pins `gpt-5.4-mini-2026-03-17`. Timeout, one bounded retry, Top-K, chunk duration, and chunk character size are configurable. Provider failures return safe 502/504 responses without exposing provider bodies or credentials.

## Product boundaries

Uploads, Direct imports, and YouTube imports use the same Q&A path after local transcription. A ready video with no ready spoken transcript cannot be queried. Single-question interactions are supported; there is no chat memory, RAG index, semantic transcript embedding, OCR, VLM, or multimodal answer generation.

Q&A Structured-Question Evidence V1 tested conditional list and temporal expansion on 36 new development questions and one frozen, source-disjoint 36-question validation. Decision D rejects the branch: Answer Correctness was 72.22%, list/count and temporal correctness were each 62.50%, unsupported claims were 20.75%, and hard-negative false answers were 30.00%. Expansion was fast but did not improve evidence recall and weakened the protected safety behavior. See the structured [protocol](../ml/evaluation/QA_STRUCTURED_QUESTION_EVIDENCE_V1_PROTOCOL.md), [results](../ml/evaluation/QA_STRUCTURED_QUESTION_EVIDENCE_V1_RESULTS.md), and [failure report](../ml/evaluation/QA_STRUCTURED_QUESTION_EVIDENCE_V1_FAILURES.md).

Final Ask Video Core Acceptance V1 used four new licensed sources and 60 frozen questions. Scope rejection, Evidence Recall@5, grounding, citation precision, unsupported-claim safety and unanswerable abstention all reached 100%, but Answer Correctness and Core User Success were 83.33% and Citation Recall was 88.89%. Decision B keeps the feature disabled and ends incremental prompt/evidence/rule tweaking. See the final [protocol](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_PROTOCOL.md), [results](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_RESULTS.md), and [failures](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_FAILURES.md).

The prior Q&A Abstention Safety V1 [protocol](../ml/evaluation/QA_ABSTENTION_SAFETY_V1_PROTOCOL.md), [results](../ml/evaluation/QA_ABSTENTION_SAFETY_V1_RESULTS.md), and [failure report](../ml/evaluation/QA_ABSTENTION_SAFETY_V1_FAILURES.md), plus the original V1 [protocol](../ml/evaluation/GROUNDED_VIDEO_QA_V1_PROTOCOL.md), [results](../ml/evaluation/GROUNDED_VIDEO_QA_V1_RESULTS.md), and [failures](../ml/evaluation/GROUNDED_VIDEO_QA_V1_FAILURES.md), remain historical evidence and may not be used for tuning.
