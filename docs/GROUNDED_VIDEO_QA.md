# Transcript-grounded Video Q&A

Grounded Video Q&A V1 is an evaluation-gated Ask Video path. It answers only from a ready Whisper transcript and keeps Find Moments unchanged. It does not inspect frames, slides, OCR, objects, actions, or outside sources.

## Data flow

1. Ordered transcript segments become deterministic evidence chunks bounded by 45 seconds and 900 characters, with one-segment overlap.
2. The existing BM25 formula ranks these Q&A-specific chunks. Production Speech Search segments and ranking are unchanged.
3. At most five chunks and the question are sent to the configured `AnswerGenerator`.
4. The OpenAI provider requests strict structured output: `answerable`, `answer`, and supplied evidence IDs.
5. The backend rejects unknown IDs and uncited substantive answers, then resolves valid IDs to trusted transcript timestamps and text.
6. The UI displays the answer and clickable sources that seek the local video player.

When BM25 finds no lexical evidence, SceneMind abstains without calling the provider. A model may also abstain. The canonical response is: “I couldn't find enough evidence in this video to answer that reliably.”

## Configuration and privacy

Ask Video is disabled by default because the frozen real-provider evaluation failed abstention safety. Enabling it for development requires both `SCENEMIND_QA_ENABLED=true` and `OPENAI_API_KEY` in the ignored local `.env.local` file, followed by a backend restart. Never put a key in source code, `.env.example`, logs, or Git.

The OpenAI request contains only the user's question, up to five selected transcript chunks, grounding instructions, and the response schema. It does not contain the video, full transcript, filenames, local paths, source URL, thumbnails, model caches, or user credentials. Requests set `store: false`. Provider-side policies and billing still apply.

The model is configurable with `SCENEMIND_QA_MODEL`; V1 specifies `gpt-5.4-mini`. Timeout, one bounded retry, Top-K, chunk duration, and chunk character size are configurable. Provider failures return safe 502/504 responses without exposing provider bodies or credentials.

## Product boundaries

Uploads, Direct imports, and YouTube imports use the same Q&A path after local transcription. A ready video with no ready spoken transcript cannot be queried. Single-question interactions are supported; there is no chat memory, RAG index, semantic transcript embedding, OCR, VLM, or multimodal answer generation.

Decision D keeps the feature evaluation-gated because the frozen run failed abstention safety. See the [protocol](../ml/evaluation/GROUNDED_VIDEO_QA_V1_PROTOCOL.md), [results](../ml/evaluation/GROUNDED_VIDEO_QA_V1_RESULTS.md), and [failure report](../ml/evaluation/GROUNDED_VIDEO_QA_V1_FAILURES.md).
