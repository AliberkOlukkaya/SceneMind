# Grounded transcript Q&A and abstention

SceneMind's Ask Video path answers questions from a video's local Whisper transcript. It is a separate evaluation-gated layer and does not alter Visual, Speech, or Hybrid Find Moments search.

## Inputs and outputs

Input is one user question plus at most five transcript chunks selected by unchanged BM25. Chunks contain text and opaque evidence IDs; video bytes, frames, paths, URLs, and timestamps are not sent to the generator. Output is strict JSON containing an answerability decision, answer text, separately cited material claims, compatibility evidence IDs, and a list of missing or unsupported requirements. The backend maps valid claim IDs to its own timestamped evidence objects.

## Preprocessing and parameters

Ordered Whisper segments are grouped deterministically into chunks of at most 45 seconds and 900 characters with one-segment overlap. BM25 returns Top-5. A small rule layer detects explicit counts, temporal words, and selected relation forms. The frozen safety configuration uses `gpt-5.4-mini-2026-03-17`, Responses API defaults, one bounded retry, a 45-second request timeout, strict JSON schema, and `store=false`.

For explicit list counts, each requested item must be a separate claim and claim count must match. Every material claim needs at least one supplied evidence ID. Unknown IDs, missing claim support, reported evidence gaps, or an invalid count cause safe abstention. This validates the response contract; it cannot prove that a model selected every semantically required fact.

## Model, source, and license

The generator is an OpenAI-hosted GPT-5.4 Mini snapshot accessed through the Responses API. Model access, terms, pricing, and retention behavior are governed by the configured OpenAI project and current provider policies. The key exists only in ignored local environment configuration. SceneMind does not distribute model weights.

## CPU, GPU, and privacy behavior

Chunking, BM25, constraint extraction, and citation validation run locally on CPU and need no GPU. Generation is remote, so local GPU availability does not change answer latency. Only the question, selected transcript text, prompt, and JSON schema leave the machine. `store=false` is set, but this is still a paid network dependency and Ask Video remains disabled by default.

## Alternatives and limits

A local instruction model would remove the paid network dependency but requires separate quality, memory, latency, and license validation. A second unrestricted verification generation pass could increase cost and latency and was not justified. Deterministic semantic validation is cheap and auditable but does not establish list completeness or temporal correctness by itself. The frozen safety validation therefore rejected production enablement despite perfect abstention and grounding metrics: overall answer correctness was 77.78%, list/count 0/2, and temporal 1/2.
