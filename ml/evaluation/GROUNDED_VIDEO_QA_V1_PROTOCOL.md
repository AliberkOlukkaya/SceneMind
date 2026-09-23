# Grounded Video Q&A V1 protocol

This protocol was fixed before any final Q&A validation output existed. V1 is transcript-only. It does not evaluate images, OCR, actions, objects, or slide text.

## Frozen configuration

- Evidence: ordered Whisper segments grouped into at most 45 seconds or 900 characters.
- Overlap: one underlying segment between adjacent chunks.
- Retrieval: unchanged SceneMind BM25 formula over Q&A chunks, Top-5.
- Generator: OpenAI Responses API, configurable `gpt-5.4-mini`, strict JSON schema, temperature not overridden.
- Context: question plus at most five retrieved chunks; no full transcript, video, path, or provenance metadata.
- Grounding: the model may return only evidence IDs. The backend resolves IDs to timestamps and rejects unknown IDs.
- Abstention: no positive BM25 evidence abstains before generation; the model may also return `answerable=false`.
- Promotion gates: Recall@5 >=90%, answer correctness >=90%, grounded answer >=95%, citation precision >=95%, unsupported claims <=5%, correct abstention >=90%, false-answer rate <=10%.

## Data discipline

The manifest uses two real, licensed, English presentation videos previously subjected to independent full-video human review. Development and validation sources have different hashes, speakers, venues, and topics. Questions were adapted from those pre-existing human review artifacts before Q&A execution and not from Q&A retrieval results. The validation source, labels, intervals, and key facts are checksum-bound in `grounded_video_qa_v1_manifest.sha256`.

Development may inform the single fixed configuration above. Validation is one source-disjoint run. Once a valid provider run begins, no prompt, chunking, Top-K, model, abstention rule, questions, labels, facts, or intervals may change.

## Two-stage scoring

Stage A reports whether at least one retrieved chunk overlaps a verified interval at ranks 1, 3, and 5. Unanswerable items are excluded from evidence recall. Stage B requires human verification of correctness, support, citation precision/completeness, unsupported claims, and abstention. Fluent text alone never earns credit.

The run records retrieval, generation and total latency, input/output tokens, provider/model, and reported cost. If a real provider cannot be configured, all quality/performance metrics remain `null`; mocked tests are never reported as real accuracy.
