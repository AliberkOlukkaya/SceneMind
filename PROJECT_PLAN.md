# Roadmap and acceptance criteria

0. Foundation: backend/frontend start, health responds, tests pass, setup and memory documents exist.
1. Video core: bounded validated upload, safe storage, metadata, sampled frames, thumbnails, status and errors; integration tests.
2. Speech: local transcription, persistent timestamp segments, retrieval and text search; mockable models.
3. Visual search: configurable open encoder, normalized frame/text embeddings, local index, timestamped scored results; evaluation protocol.
4. Hybrid retrieval: documented fusion of speech and visual evidence, tests for both modalities.
5. Product: responsive library, upload, workspace, search, transcript, click-to-seek and honest states.
6. Evaluation: reproducible relevance/latency protocol and scripts; only measured numbers.
7. Advanced understanding: add scene detection/OCR/object/action features only for demonstrated needs.
8. Grounded Q&A: local evidence-backed answers with timestamp citations.
9. Specialization: task, dataset, baseline and experiment plan before any training; no automatic expensive runs.
10. Hardening: durable jobs, deployment, PostgreSQL, security, recovery and observability after core works.

V1 requires phases 0–6. Later phases are optional extensions, not prerequisites. Every phase ends with tests, updated status and a stable commit.

## Current milestone

Final Ask Video Core Acceptance V1 is complete with Decision B. Four new licensed spoken sources and 60 frozen questions produced 100% scope rejection/recall, Evidence Recall@5, grounding, citation precision, unsupported-claim safety and hard-negative abstention. Answer Correctness and Core User Success were 83.33%, and Citation Recall was 88.89%; all missed fixed gates. Ask Video remains disabled, `v1.0.0-rc1` is unchanged, and incremental prompt/BM25/neighbor/rule tweaks are closed. Any future Q&A work must first choose and pre-register a different architecture, beginning with semantic transcript retrieval and hierarchical evidence.

SceneMind remains based on the historical **v1.0.0-rc1 Portfolio Release Candidate**. URL Ingestion V1 passed. Grounded Video Q&A V1 is implemented but remains disabled after frozen abstention-safety gates failed. Search remains Smart Search to uncapped RRF60 Hybrid, Spoken Content to Speech, and Visual Content to Visual.

URL ingestion is complete without reopening search-core ML research. Final Ask Video acceptance also closes incremental Q&A work. Frozen Q&A validation, acceptance and ranking holdouts are not future tuning sets.

## Future work

Potential post-v1.0 work includes improved multimodal ranking, OCR, temporal/action understanding,
Video RAG and stronger deployment. Each requires a concrete user scenario, licensed data, a readable
baseline, source-disjoint validation and resource gates before implementation. Public multi-tenant
deployment, phases 7–9 and Final Acceptance V3 are not active release tasks.
