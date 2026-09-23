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

Semantic Transcript Retrieval + Hierarchical Evidence Selection V1 is complete with Decision B. The source-disjoint frozen candidate improved evidence completeness and false abstention while preserving safety, but reduced Answer Correctness and Core User Success from 69.70% to 63.64% and failed catastrophically on the 44-minute source. Ask Video remains disabled, `v1.0.0-rc1` is unchanged, and production BM25 remains intact.

The required next step is a product-level decision among changing embedding architecture, adding multimodal evidence, hierarchical summarization, or postponing Ask Video. No retrieval follow-up starts automatically. All Q&A validation and acceptance sets remain frozen observed evidence and are unavailable for tuning.

SceneMind remains based on the historical **v1.0.0-rc1 Portfolio Release Candidate**. URL Ingestion V1 passed. Search remains Smart Search to uncapped RRF60 Hybrid, Spoken Content to Speech, and Visual Content to Visual.

## Future work

Potential post-v1.0 work includes improved multimodal ranking, OCR, temporal/action understanding,
Video RAG and stronger deployment. Each requires a concrete user scenario, licensed data, a readable
baseline, source-disjoint validation and resource gates before implementation. Public multi-tenant
deployment, phases 7–9 and Final Acceptance V3 are not active release tasks.
