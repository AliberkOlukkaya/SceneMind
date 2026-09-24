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

Hierarchical Video Memory V1 is complete with Decision C. Section navigation generalized to 90.91% R@3 and 100% on the new 40-minute source, but bounded local evidence fell to 84.85% completeness and final Core User Success to 57.58%. A 22-minute source reached only 27.27%, so the branch is rejected. Ask Video remains disabled, `v1.0.0-rc1` is unchanged, and production BM25 remains intact.

The required next step is a product-level choice between a genuinely new multi-section evidence/completeness architecture on new sources and postponing Ask Video. No V2, prompt tweak, K/threshold adjustment, or retrieval follow-up starts automatically. All Q&A validation and acceptance sets remain frozen observed evidence and are unavailable for tuning.

SceneMind remains based on the historical **v1.0.0-rc1 Portfolio Release Candidate**. URL Ingestion V1 passed. Search remains Smart Search to uncapped RRF60 Hybrid, Spoken Content to Speech, and Visual Content to Visual.

## Future work

Potential post-v1.0 work includes improved multimodal ranking, OCR, temporal/action understanding,
Video RAG and stronger deployment. Each requires a concrete user scenario, licensed data, a readable
baseline, source-disjoint validation and resource gates before implementation. Public multi-tenant
deployment, phases 7–9 and Final Acceptance V3 are not active release tasks.
