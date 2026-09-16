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

SceneMind is at **v1.0.0-rc1 Portfolio Release Candidate**. Phases 0–6 and the bounded local hardening work are complete. The release presents one coherent local workflow, copy-pasteable setup, conservative result UX, durable processing, reproducible evaluation and documented limitations. Search remains Smart Search → uncapped RRF60 Hybrid, Spoken Content → Speech, and Visual Content → Visual. AUTO is compatibility-only.

Search-core ML research is closed for v1.0. Cap 1.50×, calibrated fusion, verifier, reranker, detector, sampling and router alternatives remain documented rejections. Acceptance and holdout evidence stays frozen and is not a future tuning set.

## Future work

Potential post-v1.0 work includes improved multimodal ranking, OCR, temporal/action understanding,
Video RAG and stronger deployment. Each requires a concrete user scenario, licensed data, a readable
baseline, source-disjoint validation and resource gates before implementation. Public multi-tenant
deployment, phases 7–9 and Final Acceptance V3 are not active release tasks.
