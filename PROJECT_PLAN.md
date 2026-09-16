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

Calibrated Raw-Score + Agreement Fusion V1 is complete with decision C. Deterministic per-retriever logistic calibration was selected on 72 queries from three sources and evaluated on 32 queries from two new sources. Both Visual and Speech calibration underperformed a rank-only reference in validation AUC and Brier, so the experiment stopped before constructing a fusion candidate. The protected 34-query holdout remained unopened for candidate evaluation and production remains uncapped RRF60.

Cap 1.50× remains rejected. It changes Holdout ALL Top-1/3/5 from 12/19/21 to 15/19/20 and MRR@5 from 0.5560 to 0.6083, but Speech Top-5 falls from 8/11 to 7/11, one query breaks, none are rescued, and an unsupported result becomes more convincing. Production remains Smart Search (uncapped RRF60 Hybrid), Spoken Content, and Visual Content; AUTO remains compatibility-only.

The next ranking milestone is data acquisition: collect at least three additional source-disjoint calibration sources with complete Top-50 Visual/Speech traces and freeze a new validation split. Only then may confidence-aware fusion be reconsidered. Do not tune on Holdout V1 or Acceptance V2, begin a second-stage reranker, continue alpha search, change production retrieval, or start Final Acceptance V3. Phases 8-9 and public deployment remain deferred.
