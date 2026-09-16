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

Hybrid Ranking Failure Analysis V1 is complete. It reconstructs 19 available ranking failures and compares them with 12 balanced success controls without changing production ranking. Historical Acceptance V2 traces match 30/30 recorded production outputs. Irrelevant consensus explains 14/19 failures, while rank-only information loss is a supported secondary mechanism in 11. Thumbnail overlap is common in successes as well as failures, and same-thumbnail Speech duplicates do not accumulate.

Cap 1.50× remains rejected. It changes Holdout ALL Top-1/3/5 from 12/19/21 to 15/19/20 and MRR@5 from 0.5560 to 0.6083, but Speech Top-5 falls from 8/11 to 7/11, one query breaks, none are rescued, and an unsupported result becomes more convincing. Production remains Smart Search (uncapped RRF60 Hybrid), Spoken Content, and Visual Content; AUTO remains compatibility-only.

The next ranking milestone may compare broad confidence-aware agreement or second-stage reranking designs only after collecting new development data and defining a future untouched holdout. It must not tune on Holdout V1 or Acceptance V2. Do not continue alpha search, promote Cap 1.50×, change production retrieval, or start Final Acceptance V3. Phases 8-9 and public deployment remain deferred.
