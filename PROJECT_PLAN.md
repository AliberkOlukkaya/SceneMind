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

Hybrid Fusion Holdout Validation V1 is complete. Two new source-disjoint sources and 34 queries were reviewed and frozen before retrieval. Production parity passed on all queries, and only baseline RRF60 and the predeclared 1.50× cap were compared.

The candidate is rejected. It changes ALL Top-1/3/5 from 12/19/21 to 15/19/20 and MRR@5 from 0.5560 to 0.6083, but Speech Top-5 falls from 8/11 to 7/11, Jimmy Wales Top-5 falls from 6/12 to 5/12, one query breaks, and none are rescued. Strong-candidate retention remains 18/32. Production remains Smart Search (uncapped Hybrid), Spoken Content (Speech), and Visual Content (Visual); AUTO remains compatibility-only.

Do not tune, relabel, or rerun the failed holdout, promote Cap 1.50×, or start Final Acceptance V3. The next product milestone requires an explicit decision based on the accumulated acceptance and holdout evidence. Phases 8-9 and public deployment remain deferred.
