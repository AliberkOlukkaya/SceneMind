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

Hybrid Fusion Refinement V1 is complete on the frozen 32-query diagnostic development set. The offline baseline reproduces production `app.hybrid.fuse` exactly. Three predeclared shared-contribution caps and three independent normalized-rank formulas were compared without raw scores, learned weights, modality rules, or Final English Acceptance V2 evidence.

The 1.50× cap is the provisional development candidate. It raises positive Top-5 from 20/28 to 21/28 and MRR@5 from 0.4827 to 0.5155, raises Speech Top-5 from 6/11 to 7/11, preserves Visual at 7/8 and Multimodal at 7/9, and produces one rescue with no new failure. Both sources avoid Top-K regression, though only one supplies the aggregate Top-5 gain. Production remains Smart Search (Hybrid), Spoken Content (Speech), and Visual Content (Visual); AUTO remains compatibility-only.

The next bounded milestone is a new source-disjoint holdout validation with algorithms, parameters, metrics, and gates frozen before retrieval. Do not tune further on the two development videos, use Final English Acceptance V2, or change production before holdout evidence. Phases 8-9 and public deployment remain deferred.
