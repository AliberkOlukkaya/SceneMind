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

Long-video ingestion and resource hardening is complete. Final English acceptance failed its routing and Top-k gates. The human-grounded router follow-up reviewed ten new real videos and froze 360 balanced, source-disjoint queries. On the final two unseen videos, production reaches 48.33% routing accuracy and the fixed character n-gram candidate reaches 60.00%; both fail the required quality gates.

Decision C leaves production unchanged and stops automatic router experimentation. The failed acceptance media remained protected and contributed no queries, vocabulary, rules, labels, thresholds, or templates. A future router milestone would require explicit authorization, more varied independent training evidence, and another untouched frozen test. Final English Acceptance V2 remains conditional on a later promotion. Phases 8-9 remain deferred.

Phases 0-6 and the bounded local Phase 10 follow-up are implemented. Earlier retrieval experiments and the first personal acceptance run remain frozen evidence. The Turkish compatibility result above supersedes the former next-milestone statement; no candidate was promoted.
