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

Phases 0-6 and the bounded local Phase 10 follow-up are implemented, including Natural Video Benchmark V2. Phase 7 experiments evaluated pair verifiers, retrieval-statistics scoring, detector resolution and bounded secondary sampling. The coarse-to-fine branch fails its small-object abstention gate: local windows reach only half of the verified held-out evidence and secondary CLIP retrieves none of it at K=5. No verifier, detector or secondary sampler entered production. The next bounded Phase 7 option is improved coarse candidate generation with explicit temporal coverage; phases 8-9 stay deferred.
