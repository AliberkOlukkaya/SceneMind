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

The bounded Turkish compatibility milestone is complete with outcome E. Six source-disjoint development groups and 72 natural Turkish queries show that cheap routing improves held-out accuracy from 53.3% to 76.7%, but misses the 90% gate; cheap AUTO R@5 is 80%, below its 85% gate. Direct Turkish CLIP is already useful at 93.3% forced-Visual R@5, while lexical adaptation lowers it. A multilingual semantic Speech diagnostic reaches 100% R@5 but adds about 254 MiB RSS and does not fix routing. Nothing is promoted and the personal suite is not rerun. Before another ML milestone, add independent Turkish sources and freeze a new router-validation split. Unsupported-query disclosure and a deliberate 30–60 minute ingestion policy remain separate product blockers; phases 8-9 remain deferred.

Phases 0-6 and the bounded local Phase 10 follow-up are implemented. Earlier retrieval experiments and the first personal acceptance run remain frozen evidence. The Turkish compatibility result above supersedes the former next-milestone statement; no candidate was promoted.
