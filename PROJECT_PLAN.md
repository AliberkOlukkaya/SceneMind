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

Final English Acceptance V2 completed on a new independently reviewed 30:29 English presentation. The unchanged Smart Search / Hybrid path reaches useful Top-1/3/5 of 76.92%/76.92%/84.62%, missing the frozen 85% Top-3 and 90% Top-5 gates. All eight multimodal queries pass at rank 1, while Speech Top-5 is 70.00%. Explicit Speech diagnostics recover both primary Speech FAILs at rank 1; the sole PARTIAL also moves to rank 1 but remains incomplete. This identifies Hybrid fusion/ranking as the dominant subsystem. One Visual failure persists in explicit Visual mode.

Decision 031 records **C - FINAL ACCEPTANCE V2 FAILED**. The English-first v1.0 search core is not frozen. Production remains Smart Search (Hybrid), Spoken Content (Speech), and Visual Content (Visual); AUTO remains compatibility-only. The acceptance source is permanently held out and cannot be used for tuning or rerun as fresh acceptance.

The next milestone is a bounded product-level Hybrid fusion investigation using new source-disjoint development evidence. Start with how strong BM25 Speech candidates are displaced by visually generic candidates. Do not create a new model milestone automatically. Phases 8-9 and public deployment remain deferred.
