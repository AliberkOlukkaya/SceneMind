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

Hybrid fusion diagnostics are complete on 32 frozen development queries over two existing source-disjoint English videos. The evaluation-only trace exactly reproduces production `app.hybrid.fuse` output and records both candidate lists, scores, ranks, RRF terms, exact-thumbnail overlap, deduplication, displacement, and Top-K membership. Final English Acceptance V2 remained isolated.

Unchanged Hybrid Top-5 succeeds on 20/28 positives. Seven failures are ranking/exact-overlap failures with relevant input evidence and one is candidate recall. Shared Speech+Visual thumbnail buckets occupy 84.4% of final slots; the fixed formula guarantees any shared top-50 bucket outranks any single-modality bucket. Production remains Smart Search (Hybrid), Spoken Content (Speech), and Visual Content (Visual); AUTO remains compatibility-only.

The next bounded milestone may compare a per-modality preservation rule, capped overlap bonus, or normalized contribution on development evidence. Define gates before selection and evaluate any selected candidate once on a new frozen source-disjoint source. Do not use Final English Acceptance V2 for tuning and do not change production until that evidence passes. Phases 8-9 and public deployment remain deferred.
