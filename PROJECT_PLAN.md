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

Phases 0-6 and the bounded local Phase 10 follow-up are implemented, including Natural Video Benchmark V2. Phase 7 experiments rejected pair verifiers, global no-match, detector routing, bounded secondary sampling, cheap diversity and candidate-list scoring. A source-disjoint query-routing calibration promoted a tiny text-only AUTO selector while retaining every manual mode. The first real personal acceptance run is complete and selected outcome C: 83.3% useful Top-5, 77.8% AUTO routing, 77.8% Turkish Top-5, and misleading output on 14/18 negatives. Search stays interactive, but the supplied media do not validate 30–60 minute use and one original file exceeds the 250 MiB upload cap. The next bounded phase targets Turkish compatibility with the existing models; phases 8-9 remain deferred.
