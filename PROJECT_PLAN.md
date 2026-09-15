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

Long-video ingestion and resource hardening is complete. Final English acceptance failed its routing and Top-k gates. The bounded router follow-up built 450 balanced queries over 15 source scenarios and measured the current router at 50.00% frozen accuracy. A character n-gram linear candidate reaches 95.56% with every numeric gate passing, but production remains unchanged because the scenarios are not grounded in independently reviewed real videos.

Decision E identifies one next step: gather human route labels tied to new real English videos, keep sources disjoint across train/validation/test, and rerun the existing classical comparison. The failed acceptance media stays protected and cannot contribute queries, vocabulary, rules, labels, thresholds, or templates. If real-video evidence later supports promotion, Final English Acceptance V2 needs another untouched long video and new frozen labels. Phases 8-9 remain deferred.

Phases 0-6 and the bounded local Phase 10 follow-up are implemented. Earlier retrieval experiments and the first personal acceptance run remain frozen evidence. The Turkish compatibility result above supersedes the former next-milestone statement; no candidate was promoted.
