# Project status

Current phase: 3 complete; Phase 4 next.

Implemented: foundation, bounded video ingestion and timestamp browsing, local speech and literal search, pinned CLIP ViT-B/32 frame/text embeddings, exact FAISS cosine retrieval, explicit indexing and search UI. Models run locally and are optional installs.

Verified 2026-09-12: 15 pytest tests passed; Ruff, frontend ESLint/TypeScript/build passed. Real Whisper speech smoke passed. Real CLIP synthetic smoke retrieved red at 0 s and blue at 5 s; first indexing including download took 64.311 s. These are smoke tests, not real-video quality benchmarks. Evaluation protocol exists in ml/evaluation/PROTOCOL.md.

Known limitations: browser visual/interaction QA outstanding; local single-process service, no auth/durable queue, approximate VFR metadata; CLIP static-frame limitations; no calibrated no-match threshold; no held-out real-video quality results. Optional models reside under ignored data/models. Two upstream test deprecation warnings remain.

Blockers: none for Phase 4. Next: token-based transcript relevance and reciprocal-rank fusion, modality-aware evidence tests; then browser end-to-end product validation and evaluation scripts.
