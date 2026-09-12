# Project status

Current phase: 2 complete; Phase 3 next.

Implemented: Phase 0 foundation; Phase 1 video upload, metadata, timestamped frames, local library/player; Phase 2 optional faster-whisper tiny CPU transcription, SQLAlchemy/Alembic SQLite transcript storage, literal text search, transcript panel and timestamp navigation. First model load is cached; no paid API.

Verified 2026-09-12: 13 pytest tests passed; Ruff passed; frontend lint, TypeScript and production build passed. Real local speech smoke test on Windows-generated speech returned the phrase 'The learning rate controls how quickly the model learns.' at 0.0–3.7 seconds. Initial inference/setup took 13.494 seconds; this is a smoke check, not a benchmark. Model data is ignored under data/models.

Known limitations: browser QA remains unverified because no browser automation surface is available. Single-process, local-only server without authentication or durable jobs. OpenCV duration is approximate on VFR media; browser codec support varies. Speech tiny has accuracy limits; transcription has no hard model-inference cancellation. Two upstream test deprecation warnings remain. Video metadata stays in atomic manifests; relational tables store speech only.

Blockers: none. Next: Phase 3 dual encoder, normalized frame/text embeddings, FAISS retrieval and an independent relevance protocol; verify actual inference before claiming semantic search.
