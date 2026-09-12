# Project status

Current phase: 4 complete; Phase 5 product validation next.

Implemented: video ingestion/library/player, local Whisper transcripts, pinned CLIP embeddings/FAISS search, BM25 speech relevance and documented reciprocal-rank fusion. Search supports visual, speech and hybrid modes with evidence labels and explicit available modalities. All inference is local.

Verified 2026-09-12: 17 pytest tests passed; Ruff and frontend lint/type/build passed. Prior real Whisper and CLIP smoke tests passed. Hybrid tests cover lexical relevance, fusion, duplicate suppression and speech-only fallback. Real-world retrieval quality is not yet measured.

Limitations: browser QA outstanding; no authentication/durable queue; single-process deployment only; VFR duration approximate; models are not calibrated for no-match rejection or action understanding. Tiny speech accuracy and model-inference cancellation remain limited.

Blockers: no connected user browser; proceed with a dedicated Playwright development test browser. Next: end-to-end upload/seek/search validation, responsive screenshots, UI failure-state fixes, then benchmark scripts.
