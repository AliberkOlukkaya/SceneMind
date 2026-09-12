# Project status

Current phase: 1 complete; Phase 2 next.

Implemented: local bounded raw-body video upload; OpenCV metadata; portable FFmpeg extraction with actual presentation timestamps; JPEG thumbnails; atomic per-video manifests; queued/processing/ready/failed states; restart interruption detection; library, player and thumbnail seek controls.

Verified 2026-09-12: 10 pytest tests passed, Ruff passed, frontend ESLint/TypeScript/production build passed. Live upload returned 202; frontend returned 200. Tests cover real synthetic video decoding, byte-range playback, corrupt/empty/oversized uploads, concurrency, timeout and interrupted jobs. Upstream test dependencies emit two deprecation warnings.

Limitations: browser automation unavailable, so visual and interactive browser QA remains outstanding. Single backend process only, no authentication or durable queue. OpenCV duration is approximate for VFR files. Some accepted codecs cannot play in every browser. No speech or visual AI yet. Local manifests will move to relational persistence with speech.

Blockers: none for backend development. Next: add SQLAlchemy/Alembic persistence and optional local speech model, timestamp transcripts and text search; keep real model tests separate from unit tests.
