# Decisions

## 001 — Local development first
Use Python 3.13, FastAPI and Next.js, with no external services in the foundation. Python 3.14 is installed but ML wheel support favors 3.13. Alternatives: container-only setup or mandatory PostgreSQL. Consequence: fewer initial dependencies; persistence and deployment arrive with concrete requirements.

## 002 — UI direction
Visual thesis: a quiet charcoal editing workspace with warm white typography and one lime accent. Content: library first, empty-state guidance, then upload/workspace as ingestion arrives. Interaction: focus feedback and short hover transitions, respecting reduced motion. No fabricated videos or search results.

## 003 — Portable media tools and bounded local jobs
Use imageio-ffmpeg for a packaged FFmpeg executable and OpenCV for basic metadata. System FFmpeg/ffprobe was absent. Alternatives: require installation or use PyAV throughout. Consequences: easy local setup, approximate VFR duration; FFmpeg licenses matter if binary redistribution is introduced.

## 004 — Video metadata manifests before relational speech storage
Phase 1 uses atomic per-video JSON manifests, keeping source assets and processing output together. A single-process lock bounds jobs and startup detects interruption. Alternatives: premature queue/Redis or SQLite schema immediately. Consequences: no multi-worker deployment; Phase 2 introduces SQLAlchemy/Alembic when transcript querying requires relational data. These manifests are local runtime artifacts, never committed.

## 005 — Optional tiny speech model and scoped relational persistence
Use faster-whisper tiny with CPU INT8; install via the speech extra and download on explicit transcription. Alternative: larger Whisper or mandatory transcription on every upload. Consequence: ingestion stays lightweight and model accuracy is limited. Actual smoke inference is required beyond mocked unit tests.

SQLAlchemy/Alembic now own transcript jobs and segments. Keep Phase 1 video manifests as asset metadata instead of migrating them without a query requirement. This refines decision 004: no duplicate authoritative video database. SQLite is the verified local path; PostgreSQL deployment remains future work.
