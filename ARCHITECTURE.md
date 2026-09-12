# Current architecture

Next.js App Router with TypeScript and Tailwind provides the browser interface. FastAPI exposes GET /health. Pydantic Settings reads SCENEMIND_ environment variables. CORS allows localhost:3000 by default.

No database, model or retrieval pipeline exists yet. Local runtime data belongs in ignored data/. Backend code is in backend/app, tests in tests/. Later ML code will be added when a working feature needs it.

## Phase 1 architecture
POST /videos accepts raw bytes and a filename query parameter. A single ingestion slot bounds local processing. UUID directories contain source media, atomic JSON manifests and thumbnails. FastAPI BackgroundTasks executes FFmpeg after the 202 response. GET /videos and /videos/{id} expose status; /media supports range requests; /frames/{name} serves validated JPEG paths. Startup marks interrupted jobs failed. Use one worker only.

The client polls the library, uploads File bodies, and seeks the HTML video element using FFmpeg timestamps. NEXT_PUBLIC_API_URL configures its backend origin. No model has been added yet.
