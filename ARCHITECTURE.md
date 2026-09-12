# Current architecture

Next.js App Router with TypeScript and Tailwind provides the browser interface. FastAPI exposes GET /health. Pydantic Settings reads SCENEMIND_ environment variables. CORS allows localhost:3000 by default.

No database, model or retrieval pipeline exists yet. Local runtime data belongs in ignored data/. Backend code is in backend/app, tests in tests/. Later ML code will be added when a working feature needs it.

## Phase 1 architecture
POST /videos accepts raw bytes and a filename query parameter. A single ingestion slot bounds local processing. UUID directories contain source media, atomic JSON manifests and thumbnails. FastAPI BackgroundTasks executes FFmpeg after the 202 response. GET /videos and /videos/{id} expose status; /media supports range requests; /frames/{name} serves validated JPEG paths. Startup marks interrupted jobs failed. Use one worker only.

The client polls the library, uploads File bodies, and seeks the HTML video element using FFmpeg timestamps. NEXT_PUBLIC_API_URL configures its backend origin. No model has been added yet.

## Phase 2 architecture
Speech is explicit and independent of ingestion. POST /videos/{id}/transcript extracts a temporary mono 16 kHz WAV and runs the cached faster-whisper model. A separate single-job lock bounds speech inference. SQLAlchemy stores transcript status and segments in SQLite; Alembic upgrades schema at startup. GET transcript supports literal substring queries. PostgreSQL URLs can be configured with an appropriate driver, but PostgreSQL has not been tested.

Video manifests remain authoritative for media assets; only transcript records are relational. There is no cross-store transaction requirement for the current read-only video lifetime. Temporary audio is removed after processing. Interrupted speech jobs become failed on restart.

## Phase 3 architecture
Visual indexing is a separate explicit job. A cached Transformers CLIP dual encoder produces normalized 512-dimensional vectors, saved atomically as NumPy files. A sidecar records processing state, model and pinned revision. Search reconstructs a small exact FAISS IndexFlatIP index and embeds the query under the same inference lock. Results map to the original FFmpeg timestamps. Frontend displays raw cosine scores and seeks on selection. See learning documents 03 and 04.

## Phase 4 architecture
GET /videos/{id}/search accepts visual, speech or hybrid mode. BM25 ranks local transcript segments. Hybrid merges at most 50 candidates per modality through reciprocal-rank fusion in nearest-frame neighborhoods. Results expose evidence ranks/raw scores and use transcript-start navigation when speech contributes. Missing completed modalities are explicitly reported; speech mode needs no visual model. The ranking formula and tradeoffs are documented in learning/05-hybrid-retrieval.md.

## Phase 5 validation
Playwright starts isolated API/frontend instances and generates a small synthetic MP4. Desktop and mobile tests upload real files, seek the player, verify missing-index feedback, and optionally exercise real CLIP retrieval. SCENEMIND_MODEL_E2E=1 enables model-dependent tests. Default browser tests require no weights. Screenshots use synthetic red/blue scenes, not a real-video quality demo.
