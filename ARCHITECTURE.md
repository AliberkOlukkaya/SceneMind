# Current architecture

SceneMind is a local single-host application: Next.js/TypeScript/Tailwind in the browser, FastAPI in Python, local media/index files, and SQLite transcript tables managed by SQLAlchemy/Alembic. No hosted AI service is involved.

## Video ingestion

POST /videos receives a raw request body and filename query parameter. It checks extension, bounds streamed bytes and generates a UUID directory. One ingestion lock bounds concurrency. A 202 response schedules a development BackgroundTask or persists a durable job for the isolated worker. OpenCV inspects duration/FPS/resolution/codec; FFmpeg selects frames, emits actual presentation timestamps and produces 480-pixel JPEGs. Metadata and states live in atomic manifests. Inline startup marks interruptions failed; durable recovery belongs to the worker supervisor.

GET /videos lists manifests; GET /videos/{id} returns status/metadata/frames. /media supports HTTP byte ranges. /frames/{name} serves validated JPEG paths. Source media never uses user-controlled storage filenames. Supported limits: 250 MiB, 30 minutes, 4K, one ingestion at a time and a five-minute FFmpeg timeout.

## Speech

POST /videos/{id}/transcript starts a separate bounded speech job. FFmpeg extracts temporary mono 16 kHz WAV audio. A cached faster-whisper tiny model runs CPU INT8 inference with voice activity detection. SQLAlchemy stores transcript status and ordered segments; Alembic upgrades schema at application startup. Temporary audio is removed, retries replace segments transactionally, and restart recovery marks interrupted jobs failed.

GET /videos/{id}/transcript returns segments with optional literal substring search. Video manifests remain authoritative for media; relational tables hold speech and durable jobs. SQLite and PostgreSQL migration/queue operations are verified; the optional postgres extra supplies psycopg. Full deployment and container ML remain separate validation steps.

## Visual retrieval

POST /videos/{id}/index starts explicit indexing. The cached CLIP ViT-B/32 processor/model uses a pinned checkpoint. RGB frames are resized/cropped/normalized; batches produce 512-dimensional L2-normalized embeddings. An atomic NumPy file stores vectors, and a sidecar stores status/model/revision. Changing the configured checkpoint requires reindexing.

The query text encoder produces a normalized vector. FAISS IndexFlatIP performs exact cosine retrieval over the sampled frames. The small index is reconstructed per query. One visual inference lock protects model loading/index replacement and limits concurrent work. Search scores are not confidence probabilities.

## Hybrid retrieval

GET /videos/{id}/search accepts visual, speech or hybrid mode. BM25 ranks transcript segments. Reciprocal-rank fusion combines up to 50 candidates per modality, one contribution per sampled-frame neighborhood. Speech-supported results seek to the speech start and retain text/end timestamp, while the thumbnail shows nearby visual context. Evidence contains each modality's original rank/score. Hybrid explicitly reports completed modalities used.

## Product and validation

The client uploads File bodies, polls processing/index/transcript state, and provides a library, player, sampled moments, search modes and transcript navigation. Result selection updates HTMLVideoElement.currentTime. Runtime data stays under ignored data/. Models are optional dependencies downloaded on explicit first use.

Pytest uses generated media and mocked model inference. Real model smoke scripts verify the separate inference paths. Playwright starts isolated API/frontend instances and verifies desktop/mobile upload, seeking, error recovery and optional real CLIP retrieval. The benchmark runner measures the local pipeline against interval labels, keeping synthetic results distinct from real-video quality.

## Durable processing and access

Optional durable mode sends ingestion, speech and indexing to one reusable spawned inference child supervised by app.worker. SQL jobs store stage, unique active key, attempts, creation/retry time and safe errors. Transactional capacity checks and compare-and-set claims protect concurrent enqueue/claim. OS supervisor/execution locks enforce one host writer. Process-tree deadlines and parent-death monitoring isolate failed inference. Restart recovery replays jobs within the attempt budget. Model caches survive successful jobs. Files remain atomic, transcripts transactional; queue state overlays product status routes. Migrations are serialized per database.

Optional Basic/Bearer operator authentication protects APIs, docs and media. Health and preflight stay public; Origin checks cover mutations. Browser requests include managed credentials. No password is embedded in the frontend.

Evaluation records scores, intervals, hashes, split metadata and environment. Natural V2 adds versioned query IDs/types/modalities, source/license/checksum provenance, frozen calibration and source-disjoint held-out groups, Precision/Recall/MRR at 1/3/5, negative FAR, abstention, latency, path/category slices, and per-failure evidence. Calibration fits only calibration visual negatives. An opt-in artifact gates visual evidence before fusion, bound to model revision and sampling. It is not a probability estimate or a calibrated hybrid score.

## Boundaries

This is not a public multi-tenant service. Account lifecycle, tenant ownership, object storage, cross-user quotas, distributed worker leases and high availability are outside scope. Use consistent configuration on one host with local storage. API query inference remains in-process without the worker deadline. Inline mode retains V1 single-process limitations. Worker delivery is at-least-once; a crash before enqueue can leave an unqueued upload asset. OpenCV duration is approximate for VFR. Natural V2 is small and diagnostic; it does not establish population accuracy. No OCR, action recognition, identity recognition, generated Q&A or training was added.
