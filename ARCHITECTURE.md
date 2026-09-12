# Current architecture

SceneMind is a local single-process application: Next.js/TypeScript/Tailwind in the browser, FastAPI in Python, local media/index files, and SQLite transcript tables managed by SQLAlchemy/Alembic. No hosted AI service is involved.

## Video ingestion

POST /videos receives a raw request body and filename query parameter. It checks extension, bounds streamed bytes and generates a UUID directory. One ingestion lock bounds concurrency. A 202 response schedules an in-process BackgroundTask. OpenCV inspects duration/FPS/resolution/codec; FFmpeg selects frames, emits actual presentation timestamps and produces 480-pixel JPEGs. Metadata and states live in atomic manifests. Startup marks interrupted jobs failed.

GET /videos lists manifests; GET /videos/{id} returns status/metadata/frames. /media supports HTTP byte ranges. /frames/{name} serves validated JPEG paths. Source media never uses user-controlled storage filenames. Supported limits: 250 MiB, 30 minutes, 4K, one ingestion at a time and a five-minute FFmpeg timeout.

## Speech

POST /videos/{id}/transcript starts a separate bounded speech job. FFmpeg extracts temporary mono 16 kHz WAV audio. A cached faster-whisper tiny model runs CPU INT8 inference with voice activity detection. SQLAlchemy stores transcript status and ordered segments; Alembic upgrades schema at application startup. Temporary audio is removed, retries replace segments transactionally, and restart recovery marks interrupted jobs failed.

GET /videos/{id}/transcript returns segments with optional literal substring search. Video manifests remain authoritative for media; relational tables hold speech only. SQLite is verified. Configurable PostgreSQL URLs require a driver and migration/deployment validation that has not yet been performed.

## Visual retrieval

POST /videos/{id}/index starts explicit indexing. The cached CLIP ViT-B/32 processor/model uses a pinned checkpoint. RGB frames are resized/cropped/normalized; batches produce 512-dimensional L2-normalized embeddings. An atomic NumPy file stores vectors, and a sidecar stores status/model/revision. Changing the configured checkpoint requires reindexing.

The query text encoder produces a normalized vector. FAISS IndexFlatIP performs exact cosine retrieval over the sampled frames. The small index is reconstructed per query. One visual inference lock protects model loading/index replacement and limits concurrent work. Search scores are not confidence probabilities.

## Hybrid retrieval

GET /videos/{id}/search accepts visual, speech or hybrid mode. BM25 ranks transcript segments. Reciprocal-rank fusion combines up to 50 candidates per modality, one contribution per sampled-frame neighborhood. Speech-supported results seek to the speech start and retain text/end timestamp, while the thumbnail shows nearby visual context. Evidence contains each modality's original rank/score. Hybrid explicitly reports completed modalities used.

## Product and validation

The client uploads File bodies, polls processing/index/transcript state, and provides a library, player, sampled moments, search modes and transcript navigation. Result selection updates HTMLVideoElement.currentTime. Runtime data stays under ignored data/. Models are optional dependencies downloaded on explicit first use.

Pytest uses generated media and mocked model inference. Real model smoke scripts verify the separate inference paths. Playwright starts isolated API/frontend instances and verifies desktop/mobile upload, seeking, error recovery and optional real CLIP retrieval. The benchmark runner measures the local pipeline against interval labels, keeping synthetic results distinct from real-video quality.

## Boundaries

This is not a public production service. There is no authentication, durable job queue, model-inference deadline, multi-worker coordination, object storage or cross-user quota. FFmpeg has a timeout; model inference does not. Restart recovery reports failures rather than resuming work. OpenCV duration is approximate for VFR inputs. No OCR, action recognition, identity recognition, generated Q&A or fine-tuning exists.
