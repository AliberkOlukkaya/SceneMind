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

The image-text verifier lives only under `ml/experiments/image_text_verifier/`. It reuses the API and CLIP top-five candidates, batches pair scoring through a pinned BLIP ITM head, and emits ignored reports plus a model-bound calibration artifact. The experiment failed promotion gates, so the runtime architecture remains CLIP/BM25/RRF without verifier dependencies or fallback behavior.

The follow-up under `ml/experiments/lightweight_pair_scorer/` preserves the same staged boundary. It benchmarks pinned UForm3-small ONNX cosine reranking and a deterministic logistic scorer over seven CLIP score/rank statistics. Two parent-hash-locked calibration expansions remain disjoint from Natural V2 held-out. Both fail the positive-abstention gate, so their dependencies and loading paths remain outside production.

The rejected branch under `ml/experiments/object_detector_branch/` reuses frozen CLIP top-five timestamps, maps explicit aliases to COCO class groups, and emits versioned class/confidence/source-pixel box/area/center evidence from pinned YOLOX-Nano ONNX. It runs the five candidate frames sequentially on CPU and fits a presence threshold on calibration rows only. Query routing preserves CLIP results for non-triggered rows. The branch fails small-object abstention, so no detector dependency, configuration, route or UI behavior entered the application. Its calibration artifact is explicitly non-promotable.

The follow-up under `ml/experiments/small_object_ablation/` remains outside production. It freezes source-disjoint visible-frame labels before inference, reproduces the production FFmpeg selector at 5/2/1-second intervals, and measures CLIP frame/index growth separately from detector recall. YOLOX-Nano 416/640/768 and RT-DETR-R18 640 run only on the same human-verified visible JPEGs. Best target-class boxes receive a second manual target-match review so a same-class distractor cannot count as recall. Both higher-resolution Nano configurations passed bounded experiment gates and repeated identically, but no runtime route or sampler setting changed. The next proposed branch is a cached, query-gated 2-second secondary sample path with Nano 640 over bounded windows.

The rejected coarse-to-fine prototype under `ml/experiments/bounded_secondary_sampling/` starts from the frozen five-second CLIP index, expands rule-routed object queries into calibration-selected local windows, extracts two-second JPEGs through a versioned byte-bounded cache, rescoring them with the existing CLIP session before optional Nano-640 evidence. Cache keys hash policy version, video ID and millisecond timestamp; atomic writes and least-recently-used cleanup prevent partial or unbounded data. Unsupported, speech, scene and action routes retain coarse candidates. The branch is experimental only: verified held-out evidence coverage is 50%, detector fusion receives zero calibration weight, and no-match gating causes 100% small-object abstention. No endpoint, worker, configuration, manifest or UI contract changed.

The rejected candidate-generation study under `ml/experiments/coarse_candidate_diversity/` retrieves up to 50 exact-FAISS CLIP candidates and deterministically evaluates temporal NMS, embedding MMR, combined diversity, embedding-change segments and a five-second-base plus high-change two-second diagnostic. Candidate intervals group neighboring evidence without changing relevance labels. A 50 ms timestamp tolerance handles measured VFR sampling jitter. Calibration selects all policy parameters before held-out evaluation. The high-recall raw pool is useful evidence for a future bounded list scorer, but no final-five selector meets quality and dense-index gates, so production remains the original five-second top-K search.

## Boundaries

This is not a public multi-tenant service. Account lifecycle, tenant ownership, object storage, cross-user quotas, distributed worker leases and high availability are outside scope. Use consistent configuration on one host with local storage. API query inference remains in-process without the worker deadline. Inline mode retains V1 single-process limitations. Worker delivery is at-least-once; a crash before enqueue can leave an unqueued upload asset. OpenCV duration is approximate for VFR. Natural V2 is small and diagnostic; it does not establish population accuracy. No OCR, action recognition, identity recognition, generated Q&A or training was added.
