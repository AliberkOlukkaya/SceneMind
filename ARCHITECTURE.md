# SceneMind architecture

SceneMind is a single-operator, local-first video search application. Its production search
core is frozen at five-second frame sampling, CLIP/FAISS Visual retrieval, Whisper/BM25 Speech
retrieval and uncapped RRF60 Hybrid ranking.

Ask Video remains disabled after Final Core Acceptance Decision B. Its candidate path uses a deterministic scope gate followed by the existing 45-second/900-character BM25 Top-5, strict structured generation, claim-level evidence IDs and server-resolved citations. Explicit list/count, temporal-ordering, long-range, visual-only and OCR-dependent questions return a concise capability response before generation. The failed structured neighbor/temporal expansion remains historical evaluation code and is not called by the candidate path. Frozen acceptance reached 100% Evidence Recall@5 and safety but only 83.33% answer correctness/core user success, so the feature flag stays off.

The isolated semantic/hierarchical experiment in `semantic_qa.py` is not wired to the API. It builds non-overlapping fine transcript units and three-unit context sections, encodes both with pinned local MiniLM, persists local FAISS indexes plus timestamp metadata, and selects at most five chronological fine units. Frozen Decision B rejects integration because improved evidence coverage did not improve answers and long-video section selection failed.

The later `video_memory.py` experiment is also isolated from the API. It groups contiguous L1 units with interpretable pause, adjacent-topic and 180-second boundaries, then persists extractive L3 navigation summaries, topics and complete L1/L0 provenance. Section memory selects at most three regions; semantic/BM25 local search can emit only original transcript units. Transcript and configuration fingerprints invalidate stale memory, and indexes reload after restart. Frozen Decision C rejects integration: Section R@3 reached 90.91%, but evidence completeness and Core User Success fell to 84.85% and 57.58%. Summaries never become evidence or citations.

## End-to-end data flow

```mermaid
flowchart TD
    Browser[Next.js workspace] -->|streamed bytes| API[FastAPI]
    URL[Supported public URL] --> Provider[SSRF-safe Direct / YouTube provider]
    Provider -->|bounded local artifact| Validate
    API --> Validate[Extension, byte, disk and video validation]
    Validate --> Store[UUID-owned local source and atomic manifest]
    Store --> Queue{Durable jobs enabled?}
    Queue -->|No| Inline[Bounded background task]
    Queue -->|Yes| SQLJob[Persisted SQL job]
    SQLJob --> Worker[Single-host worker supervisor]
    Inline --> Ingest[Ingest stage]
    Worker --> Ingest
    Ingest --> Frames[FFmpeg frames every 5 seconds]
    Ingest --> Metadata[OpenCV metadata]
    Frames --> VisualJob[Visual-index stage]
    Store --> SpeechJob[Speech stage]
    VisualJob --> CLIP[CLIP image inference]
    CLIP --> Vectors[Normalized embeddings and FAISS index]
    SpeechJob --> Audio[Temporary 16 kHz mono WAV]
    Audio --> Whisper[Whisper inference]
    Whisper --> Transcript[Timestamped SQL segments]
    Query[Natural-language query] --> Mode{Product mode}
    Mode -->|Visual Content| CLIPText[CLIP text inference]
    CLIPText --> Vectors
    Mode -->|Spoken Content| BM25[BM25 over transcript]
    Transcript --> BM25
    Mode -->|Smart Search| Both[Run Visual and Speech]
    Both --> RRF[Group by thumbnail and apply RRF60]
    Vectors --> RRF
    BM25 --> RRF
    Vectors --> Results[Ranked moments]
    BM25 --> Results
    RRF --> Results
    Results -->|timestamp, thumbnail, evidence| Browser
```

## Upload and storage

`POST /videos` accepts MP4, MOV, WebM, MKV and AVI names, streams the request directly to disk,
and enforces the configured byte limit without buffering the complete file in memory. A declared
oversize request is rejected before allocation; the streamed byte counter remains authoritative.
Disk checks retain a fixed reserve plus source-relative processing headroom.

Each video owns a UUID directory below `SCENEMIND_DATA_DIR`. `folder_for` canonicalizes UUID input,
so callers cannot construct arbitrary paths. OpenCV verifies a readable video stream, positive
metadata, the duration limit and the 4K pixel bound. FFmpeg extracts scaled JPEGs into a temporary
directory. Only a complete result replaces the live frame directory, and the manifest is written
through an atomic temporary file.

The browser polls `/videos` and displays real states: waiting, extracting frames, ready or failed.
It never invents a percentage.

## Visual retrieval

The Visual stage loads the pinned `openai/clip-vit-base-patch32` revision, applies its matching
processor, encodes JPEG batches and L2-normalizes the output. Embeddings are stored locally and
loaded into FAISS `IndexFlatIP`. Because query and frame vectors are normalized, inner product is
cosine similarity. Exact search is appropriate for the measured local collections and avoids an
approximate-index tuning surface.

At query time CLIP encodes text and FAISS returns frames with timestamps. Five-second sampling is a
known recall boundary: content between samples cannot be recovered by the retriever.

## Speech retrieval

The Speech stage extracts a temporary mono 16 kHz WAV, runs configurable faster-whisper inference,
validates segment timestamps and replaces database rows transactionally. Segment ends are clipped
to playable duration and fully out-of-range tails are discarded. The WAV is deleted in `finally`.

BM25 uses `k1=1.2` and `b=0.75` to rank transcript segments. Speech thumbnails use the nearest
sampled frame, while the seek timestamp is the segment start. BM25 is lexical: exact phrases and
rare terms work well, while paraphrases can fail.

## Product modes and fusion

- **Smart Search** directly selects Hybrid and is the UI default.
- **Spoken Content** directly selects Speech.
- **Visual Content** directly selects Visual.
- **AUTO** remains an API-compatible experimental route and is outside the normal v1.0 interface.

Hybrid retrieval groups contributions by exact nearest-frame thumbnail. Within each modality only
the best contribution to a thumbnail counts. The score is the sum of `1 / (60 + rank)`. This
uncapped RRF60 baseline avoids comparing incompatible CLIP and BM25 raw values. It has a documented
failure mode: weak evidence from both modalities can outrank strong evidence from one modality.
Alternative caps, calibrated raw-score fusion and evidence-preserving rank quotas failed frozen
gates, so runtime behavior remains unchanged. The evidence-preserving candidate exists only under
`ml/experiments/evidence_preserving_fusion_v1`; it reserves one unique Top-5 bucket from each lane
and fills the remainder in production RRF order. Its aggregate validation gain came only from
Speech while Hybrid MRR regressed, so it is not imported by the backend.

Responses include timestamps, thumbnails, evidence type and transcript excerpts where available.
The UI hides raw scores and presents results as possible matches.

## Durable processing

Inline mode is the simplest local path. With `SCENEMIND_DURABLE_JOBS=true`, upload and stage requests
enqueue SQL jobs and return. One worker supervisor per shared data directory claims jobs with
compare-and-set transitions. It starts a reusable spawned inference child so model weights can stay
warm across jobs.

Retries are bounded and receive persisted backoff. Each kind has a configurable deadline. On a
deadline or crashed child, the supervisor terminates the process tree and removes only disposable
partial artifacts. Worker locks prevent multiple supervisors or inference writers from owning the
same local directory. Delivery is at-least-once; stage operations are designed to replay safely.

## Persistence

- Atomic JSON manifests own video metadata and frame state.
- Local NumPy/FAISS-compatible artifacts own visual embeddings and index metadata.
- SQLAlchemy models store transcripts, segments and durable jobs.
- Alembic migrations run at API/worker startup with SQLite immediate locking or a PostgreSQL
  advisory transaction lock.
- SQLite is the verified simple local default. PostgreSQL migration and queue behavior have a
  disposable validation path.

Uploaded media, frames, audio intermediates, embeddings, indexes, databases, model weights and
caches are ignored by Git.

## URL acquisition

`POST /videos/import-url` creates the same UUID-owned video record as upload and queues a durable
`acquire` job. Provider code lives in `app.url_ingest`; Direct Media streams with manual validated
redirects, while the YouTube adapter wraps pinned yt-dlp with structured metadata, a 720p ceiling,
process timeout and live output-size monitoring. After acquisition the existing `inspect_video`
and `ingest` job remain authoritative. Provenance lives in the atomic manifest; retryability lives
on the normal durable job row through migration 003. Search modules never branch on source type.

## Transcript-grounded Ask Video

Ask Video is a separate Q&A evidence layer over the existing SQL transcript. It groups ordered
Whisper segments into deterministic chunks bounded by 45 seconds and 900 characters with one
segment of overlap. Each chunk carries a stable evidence ID, video ID, start/end timestamps, text,
and underlying segment IDs. The existing BM25 implementation ranks chunks without changing
production Speech Search.

The `AnswerGenerator` boundary receives one question and the selected evidence. Ordinary questions
retain at most five chunks. The rejected structured branch can expand to eight bounded evidence
units. A deterministic analyzer extracts explicit count, temporal and relation constraints. The
OpenAI Responses implementation requests strict structured JSON with an answerability decision,
separately cited claims, temporal roles and missing requirements; response storage is disabled. The
model returns evidence IDs, never timestamps. The backend rejects unknown IDs, uncited claims,
declared evidence gaps, duplicate list claims, wrong explicit counts, invalid temporal roles and
wrong timestamp direction, then resolves citations to original timestamps. An invalid contract
safely abstains. No positive lexical evidence causes a pre-provider abstention.

`POST /videos/{id}/ask` requires a ready video, ready non-empty transcript, the feature gate and a
local provider key. `GET /videos/{id}/ask/status` exposes only enabled/configured state and model
name. Upload and URL provenance do not alter this path. The feature is disabled by default after
Decision D; structured validation failed correctness, grounding, citation, abstention and
hard-negative safety gates. Find Moments, BM25 Top-5 and general transcript chunking are unchanged.

The semantic experiment deliberately keeps discovery units separate from citation units. Context
sections may nominate a region, but generator evidence and citations contain only 30-second/600-
character fine units with original segment IDs and timestamps. The encoder is local
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; stored vectors are 384-dimensional normalized
float32 values. Validation rejected this path, so indexes are experimental local artifacts only.

## Failure handling

Upload failures remove the partial UUID directory. Failed frame extraction removes staged and live
partial frames. Visual artifacts are promoted atomically. Transcript replacement is transactional.
The durable supervisor marks abandoned running jobs failed on restart, applies bounded retry, and
exposes terminal errors for explicit retry. Inline startup marks interrupted stages failed rather
than pretending they completed.

Public error messages are bounded. The UI handles backend unavailability, invalid or oversized
uploads, processing failure, search/index failure and empty result lists. Optional Basic/Bearer
authentication protects all endpoints except health; cross-origin writes are limited to configured
origins.

## Resource model

The default envelope is 1 GiB and 60 minutes. Upload API memory stayed near 110 MB in measured
419 MiB and 45-minute runs because bytes stream to disk. The 45-minute sequential Whisper/CLIP run
completed in 245.4 seconds and peaked at about 3.02 GB across the worker process tree. Completed
storage was 1.064× source size, and temporary residue was zero.

Model loading dominates cold search. Final English Acceptance V2 measured a 10.42-second first
CLIP-backed search and 46–252 ms for the next 29 requests. Hardware, content and model cache state
will change these values.

## Deployment boundary

SceneMind is a local single-operator application. Optional authentication is not tenant ownership
or RBAC. Public TLS termination, object storage, user quotas, distributed leases, high availability
and untrusted multi-user isolation are outside v1.0. Use one API and one durable worker per shared
local data directory.

The Docker/Compose files validate the backend, PostgreSQL and job foundation. The base image does
not install the complete ML environment or serve the frontend. Full containerized ML deployment is
not claimed.

## Evaluation boundary

Production modules live under `backend/app`. Candidate models and ranking methods live under
`ml/experiments` and cannot enter runtime implicitly. Evaluation uses source-disjoint splits,
checksum-frozen manifests and query-level reports. Protected holdout evidence is not a tuning set.
The current search core is frozen for `1.0.0-rc1`. Evidence-Preserving Hybrid Fusion V1 used
development and one source-disjoint validation split, stopped at its cross-category gate, and did
not consume Hybrid Holdout V1. Speculative work is Future Work.

See [README.md](README.md), [long-video support](docs/LONG_VIDEO_SUPPORT.md),
[DECISIONS.md](DECISIONS.md), and the [portfolio case study](docs/PORTFOLIO_CASE_STUDY.md).
