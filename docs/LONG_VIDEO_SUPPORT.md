# Long-video infrastructure support

SceneMind accepts local videos up to 1 GiB and 60 minutes by default. These are configurable infrastructure limits, not a claim that search quality has been validated for every long video.

## Data path and memory behavior

The browser sends the File body directly. FastAPI consumes `request.stream()` and writes each received chunk once to the final UUID-owned source path. The application does not call `request.body()`, construct a full upload byte string, create a second source copy or keep the request open after the durable job is queued. `Content-Length`, when present, is rejected before a video folder is allocated if it exceeds the configured maximum. The byte counter remains authoritative when the length is absent or incorrect.

The API creates the data root, verifies a configurable free-disk reserve and includes 25% source-size processing headroom when the size is known. It checks the reserve every 64 MiB while streaming and checks processing headroom again before enqueue. A failed or oversized upload removes its partial UUID folder.

FFmpeg reads the source from disk. Ingest writes sampled 480-pixel JPEGs to `frames.tmp` and replaces the completed frame directory only after timestamps and output are valid. The unchanged sampling interval is five seconds. Speech writes one temporary mono 16 kHz PCM WAV and deletes it in `finally`; segment replacement is transactional. CLIP decodes only the configured image batch, eight frames by default, and atomically replaces `embeddings.npy`. The final vector matrix is small: 540 frames used 1,106,048 bytes in the 45-minute run.

## Durable jobs and stages

Set `SCENEMIND_DURABLE_JOBS=true` and run one API plus `python -m app.worker` against the same database, data directory and model cache. Upload returns HTTP 202 after the source is durable and the ingest job is queued; the browser can close. The API exposes product status plus `job_status` and a real stage:

- queued/running ingest: Preparing video
- queued/running speech: Transcribing
- queued/running visual: Indexing
- completed manifest/index/transcript: Ready
- terminal queue or stage error: Failed with a bounded public error

The UI displays upload, queue/preparation, transcript and index states without inventing a percentage.

Durable retries remain bounded by `SCENEMIND_JOB_ATTEMPTS`. Each failed attempt receives persisted exponential backoff. A timed-out inference child and its process tree are terminated, then the supervisor removes only disposable stage artifacts: staged frames, temporary manifest/index/embedding files, or `audio.wav`. Atomic manifests, transactional transcript writes and the index status file prevent a partial index from being treated as ready. Manual retry remains available only for a terminal failed job.

## Configuration

| Setting | Default | Purpose |
| --- | ---: | --- |
| `SCENEMIND_MAX_UPLOAD_BYTES` | 1,073,741,824 | Maximum streamed source bytes |
| `SCENEMIND_MAX_DURATION` | 3,600 | Maximum inspected duration in seconds |
| `SCENEMIND_MIN_FREE_DISK_BYTES` | 536,870,912 | Free space retained after writes |
| `SCENEMIND_PROCESSING_DISK_HEADROOM_RATIO` | 0.25 | Extra source-relative headroom reserved for derived files |
| `SCENEMIND_PROCESSING_TIMEOUT` | 1,800 | Individual FFmpeg subprocess deadline |
| `SCENEMIND_INGEST_JOB_TIMEOUT` | 1,800 | Durable ingest job deadline |
| `SCENEMIND_SPEECH_JOB_TIMEOUT` | 7,200 | Durable speech job deadline |
| `SCENEMIND_VISUAL_JOB_TIMEOUT` | 3,600 | Durable visual job deadline |
| `SCENEMIND_JOB_TIMEOUT` | 7,200 | Bounded fallback for unknown job kinds |

All timeout fields accept 1 through 86,400 seconds. The upload and duration settings retain positive validation. The disk headroom ratio accepts 0 through 2.

## Storage amplification

Persistent storage consists of the source, sampled JPEGs, a float32 embedding matrix, small atomic JSON manifests/index state and transcript/job rows in SQL. The temporary PCM audio is about 32,000 bytes per second and exists only during transcription.

Measured completed amplification was 1.013x for the 419 MiB silent tutorial and 1.064x for the 45-minute audio stress video. The stress run used 162,942,220 source bytes, 9,037,471 frame bytes, 1,106,048 embedding bytes and a 135,168-byte database. During speech, its temporary 45-minute PCM audio can add about 86.4 MB before cleanup. Actual JPEG size depends on content.

## Reproduce the measurements

The validation tool starts separate Uvicorn and durable-worker processes, uploads in 1 MiB HTTP chunks, polls real endpoints, and records API/worker process-tree RSS and storage:

    .venv/Scripts/python scripts/validate_long_video.py VIDEO --root data/RUN --output data/RUN.json

For a silent video use `--skip-speech`. Choose an unused port with `--port`. The root must not already exist, preventing accidental deletion or result mixing.

Create a 45-minute ingestion-only stress fixture from local media without committing it:

    .venv/Scripts/python scripts/create_long_video_stress.py SOURCE data/long-video-stress-45m.webm --minutes 45

The command stream-copies repeated source material. It tests duration, resources and worker stability only; it provides no retrieval-quality evidence. See [measured results](../ml/evaluation/LONG_VIDEO_INGEST_RESULTS.md).
