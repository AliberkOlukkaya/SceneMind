# URL ingestion

SceneMind accepts a local video upload or a supported public video URL. URL media is first
materialized inside the video's UUID-owned local directory, validated by the existing trusted
media inspection path, and then handed to the same FFmpeg frame, Whisper, CLIP/FAISS, BM25 and
RRF60 pipeline as an upload. URL videos do not have a separate search implementation.

## Providers

`URLIngestProvider` defines `supports`, `external_id`, `inspect`, `acquire` and `cleanup`.

- `DirectMediaProvider` accepts HTTP(S) URLs whose path ends in MP4, MOV, WebM, MKV or AVI. It
  follows at most the configured redirect limit, streams to a `.part` file, enforces declared and
  observed byte limits, reserves disk and validates the resulting stream with OpenCV/FFmpeg.
- `YouTubeProvider` accepts normal `youtube.com`, mobile, Shorts and `youtu.be` video URLs. It wraps
  pinned `yt-dlp==2026.8.19` behind one module, disables playlists, selects video at or below 720p
  with audio, uses the bundled FFmpeg for merging, monitors output size and enforces a process
  deadline. Metadata comes from structured JSON rather than progress text.

No provider uses cookies, browser profiles, credentials, login automation, paywall workarounds or
DRM circumvention. YouTube availability can change outside SceneMind when the platform changes its
extractor requirements. Users are responsible for permission to process submitted content.

## Request and durable flow

`POST /videos/import-url` accepts `{ "url": "https://…" }`. It performs bounded URL/provider
validation, writes a provenance-bearing manifest, queues an `acquire` job and returns `202`. The
worker performs:

```text
queued → fetching → validating → normal ingest/frame extraction → ready
```

Speech transcription and visual-index construction retain the existing on-demand controls and
durable `speech`/`visual` jobs used by uploaded videos. The normal library, local player, three
search modes and click-to-seek behavior are shared. A source link is provenance only; playback and
analysis use the authorized local artifact.

With durable jobs disabled, the same work runs as a FastAPI background task. Durable deployments
should enable `SCENEMIND_DURABLE_JOBS` and run the existing worker.

## URL safety

Only HTTP and HTTPS are accepted. URLs with embedded credentials are rejected. Every submitted
host and every redirect destination is resolved and all returned addresses must be globally
routable. Loopback, private, link-local, multicast, reserved, unspecified and known metadata hosts
are rejected for both IPv4 and IPv6. A direct-media destination is resolved again after connection
headers arrive; an address-set change fails closed. Redirects are manual and bounded.

YouTube media subrequests are selected by the pinned YouTube extractor after the submitted YouTube
host passes validation; SceneMind does not offer arbitrary extractor/provider selection.

## Limits and failures

Remote imports use the same configured maximum bytes, duration, 4K stream limit, disk reserve and
processing deadlines as uploads. Remote metadata can reject obvious size/duration violations
early, but it is never authoritative; the local artifact is measured and inspected again.

Direct downloads do not trust Content-Type or Content-Length. Content-Length can only reject early;
the streaming byte counter is authoritative. YouTube output is watched while yt-dlp runs and is
removed on overflow, timeout or provider failure. Partial downloads, staged frames, temporary
audio and temporary indexes are cleaned on failure/deadline paths.

Temporary network, 5xx and timeout failures are retryable within the existing bounded attempt
limit. Unsupported schemes/providers, private destinations, unavailable private content, size or
duration violations and invalid media are non-retryable. User messages omit command output and
internal paths; technical details remain in local logs.

## Provenance and duplicates

Manifests distinguish `source_type: upload` from `source_type: url`. URL records also retain
`source_provider`, canonical `source_url`, `source_external_id` and bounded provider metadata.
Existing upload manifests remain valid; absence of `source_type` means a historical upload.

An exact provider plus external-ID duplicate returns the existing video. SceneMind does not perform
global perceptual deduplication or silently reprocess the same provider item.

Alembic migration `003` adds durable-job retry classification. Video provenance remains in the
existing canonical atomic manifest because SceneMind has no parallel SQL video table.

## Verified behavior and limitations

The DirectMedia equivalence run produced identical media, duration, frame timestamps, transcript,
visual-index shape and all nine frozen Top-5 query outputs versus upload. A real public YouTube run
acquired and processed the official Blender Foundation Big Buck Bunny video, built its visual index
and returned visual results with no partial residue. See
[`URL_INGESTION_V1_EQUIVALENCE.md`](../ml/evaluation/URL_INGESTION_V1_EQUIVALENCE.md).

URL ingestion creates a clean future boundary for evidence consumers, but this milestone adds no
LLM, answer generation, RAG endpoint, prompt or chat UI.
