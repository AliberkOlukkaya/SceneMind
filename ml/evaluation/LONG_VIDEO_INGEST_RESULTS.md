# Long-video ingestion and resource results

Date: 2026-09-15

Base commit: 2ac5202f9c9dc6124c62d8298f93c1a5eee8abe5

Outcome: **Infrastructure success gate passed.** Production search models, sampling, routing and ranking are unchanged. This does not establish 30-60 minute search quality.

## Method

Both runs used separate local Uvicorn and durable-worker processes. The client streamed 1 MiB chunks through the normal HTTP upload endpoint. Durable SQL jobs then ran production FFmpeg ingestion and CLIP indexing; the audio stress run also ran the unchanged Whisper-tiny pipeline. RSS includes each measured process tree. All large sources and outputs remain ignored.

The real source was the unmodified `Blender_Tutorial._A_cup_of_tea..webm`, 439,295,727 bytes. It is silent, so its ASR stage was explicitly skipped rather than reported as successful. The stress source repeats the local Blender interface demo by stream copy to exactly 2,700 seconds. It is an infrastructure fixture, not retrieval evidence.

## Measurements

| Metric | Real tutorial | 45-minute stress video |
| --- | ---: | ---: |
| Source bytes | 439,295,727 | 162,942,220 |
| HTTP result | 202 | 202 |
| Upload time | 1.641 s | 0.646 s |
| Peak API RSS during upload | 109,314,048 B | 109,953,024 B |
| Duration | 1,369.633 s | 2,700.000 s |
| Ingest time | 54.174 s | 43.792 s |
| Whisper time | not applicable: no audio | 160.569 s |
| CLIP indexing time | 67.162 s | 41.033 s |
| Total processing time | 121.335 s | 245.394 s |
| Worker process-tree peak RSS | 1,148,583,936 B | 3,017,990,144 B |
| Steady worker process-tree RSS | 1,142,169,600 B | 1,562,931,200 B |
| Sampled frames | 274 | 540 |
| ASR segments | 0 / skipped | 373 |
| Transcript status | not started / silent | ready, English |
| Timestamp continuity | not applicable | valid |
| Visual index | ready | ready |
| Persistent run storage | 444,845,520 B | 173,378,165 B |
| Completed storage/source | 1.013x | 1.064x |
| Temporary bytes after completion | 0 | 0 |

Peak API RSS stayed near 110 MB for both a 419 MiB source and a 163 MiB source. This, together with the 1 MiB client chunks and server `request.stream()` path, shows that upload memory does not scale as a full source-file buffer. The worker peak is material: users should budget about 3.1 GB for the measured sequential Whisper-plus-CLIP workload, with roughly 1.6 GB retained by the persistent worker after both models load.

## Failure and cleanup verification

Automated tests cover configurable size/duration, a declared upload above the old 250 MiB limit without a huge fixture, declared and streamed over-limit rejection, free-disk/headroom rejection, multi-chunk streaming, failed FFmpeg staging cleanup, supervisor cleanup after a killed stage, stage-specific deadlines and runtime limit display. Existing durable-worker tests cover queue/running/failed states, bounded retry/backoff, explicit retry, child termination on deadline and stable reuse.

Frame and embedding outputs are atomically promoted. Transcripts are replaced in one transaction. Both real runs ended with zero `frames.tmp`, `embeddings.tmp` or `audio.wav` bytes.

## Final report

| # | Item | Result |
| ---: | --- | --- |
| 1 | Previous upload limit | 262,144,000 bytes / 250 MiB |
| 2 | New configurable upload limit | 1,073,741,824 bytes / 1 GiB |
| 3 | Previous duration limit | 1,800 seconds / 30 minutes |
| 4 | New configurable duration limit | 3,600 seconds / 60 minutes |
| 5 | Streaming upload implemented? | Yes; direct request chunks to disk |
| 6 | 419 MiB real video accepted? | Yes; HTTP 202 without transcoding |
| 7 | Real video processing completed? | Yes; ingest and visual index ready; source has no audio |
| 8 | Real video processing time | 121.335 s |
| 9 | Real video peak RAM | 1,148,583,936 B worker tree; 109,314,048 B API upload peak |
| 10 | Real video disk usage | 444,845,520 B |
| 11 | Stress-test duration | 2,700 s / 45 minutes |
| 12 | Stress-test processing time | 245.394 s |
| 13 | Stress-test peak RAM | 3,017,990,144 B worker tree; 109,953,024 B API upload peak |
| 14 | Stress-test disk usage | 173,378,165 B |
| 15 | Whisper long-job status | Ready; 373 segments, valid ordered timestamps |
| 16 | CLIP/indexing long-job status | Ready; 540 frames, unchanged five-second sampling |
| 17 | Cleanup verified? | Yes; real runs and failure injection |
| 18 | Retry/failure behavior verified? | Yes; bounded durable retry, timeout termination and partial cleanup |
| 19 | Search regression passed? | Yes; frozen retrieval/routing suites passed in the 118-test repository run |
| 20 | Exact remaining blocker before v1.0 | Long-video infrastructure works, but real 30-60 minute user search usefulness remains unvalidated; Turkish compatibility and honest unsupported-query behavior also remain acceptance blockers |
