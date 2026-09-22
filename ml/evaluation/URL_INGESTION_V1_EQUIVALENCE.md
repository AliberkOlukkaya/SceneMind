# URL Ingestion V1 Functional Equivalence

Protocol frozen before the real comparison. This milestone compares acquisition paths, not search
models, and does not tune retrieval.

## Source and paths

Use the CC BY-SA 4.0 Wikimedia Commons video `Find link - lightning talk.webm` from the frozen
path-aware no-match inventory. Materialize it independently through:

1. `DirectMediaProvider` from its public `upload.wikimedia.org` URL.
2. The existing local upload path from the checksum-bound local copy.

Process both with the same five-second frame sampler, Whisper configuration, CLIP revision, BM25,
FAISS and uncapped RRF60. Downloaded media, frames, transcripts, indexes and databases stay ignored.

## Frozen checks

- Actual media must pass the existing size, duration and video-stream validator.
- Duration difference must be at most 0.25 seconds.
- Frame count must match and paired frame timestamps must differ by at most 0.05 seconds.
- Visual index vector count and dimension must match.
- Normalized transcript similarity must be at least 0.95 and segment-count difference at most one.
- For three predeclared Visual, Speech and Smart Search queries, each path must return results and
  paired Top-5 timestamps must differ by at most one five-second sampling interval. Exact order is
  preferred but is not required for model/runtime nondeterminism.
- Both records must use the normal local media, transcript, index, search and click-to-seek API
  contracts. Search implementation and expected retrieval values must not change.

Queries are frozen as:

| Mode | Query |
| --- | --- |
| Visual | `a presentation slide with a browser interface` |
| Visual | `a speaker standing beside the projected screen` |
| Visual | `a web page shown on the screen` |
| Speech | `Kansas` |
| Speech | `airport` |
| Speech | `languages` |
| Smart Search | `the speaker discusses Kansas while a slide is visible` |
| Smart Search | `the airport example on the presentation screen` |
| Smart Search | `languages mentioned during the browser demonstration` |

The comparison passes only if every structural gate and at least eight of nine query comparisons
meet the timestamp tolerance, with no complete failure in any mode.

## Results

The one-shot DirectMedia comparison passed every gate. The acquired and uploaded copies were
byte-identical at 35,611,959 bytes and both measured 219.443 seconds. Each produced 44 frames at
identical timestamps, 60 identical Whisper segments (normalized similarity 1.0000), and a 44×512
visual index. All nine frozen Visual, Speech and Smart Search comparisons returned identical Top-5
timestamps. Acquisition took 3.994 seconds. URL ingest, speech and visual processing took
6.827/19.387/2.494 seconds; the upload path took 6.270/21.925/32.776 seconds, with its visual time
including the cold model load. Neither path left partial artifacts.

A separate public-provider run used the official Blender Foundation Big Buck Bunny YouTube video
(`aqz-KE-bpKQ`, CC BY 3.0). The pinned yt-dlp adapter acquired a bounded 720p-or-lower 81,554,084
byte artifact in 14.720 seconds. The normal pipeline measured 634.567 seconds, extracted 127
five-second frames, completed Whisper with zero speech segments, built the normal visual index and
returned five results for `animated rabbit in a field`. Processing took 138.191 seconds. It used
zero retries and left no partial artifact. Zero segments is expected for this non-spoken animated
film and is not presented as speech-retrieval evidence.

The machine-readable observations are in `ml/evaluation/reports/url-ingestion-v1.json`. Media,
frames, embeddings, model cache and databases remain ignored.
