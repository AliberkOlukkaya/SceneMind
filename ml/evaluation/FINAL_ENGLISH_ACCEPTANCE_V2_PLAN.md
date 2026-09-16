# Final English Acceptance V2 plan

Status: executed once on 2026-09-16 with Decision C. This document remains the unchanged pre-run protocol and frozen gates; see `FINAL_ENGLISH_ACCEPTANCE_V2_RESULTS.md`. The acceptance video must never be used for tuning or rerun as fresh acceptance.

## Product path

Use the normal frontend and durable backend pipeline exactly as a new user would. The primary and default mode is **Smart Search**, which sends the existing internal `hybrid` mode. AUTO routing is outside this protocol and receives no routing score. **Spoken Content** (`speech`) and **Visual Content** (`visual`) may be used only after the primary judgment to diagnose failures.

Do not change CLIP, Whisper, BM25, RRF, FAISS, embeddings, five-second sampling, ranking, frame extraction, workers, or no-match behavior for this run.

## Independent source and frozen evidence

Use one real, continuous 30-60 minute English-speaking lecture, tutorial, technical presentation, or software/project demonstration that has legal local-use rights and has never contributed to training, tuning, thresholds, vocabulary, rules, prompts, or prior SceneMind evaluation.

Watch the entire source before searching. Record its exact SHA-256, rights, duration, inspection status, and lack of tuning history. Write and freeze 25-40 natural English queries with reviewed evidence intervals and rationales. Include spoken topics and phrases, visual objects and scenes, supported compositions, mixed spoken-plus-visual requests, and difficult plausible negatives. Freeze the source and query manifest before product retrieval.

## Measurements

For each query, record Smart Search results and human judgment for useful Top-1, Top-3, and Top-5, timestamp usefulness/error, latency, PASS/PARTIAL/FAIL, and a concrete failure reason. A result is useful only when a normal user clicking it would reasonably feel the requested moment was found. Candidate return or interval overlap alone is insufficient.

Record upload and total processing time, frame and ASR segment counts, peak worker/API memory where available, persistent storage, retries, failures, and temporary residue. For negative queries, record whether "Most relevant moments" and "Possible matches" remain honest and avoid implying that unsupported content definitely exists.

## Frozen gates

- Useful Top-5: at least 90%.
- Useful Top-3: at least 85%.
- Useful Top-1: 70% or higher preferred and reported separately.
- No catastrophic ingest or processing failure.
- Search latency remains interactive and is reported as median and p95.
- Returned timestamps are useful to a normal user and timestamp errors are reported.
- Negative-query presentation remains conservative and does not imply unsupported certainty.

Do not change these gates after observing results. Explicit-mode diagnostics cannot replace or raise the primary Smart Search metrics. Report failures even when an explicit mode recovers them.

## Next action

Wait for the new independent video. Once supplied, inspect it fully, freeze the manifest, validate the checksum and eligibility, run the normal durable pipeline once, and publish the human-reviewed V2 result. Do not start another router or retrieval-model experiment in preparation for this run.
