# Current Hybrid fusion architecture

This is a code audit of the production path at commit `c42ee485dcc4af8b5c793291c306f7aac1826e6a`. It describes observed implementation, not intended behavior from older design documents. Final Acceptance V2 media and query evidence were not used to derive rules or parameters.

## Request and routing

The frontend initializes `mode` to `hybrid` and renders `<option value="hybrid">Smart Search</option>` in `frontend/src/app/search.tsx`. A search sends `GET /videos/{video_id}/search?q=...&k=5&mode=hybrid`. The normal UI does not send AUTO.

`backend/app/hybrid.py::search` validates the nonblank query and calls `app.routing.resolve_mode`. An explicit `hybrid` value is returned unchanged with reason `explicit_override`; query preprocessing does not happen at this routing boundary. The response records `requested_mode=hybrid` and `selected_route=hybrid`.

## Speech candidate path

`backend/app/hybrid.py::speech_results` loads all ordered `Segment` rows for the video from SQL. These segments were produced by `backend/app/speech.py::infer_audio`: FFmpeg first creates mono 16 kHz PCM audio, then cached `faster-whisper` tiny runs with beam size 5 and VAD. The stored segment text and original Whisper start/end times are the retrieval units.

`backend/app/hybrid.py::tokens` applies Unicode-aware `\w+` tokenization after `casefold()` and removes the fixed stopword set `a, an, the, is, are, where, when, does, find, show, in, of`. There is no stemming, lemmatization, synonym expansion, query expansion, language model, phrase index, or semantic speech encoder.

`backend/app/hybrid.py::bm25` computes BM25 over every transcript segment for each request. Parameters are `k1=1.2`, `b=0.75`, with positive smoothed IDF `log(1 + (N-df+0.5)/(df+0.5))`. Segments sort by descending raw BM25 score and then ascending segment start. Zero-score segments are dropped. Hybrid requests retain at most 50 Speech candidates.

Each Speech segment is assigned the production frame whose timestamp has the smallest absolute distance from the segment start. Its candidate/grouping identity becomes that frame's thumbnail URL. The result timestamp remains the exact segment start, not the frame time.

## Visual candidate path

`backend/app/visual.py::visual_search` loads the video's production `embeddings.npy` and passes the raw query string to `backend/app/encoder.py::CLIPEncoder.text`. The pinned CLIP processor tokenizes with padding/truncation and `max_length=77`. Image and text embeddings are L2-normalized.

`backend/app/encoder.py::rank_vectors` creates a fresh exact FAISS `IndexFlatIP`, adds every normalized five-second frame vector, and searches the normalized query vector. Inner product is therefore cosine similarity. Results remain in FAISS score order. Hybrid requests retain at most 50 Visual candidates. A separately configured calibration cutoff can filter Visual candidates, but normal production has no score normalization between modalities and RRF never consumes raw CLIP magnitude.

## Candidate grouping and fusion

`backend/app/hybrid.py::fuse` is the complete fusion and post-ranking implementation:

1. It visits Visual candidates, then Speech candidates.
2. Within each modality it permits one vote per thumbnail key. A second Speech segment mapped to the same nearest frame is discarded for fusion. Rank is assigned before this deduplication, so a later unique Speech frame retains its original list rank even when earlier duplicates were skipped.
3. Cross-modality candidates overlap only when their thumbnail strings match exactly. There is no independent interval-overlap test or configurable temporal window.
4. Each retained modality contributes `1 / (60 + retriever_rank)`. The fixed RRF constant is 60. There are no modality weights, learned weights, raw-score addition, min-max/z-score normalization, thresholds on total RRF, or other fusion terms.
5. A shared thumbnail receives both contributions. This overlap bonus can exceed either single-modality candidate even when both raw scores are weak; raw scores are provenance only.
6. For a shared bucket, Speech overwrites the displayed timestamp/end/text with the selected Speech segment. The thumbnail remains the grouping key and visual context.
7. Buckets sort by descending summed RRF score, then ascending displayed timestamp.
8. `fuse` slices the sorted list to the requested `k`; the normal frontend requests five. There is no reranker, diversity pass, spacing rule, no-match step, or other post-ranking logic.

If only one index is ready, Hybrid fuses the available modality and reports it in `modalities_used`. If neither is ready, it returns HTTP 409. Speech-only requires a ready transcript; Visual-only requires a ready visual index.

## Diagnostic boundary

`ml/evaluation/hybrid_fusion_trace.py::trace_fusion` is evaluation-only. It accepts the same already-ranked Visual/Speech lists, calls `app.hybrid.fuse` for the authoritative output, reconstructs the fixed RRF bookkeeping, and asserts identical candidate IDs, timestamps, scores, and order. It records raw candidates, per-retriever ranks/scores, RRF contributions, overlap, thumbnail deduplication, post-fusion rank, displacement, and final Top-K membership.

The current architecture has no normalized score, global pre-fusion rank, independent moment ID, or non-RRF fusion contribution. The trace records these as `null` with an explicit `not_applicable` reason instead of inventing values. It is not imported by the FastAPI application, adds no endpoint or flag, and exposes nothing to the frontend.
