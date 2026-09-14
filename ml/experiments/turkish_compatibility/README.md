# Turkish query compatibility experiment

This isolated experiment measures where natural Turkish queries lose quality in SceneMind's existing AUTO, CLIP, BM25 and reciprocal-rank-fusion paths. It does not alter production search.

## Frozen evidence

The manifest contains 72 natural Turkish queries: 42 calibration queries from three source groups and 30 held-out queries from three different source groups. Seven legally reusable Wikimedia Commons videos cover Turkish tutorials, a software demonstration, a presentation, speech-heavy education and ordinary visual footage. The two Vikipedi 101 clips share one calibration source group, so they cannot cross the source boundary independently.

The manifest records source pages, download URLs, licenses, attribution, checksums, duration, language and real Whisper-tiny observations. Source media, frames, transcripts, embeddings and model caches stay under ignored data directories.

The frozen personal acceptance manifest is protected in two ways: its file hash is embedded in the development manifest, and validation rejects any development source checksum matching a personal-acceptance video. Calibration and held-out source groups must also be disjoint.

## Reproduction

Run from the repository root with the project virtual environment:

    .venv/Scripts/python -m ml.experiments.turkish_compatibility.build_manifest
    .venv/Scripts/python -m ml.experiments.turkish_compatibility.prepare
    .venv/Scripts/python -m ml.experiments.turkish_compatibility.run_experiment

The prepare step downloads licensed sources and performs real production-style frame and speech extraction. The experiment first freezes a native-production baseline, selects router and retrieval candidates only on calibration data, and evaluates the selected candidate once on held-out sources. The ignored baseline artifact is bound into the committed report by SHA-256.

The optional semantic diagnostic runs only after the cheap candidate fails:

    .venv/Scripts/python -m ml.experiments.turkish_compatibility.semantic_diagnostic

It uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 at revision e04d38a7d68c60a7a95390045400a555127ab033 through direct Transformers mean pooling. The model is Apache-2.0, produces 384-dimensional vectors, runs locally on CPU and is not a production dependency. No weights are committed.

## Candidate design

The selected order is original Turkish query to a Turkish router, followed by path-specific query adaptation for retrieval. Cheap processing includes NFC, Turkish casing, punctuation and apostrophe handling, auditable intent roots, suffix-aware tokens and a small technical-term lexicon. Source-leave-one-out calibration compares rules, the 14-feature linear router, word n-grams, character n-grams and combined n-grams.

The held-out gates were fixed before evaluation: routing at least 90%, AUTO R@5 at least 85%, English R@5 regression no more than two percentage points, and preferred added routing/adaptation latency below 5 ms. Routing reached 76.67% and cheap AUTO R@5 reached 80%, so the candidate is not eligible for promotion or a personal-acceptance rerun.

See [results](../../evaluation/TURKISH_COMPATIBILITY_RESULTS.md), [failures](../../evaluation/TURKISH_COMPATIBILITY_FAILURES.md), and the [machine report](../../evaluation/reports/turkish-compatibility-v1.json).
