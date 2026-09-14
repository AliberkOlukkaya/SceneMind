# Query routing calibration and AUTO mode

This milestone selects exactly one existing search path from query text: `VISUAL`, `SPEECH`, or `HYBRID`. It adds no semantic model, embedding API, detector, OCR, RAG, action model, tracking, segmentation, pose estimation, or fine-tuning. The production artifact is a 54-parameter multinomial linear classifier over 18 readable lexical features.

## Frozen calibration evidence

Three source-disjoint Wikimedia Commons videos provide 961.304 seconds of real lecture/tutorial/demo material, 205 Whisper-tiny segments, and 36 manually reviewed queries: 12 per route. Each source includes checksum, license, attribution, duration, speech notes, and 12 route/relevance annotations in `calibration_v1.json`. Media, frames, transcripts, embeddings, and model caches remain ignored.

The sources are Akwugo's *Video tutorial* (CC BY-SA 4.0), AlexandraShagzhina's *Demo Video Tutorial* (CC BY-SA 4.0), and the MITx *Course Overview (6.002x)* (CC BY 2.0). The set includes paired wording such as “When does he say Python?” versus “Where is Python visible?”, explicit mixed queries, abstract concepts, spoken visual nouns, and screen-plus-speech requirements. Natural V2's three held-out source groups remain untouched.

Prepare the ignored inputs and run the experiment:

```powershell
$env:PYTHONPATH='backend;.'
.venv/Scripts/python -m ml.experiments.query_routing.prepare
.venv/Scripts/python -m ml.experiments.query_routing.run_experiment
.venv/Scripts/python -m ml.experiments.query_routing.assemble_report `
  data/query-routing/report.json
```

Preparation downloads checksum-bound originals, extracts production-equivalent five-second 480-pixel JPEGs, and creates a local Whisper-tiny transcript when absent. The runner performs real CLIP, BM25, and RRF retrieval on calibration, fits only calibration rows, and then reads the already frozen Natural V2 query-path evidence once. A second identical held-out evaluation is required and executed only after the gate passes.

## Routers

The heuristic recognizes explicit speech verbs, visual/color/spatial terms, and mixed “while/on screen” phrasing; uncertain text routes Hybrid. It is deterministic and readable. The learned router uses quote presence, query length, opening words, lexical category counts, mixed-intent indicators, and punctuation. Class-balanced softmax regression uses fixed initialization and leave-one-source-out selection over regularization 0.01/0.1/1.0. It has no text embedding or hidden language model.

Calibration chooses regularization 0.01. Leave-one-source-out routing accuracy is 94.44%; the final calibration fit is 100%. Confidence-based Hybrid fallback tests thresholds 0/0.45/0.55/0.65/0.75 using calibration end-to-end search quality. Threshold 0 wins, so fallback adds no behavior and confidence is diagnostic rather than a no-match score.

The classifier input is a query string. The output is one route and a softmax confidence. Median/p95 CPU routing latency is 0.031/0.042 ms, the compact measured model is 2,077 bytes, and production requires only NumPy already used by visual search. A rule-only alternative is cheaper but less accurate; a tree adds unjustified capacity, and an LLM/transformer router violates the local latency and model constraints.

AUTO is implemented in `backend/app/routing.py` and bound to `query_router_v1.json`. `SCENEMIND_AUTO_ROUTING_ENABLED=false` makes AUTO safely fall back to Hybrid. Explicit Visual, Speech, and Hybrid overrides always bypass the classifier. The UI defaults to Auto and shows the selected route within its existing evidence line. Speech-routed searches still require a ready transcript.

See [results](../../evaluation/QUERY_ROUTING_RESULTS.md), [failures](../../evaluation/QUERY_ROUTING_FAILURES.md), and the [personal-video acceptance protocol](../../evaluation/PERSONAL_VIDEO_ACCEPTANCE_PROTOCOL.md).
