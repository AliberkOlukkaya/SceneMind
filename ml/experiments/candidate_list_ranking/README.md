# Candidate-list ranking and no-match experiment

This isolated experiment asks whether cheap statistics can turn the existing five-second CLIP top-20/top-50 pool into a better final list and a trustworthy query-level rejection decision. It does not change an endpoint, index, model, sampler, or UI.

The frozen input contains 66 calibration queries from seven video sources and 31 held-out queries from three disjoint sources. Candidate labels are computed from the existing frozen relevance intervals. Model fitting rejects any non-calibration row. Regularization and temporal weights use leave-one-video-source-out calibration or calibration aggregate metrics; held-out labels never select a feature, weight, model, or threshold.

Candidate features are CLIP score, within-list z/min-max normalization, reciprocal/log rank, top and adjacent margins, list mean/standard deviation/range, ten-second temporal support, nearest time gap, cosine to the top frame and candidate centroid, and adjacent-frame visual change. Query-level no-match features summarize the best score, margins, distribution, normalized entropy, support, centroid agreement, upper-tail fraction, and list size. The models are dependency-free class-balanced L2 logistic regressions. A deterministic temporal linear score is also measured. Normalization and margin baselines preserve the original order by construction. The calibration set is too small for gradient boosting or an MLP, so neither was run.

The existing API already makes routing explicit through `mode=visual|speech|hybrid`. The routing audit uses that policy and the benchmark's frozen modality requirement to select an existing real CLIP, BM25, or RRF result. It does not infer a route from query text: calibration contains no speech queries, so a learned or lexical automatic router would be held-out leakage. Cross-path best scores and result counts are recorded only as diagnostic evidence.

Run the real measurement and freeze the decision:

```powershell
$env:PYTHONPATH='backend;.'
.venv/Scripts/python -m ml.experiments.candidate_list_ranking.run_experiment
.venv/Scripts/python -m ml.experiments.candidate_list_ranking.assemble_report `
  data/candidate-list-ranking/report.json
```

The runner loads the pinned cached CLIP model and ignored frame embeddings, reconstructs exact top-50 lists, fits on calibration, and evaluates held-out once. It stores the raw run under ignored `data/`; the assembler writes the compact reproducible evidence report. Source media, embeddings, databases, model weights, and caches remain ignored.

Inputs are normalized query text and existing L2-normalized 512-dimensional CLIP frame embeddings. Outputs are an ordered candidate list, a query match probability used only in this experiment, and diagnostic route evidence. CPU inference adds about 0.44 ms median and 0.96 ms p95 in the frozen run; its 29 float64 coefficients occupy 232 bytes. These probabilities do not transfer well enough to be product confidence.

The only semantic model is the existing `openai/clip-vit-base-patch32` at the pinned repository revision, sourced through Hugging Face Transformers under its published MIT license. CLIP preprocessing and CPU/GPU behavior are documented in `docs/learning/03-clip-and-multimodal-embeddings.md`; this experiment adds only NumPy CPU arithmetic. A cross-encoder could use image-query joint attention and a learned ranker could model richer interactions, but both need new evidence and the former costs substantially more CPU. Gradient boosting and a tiny MLP were rejected for this run because 66 calibration lists do not support their additional capacity.

The measured decision is outcome E: routing is dominant. Explicit correct-channel selection raises held-out R@5 to 95.24%, while the calibration-selected list scorer lowers it to 76.19% and the no-match rule falsely rejects 90.48% of positives. Production therefore remains raw five-second CLIP plus explicit BM25/RRF modes. See [results](../../evaluation/CANDIDATE_LIST_RANKING_RESULTS.md) and [failures](../../evaluation/CANDIDATE_LIST_RANKING_FAILURES.md).
