# Calibrated Raw-Score + Agreement Fusion V1

This evaluation-only experiment asks whether CLIP and BM25 raw scores can support a
confidence-aware replacement for production's uncapped RRF60. It does not import into
`app`, change an endpoint, or create a production artifact.

## Data and isolation

- Calibration: 72 frozen queries from three sources in the path-aware no-match suite.
- Source-disjoint validation: 32 frozen queries from two Hybrid Fusion development sources.
- Protected evaluation: 34 Hybrid Fusion Holdout V1 queries plus Acceptance V2. Their source
  hashes are checked for leakage, but no candidate evaluation reads their results.
- The calibration reconstruction matches all 72 historical Visual, Speech, and Hybrid Top-5
  outputs. A `1e-12` score tolerance permits JSON floating-point roundoff only; timestamps and
  rank order must match exactly.

For each retriever, the runner compares four small logistic calibrators selected by
leave-one-source-out calibration Brier score: rank only, raw score only, query-relative rank
and margins, and their bounded combination. Features are standardized using calibration-only
statistics. The declared generalization gate requires the selected calibrator to be no worse
than a rank-only reference on both AUC and Brier for both modalities.

## Outcome

The gate failed for both modalities. Visual selected raw score only, then lost 0.0149 AUC and
added 0.0101 Brier loss on source-disjoint validation. Speech selected query-relative features,
then lost 0.1190 AUC and added 0.0848 Brier loss. The runner therefore stops before constructing
or selecting a fusion formula. Agreement and temporal-distance features are exposed only for
analysis; they receive no reward or weight.

The saved calibration artifact is diagnostic and explicitly has `production_eligible: false`.
Production remains uncapped RRF60. This is decision **C — calibration does not generalize**.

## Reproduce

The ignored reconstruction cache contains local retrieval data and is never committed. With
the existing local evaluation assets present, run:

```powershell
.venv/Scripts/python -m ml.experiments.calibrated_fusion_v1.run_experiment
.venv/Scripts/python -m pytest tests/test_calibrated_fusion_v1.py
```

Outputs are `ml/evaluation/reports/calibrated-fusion-v1.json` and the rejected diagnostic
artifact `calibration_artifact_v1.json`. See
`ml/evaluation/CALIBRATED_FUSION_V1_RESULTS.md` and
`ml/evaluation/CALIBRATED_FUSION_V1_FAILURES.md`.
