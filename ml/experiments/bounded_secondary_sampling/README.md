# Bounded secondary sampling experiment

This experiment evaluates a coarse-to-fine visual path without changing production:

```text
frozen 5 s CLIP index -> top-K coarse timestamps -> bounded windows
-> cached 2 s JPEGs -> secondary CLIP -> YOLOX-Nano 640 -> fusion/abstention
```

The 97-query parent pool remains frozen: 66 calibration rows and 31 source-disjoint held-out rows. Window radius, top-K, temporal spacing, fusion weight and no-match threshold are selected using calibration rows only. Held-out rows are evaluated once with the selected settings.

## Reproduce

Source videos, secondary JPEGs, embeddings, detector weights and raw reports remain in ignored `data/` storage. Verify the existing benchmark media and pinned YOLOX-Nano 640 artifact, set `PYTHONPATH=backend`, then run:

```powershell
.venv/Scripts/python -m ml.experiments.bounded_secondary_sampling.run_experiment `
  --run-root data/bounded-secondary-run `
  --output data/bounded-secondary-run/report.json
```

Run `visible_diagnostic` with the selected policy and the maximum tested policy, then use `assemble_report` to create the committed artifact. Unit tests mock extraction and inference; only the explicit experiment runner performs real CLIP, FFmpeg and YOLOX work.

## Cache contract

The frame key is SHA-256 over policy version, video/document ID and millisecond timestamp. Policy changes use a new directory, providing deterministic invalidation. Atomic temporary-file replacement prevents partial JPEGs. Least-recently-used cleanup enforces a byte ceiling. The cache never contains source media, embeddings or secrets and is ignored by Git.

## Outcome

The experiment fails production gates and selects outcome C. Calibration chooses ±2 seconds, top-5 and no temporal spacing, but verified-visible held-out secondary-window recall is only 50% and secondary CLIP R@5 on those events is 0%. The calibration-only object weight is zero, and the no-match threshold causes 100% small-object false abstention. Production remains unchanged. See `ml/evaluation/BOUNDED_SECONDARY_RESULTS.md` and `ml/evaluation/BOUNDED_SECONDARY_FAILURES.md`.
