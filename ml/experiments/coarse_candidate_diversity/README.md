# Coarse candidate diversity experiment

This isolated experiment asks whether cheap, deterministic selection can turn a larger CLIP result pool into five useful, temporally distinct moments. It does not change production search or fit an acceptance threshold.

The runner hash-locks the existing Natural V2 and visibility evidence, loads the frozen 66/31 calibration/held-out query split, and performs real CLIP/FAISS retrieval over both the existing five-second frames and the previously extracted two-second frames. Policy selection uses calibration rows and the two calibration-only reviewed small-object events. Held-out results are calculated only after the winner is fixed.

Compared methods are raw five- and two-second CLIP, temporal NMS, embedding MMR, temporal plus embedding diversity, CLIP-change scene grouping, and a bounded multi-scale diagnostic that retains five-second frames and adds high-change two-second frames. The final result count is always five. Timestamps within 50 ms of a radius boundary are treated as equal to account for measured VFR sampling jitter.

Run from the repository root:

```powershell
.venv\Scripts\python -m ml.experiments.coarse_candidate_diversity.run_experiment `
  --output data\coarse-candidate-diversity-final\report.json `
  --run-root data\coarse-candidate-diversity
.venv\Scripts\python -m ml.experiments.coarse_candidate_diversity.assemble_report `
  data\coarse-candidate-diversity-final\report.json
```

Local JPEGs, vectors and raw reports remain ignored. The committed machine report contains input hashes, per-query raw top-20 timestamps, gaps, score spread, duplicate statistics, calibration trials, selected intervals, resources, failures and the final decision.

The calibration-selected multi-scale policy fails promotion. Production remains the five-second raw CLIP path, and a second frozen run is not warranted.
