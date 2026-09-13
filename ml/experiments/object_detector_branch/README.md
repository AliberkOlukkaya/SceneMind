# Object detector branch experiment

This bounded experiment asks whether object evidence can repair SceneMind's calibrated small-object failures without replacing CLIP. The proposed path is:

```text
query -> existing CLIP top five -> transparent object-query gate
      -> YOLOX-Nano on candidate frames -> structured detections
      -> presence decision and optional calibration-only reranking
```

The frozen preselection evidence is in [`failure_inventory_v1.json`](failure_inventory_v1.json) and [`FAILURE_INVENTORY.md`](FAILURE_INVENTORY.md). Detector research and the closed-set decision are in [`MODEL_SHORTLIST.md`](MODEL_SHORTLIST.md).

Production code remains unchanged unless all quality and resource gates pass and a second frozen run confirms the result. Model files, extracted frames and benchmark run databases remain under ignored `data/` paths.

## Run

Download the official release artifact to ignored storage, verify the SHA-256 shown in `MODEL_SHORTLIST.md`, then run:

```powershell
.venv/Scripts/python -m ml.experiments.object_detector_branch.evaluate
.venv/Scripts/python -m ml.experiments.object_detector_branch.validate
```

The runner verifies the pinned model byte count and checksum, all source-media checksums, the frozen benchmark/calibration hashes, and the parent CLIP report hash. It writes an ignored report and calibration artifact under `data/` by default. Unit tests mock inference; real model execution occurs only through the command above.

## Outcome

The frozen run rejected YOLOX-Nano for production. It passed resources at 90.5 ms warm top-five median, 99.6 ms p95 and 58.0 MiB added peak RSS, but the calibration-only threshold produced 100% small-object positive false abstention. Reranking and box-based relationship work were skipped because calibration presence evidence failed its gate. See [`OBJECT_DETECTOR_RESULTS.md`](../../evaluation/OBJECT_DETECTOR_RESULTS.md) and [`OBJECT_DETECTOR_FAILURES.md`](../../evaluation/OBJECT_DETECTOR_FAILURES.md).
