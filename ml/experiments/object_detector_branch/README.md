# Object detector branch experiment

This bounded experiment asks whether object evidence can repair SceneMind's calibrated small-object failures without replacing CLIP. The proposed path is:

```text
query -> existing CLIP top five -> transparent object-query gate
      -> YOLOX-Nano on candidate frames -> structured detections
      -> presence decision and optional calibration-only reranking
```

The frozen preselection evidence is in [`failure_inventory_v1.json`](failure_inventory_v1.json) and [`FAILURE_INVENTORY.md`](FAILURE_INVENTORY.md). Detector research and the closed-set decision are in [`MODEL_SHORTLIST.md`](MODEL_SHORTLIST.md).

Production code remains unchanged unless all quality and resource gates pass and a second frozen run confirms the result. Model files, extracted frames and benchmark run databases remain under ignored `data/` paths.
