# Small-object sampling and detector-resolution ablation

This isolated experiment separates missing visual evidence from detector failure. It never changes the API, production five-second sampler, index, routes or UI.

`visibility_v1.json` freezes four target events across three new Wikimedia Commons sources. Calibration and held-out source groups are disjoint. Every 1 s JPEG in each event window was manually assigned one of four statuses: visible, not visible, partially cropped, or too small to judge. The per-interval A/B/C/D assessment is derived from those exact frames; timestamp overlap is insufficient.

Source videos, JPEGs, model weights and raw measurements stay under ignored `data/`. To reproduce the run:

1. Download each manifest `download_url` to `data/small-object-ablation/source/<filename>`.
2. Run `python -m ml.experiments.small_object_ablation.prepare` to verify checksums and extract exact 5 s, 2 s and 1 s JPEGs.
3. Run `run_sampling` with `PYTHONPATH=backend` to measure CLIP indexing/search.
4. Run `run_detector` for `yolox-416`, `yolox-640`, `yolox-768` and `rtdetr-r18-640` in separate processes.
5. Inspect the best-box overlays and freeze target mismatches in `detection_review_v1.json` before gate evaluation.
6. Use `assemble_report` to generate the machine report.

YOLOX-Nano uses the official `0.1.1rc0` checkpoint at commit `e1052df71842031413f6030723c3607b839c80ce`. Its source checkpoint is 7,694,953 bytes with SHA-256 `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b`. The 640 and 768 static ONNX files use the same weights and graph, exported with only the test input size changed. RT-DETR-R18 uses the Apache-2.0 `PekingU/rtdetr_r18vd` safetensors artifact at revision `6401be7fee8b49ee00b42fcc4e0064bba8061777` through Transformers on CPU.

The committed report is `ml/evaluation/reports/small-object-ablation-v1.json`; the readable analysis is `ml/evaluation/SMALL_OBJECT_ABLATION_RESULTS.md`.
