# OCR Evidence V1 experiment

This evaluation-only package extracts the same 480-pixel-wide JPEGs as production at the unchanged five-second interval. One-second frames exist only for manual sampling-coverage review. Local source media, frames, model weights and raw outputs stay under ignored `data/ocr-evidence-v1/`.

The frozen protocol is in `ml/evaluation/OCR_EVIDENCE_V1_PROTOCOL.md`. `prepare.py` reproduces sampling and contact sheets. `run_experiment.py` executes the predeclared OCR candidates or scores the selected engine against a checksum-bound manifest. `apply_false_text_audit.py` verifies that the manual audit covers every OCR line on the predeclared frames before attaching its result to the report. Nothing here is connected to production routes or indexes.

Install the optional evaluation environment with `pip install -r ml/experiments/ocr_evidence_v1/requirements.txt`. With `backend` and the repository root on `PYTHONPATH`, run candidates as modules, for example:

```powershell
$env:PYTHONPATH = "backend;."
python -m ml.experiments.ocr_evidence_v1.run_experiment --engine rapidocr --split development
python -m ml.experiments.ocr_evidence_v1.run_experiment --engine easyocr --split development
python -m ml.experiments.ocr_evidence_v1.run_experiment --engine rapidocr --split validation
python -m ml.experiments.ocr_evidence_v1.apply_false_text_audit
```

Validation refuses to run when the manifest bytes differ from `ocr_evidence_v1_manifest.sha256`. Local licensed media and generated frames are intentionally absent from Git; the manifest preserves source pages, media hashes, exact frame timestamps, labels and queries.
