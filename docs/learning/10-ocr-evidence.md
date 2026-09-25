# OCR evidence from sampled video frames

Optical character recognition turns pixels that look like writing into timestamped text. In SceneMind's evaluated pipeline, the input is the existing 480-pixel-wide JPEG sampled every five seconds. The output is a sequence of text lines with bounding boxes and confidence values, grouped into a timestamped evidence record that can be searched with BM25.

The experiment has five deterministic steps:

1. Read the existing sampled JPEG; do not decode a different frame for OCR.
2. Detect text regions and recognize their characters with the selected local OCR engine.
3. Normalize Unicode with NFKC and collapse whitespace while preserving useful command/error punctuation.
4. Merge only adjacent near-identical frame records (similarity at least 0.88 and gap at most 5.25 seconds), retaining every contributing frame ID and the evidence time span.
5. Search the normalized evidence with the existing BM25 primitive and return timestamped records.

OCR has two independent opportunities to fail. Sampling can omit a short-lived text event entirely. When the correct JPEG exists, detection can miss its region, recognition can corrupt its characters, or retrieval can rank the recognized evidence poorly. OCR Evidence V1 reports sampling coverage separately and measures recognition only on manually verified visible targets.

## Evaluated engines

RapidOCR 3.9.2 with ONNX Runtime 1.30 was the selected CPU engine. The evaluated defaults loaded the bundled PP-OCRv6 small detection and recognition ONNX models plus the mobile direction classifier. Inputs are JPEG paths; outputs are polygons, strings and confidence scores. RapidOCR and its code are Apache-2.0 and documented at <https://github.com/RapidAI/RapidOCR>.

EasyOCR 1.7.2 with the English model and `gpu=False` was the only alternative. It accepts RGB arrays and returns bounding quadrilaterals, text and confidence. Its code is Apache-2.0 and documented at <https://github.com/JaidedAI/EasyOCR>. Its locally downloaded weights live under ignored `data/models/easyocr`; models and weights are never committed.

Both can use accelerated runtimes when configured, but this milestone measured CPU only because SceneMind's resource gate targets ordinary local hardware. They are optional evaluation dependencies and are absent from production requirements. Tesseract was excluded before comparison because no executable was installed. A VLM, fine-tuning and cloud OCR were outside scope.

## Persistence and invalidation

`backend/app/ocr_evidence.py` defines the isolated evidence contract. Saved evidence includes the engine revision, configuration and a fingerprint of frame IDs, timestamps and normalized text. Loading with a different fingerprint raises a rebuild error. Writes are atomic. Nothing imports this module into production routes or workers.

The frozen validation showed why the distinction matters. The sampler retained 95.45% of reviewed events, while conditional-visible character accuracy reached 84.14% on frozen labels and 84.84% after one disclosed label correction; end-to-end Recall@5 was only 63.64%. Large slides and UI labels worked well; dense source text and environmental signs did not. This branch therefore remains an experiment rather than product evidence.
