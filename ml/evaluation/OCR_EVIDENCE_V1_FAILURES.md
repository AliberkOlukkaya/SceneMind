# OCR Evidence V1 failures

## Gate failures

OCR Evidence V1 ends at Decision B. Three quality/resource gates failed on the frozen source-disjoint validation:

- Conditional-visible character accuracy was 84.14% on the frozen labels and 84.84% after a disclosed correction to one erroneous human label, versus 85% required.
- End-to-end OCR retrieval Recall@1/3/5 was 59.09%/63.64%/63.64% versus 75%/90%/95% required.
- CPU p95 latency was 2.304 seconds/frame versus at most 2 seconds.

The five-second sampler passed at 95.45% coverage. Detection also passed at 90.48%, so treating every failure as a sampling miss would contradict the reviewed JPEGs.

## Observed failure classes

### Dense small text loses characters

The coding/tutorial source was scaled to the production width of 480 pixels. RapidOCR detected four of five targets, but category character accuracy was only 69.93%. For example, the visible sentence beginning `The following is a full list of all editors who have joined` was rendered as `The fallowng ss a fall list o all editors who have jacned...`; its similarity fell below the fixed detection threshold. Dense coding frames also had a 2.370-second median OCR time.

### Environmental text is fragile

Real-world character accuracy was 79.23% on frozen labels and 83.58% after the label audit; detection was 5/6 among sampled-visible targets. Perspective, distance, motion and small lettering damaged `Mottisfont & Dunbridge`; the best recognized span was `Motislantsm DBGO`. A post-run visual audit also found that the departure-board reference was transcribed incorrectly before freezing: the JPEG and OCR both say `1st 11:56 Salisbury`, not the frozen `11:59 Salisbury`. The correction is preserved separately and does not change the failed gate or decision.

### Exact recognized text can rank outside Top-5

Five visible strings had strong or exact recognition but no relevant evidence in the frozen BM25 Top-5: `Publish changes...`, `A unified interface for LLMs`, `DUNBRIDGE Please drive carefully`, `MILL ARMS` and `Salisbury`. Question framing and repeated screen text can favor another evidence group. This explains why conditional-visible Recall@5 was only 66.67% even though detection was 90.48%.

### One genuine five-second sampling miss

The long-vehicle warning is fully readable in one-second frames 29 and 30 (28 and 29 seconds), then mostly cropped in the production frame at 30 seconds. It correctly counts as absent from five-second evidence. No OCR output from the partial production view was treated as proof of visibility.

### Non-text controls become text

The predeclared eight-frame audit found 14 false lines among 160 outputs (8.75%). Typical cases were browser control glyphs, toolbar icons and isolated symbols. This passed the 10% gate but leaves noise that can pollute lexical retrieval.

## What the result rules out

- Global denser sampling is not justified by this set: it addresses 1/22 events, while recognition or retrieval misses affect substantially more.
- RapidOCR should not be added to ingestion or Find Moments from this milestone.
- EasyOCR is not a fallback candidate: on development it reached only 60% detection, 65.75% character accuracy and 70% Recall@5 while using more memory and time.
- The observed validation must not be used to adjust normalization, duplicate thresholds, BM25 query handling or OCR parameters.
- Ask Video remains disabled; OCR evidence was never connected to it.

The failed first validation process was an instrumentation failure, not a model run: RapidOCR returned `None` collections for an empty-text frame, the adapter raised before writing raw output or a report, and the adapter was corrected to represent the result as an empty list. The manifest and inference configuration remained checksum-identical. The subsequent complete run is the only scored validation.
