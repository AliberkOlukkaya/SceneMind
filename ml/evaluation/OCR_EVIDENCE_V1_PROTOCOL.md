# OCR Evidence V1 protocol

This experiment asks whether SceneMind's unchanged five-second, 480-pixel-wide sampled JPEGs can support searchable timestamped on-screen text. It is evaluation-only. Find Moments, Whisper, CLIP, RRF60, Q&A, AUTO routing and the disabled Ask Video feature remain unchanged.

## Predeclared candidates and selection

Exactly two local/free CPU candidates are eligible: RapidOCR 3.9.2 with ONNX Runtime 1.30 and EasyOCR 1.7.2 with its English model on CPU. Both are Apache-2.0. Tesseract is excluded because no local executable is installed. Candidate packages are evaluation dependencies and are not added to SceneMind's production requirements.

The development split contains one source each for presentation/slides, coding/tutorial, UI/demo and ordinary real-world video. Candidate selection orders engines by text-detection recall, normalized character accuracy, then p95 frame latency. The selected engine and all normalization, duplicate suppression, query and scoring rules are fixed before validation. Validation contains four source-disjoint videos in the same categories and runs once.

## Ground truth and metrics

Reviewers inspect the exact production-sampled JPEGs, not source time ranges. Each target records its exact visible string, first/last visible time in a one-second diagnostic review, matching five-second frame IDs, text role and frozen natural-language query. A target absent from all five-second JPEGs counts against sampling coverage and is excluded from recognition accuracy. Two validation frames per source are fixed for a post-run visual audit of every returned OCR line. This audit, rather than an incomplete list of target strings, supplies false-text scoring and may not change queries, targets or engine settings.

- **Sampling coverage:** target events with readable text in at least one production-sampled JPEG / readable target events in the one-second review.
- **Text detection recall:** verified-visible target strings localized by at least one OCR line / verified-visible targets.
- **Text accuracy:** `1 - total Levenshtein distance / total reference characters` after NFKC and whitespace normalization, clipped to `[0, 1]`.
- **False text rate:** unmatched OCR lines / all OCR lines on fully annotated validation frames.
- **Duplicate rate:** adjacent repeated evidence records / evidence records before suppression; the post-suppression residual duplicate rate is reported separately.
- **Timestamp error:** distance from the retrieved evidence start to the nearest frozen target interval, zero inside the interval.
- **Retrieval:** target-event Recall@1/3/5 using deterministic BM25 over normalized OCR evidence. End-to-end recall includes sampling misses; conditional-visible recall excludes them. Queries are frozen before OCR output is inspected.
- **Resources:** median and p95 CPU latency per frame, end-to-end OCR time, peak RSS delta and serialized JSON index bytes.

## Decision gate

Decision A requires validation sampling coverage and conditional-visible detection recall at least 80%, conditional-visible text accuracy at least 85%, end-to-end retrieval Recall@1 at least 75%, Recall@3 at least 90%, Recall@5 at least 95%, false text at most 10%, residual duplicates at most 5%, median timestamp error at most 5 seconds, p95 OCR latency at most 2 seconds/frame, peak RSS delta at most 1 GiB and index growth at most 50 KiB per source minute. If quality misses because recognition is poor, choose B; if five-second sampling coverage is below 80%, choose C; if quality passes but a resource gate fails, choose D; otherwise choose E. Only A advances to Multimodal Evidence Index V1.

The source manifest, exact sampled-frame annotations and queries are checksum-bound before the validation engine runs. No previous Q&A validation artifact may select OCR configuration or labels. Real measurements come only from this run; mocked tests verify contracts and never count as accuracy.
