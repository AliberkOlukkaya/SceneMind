# OCR Evidence V1 results

## Decision

**B — detection works, but recognition and retrieval quality are insufficient.** The unchanged five-second sampler preserved 21 of 22 reviewed text events (95.45%), and RapidOCR localized 19 of those 21 visible targets (90.48%). It did not produce dependable general OCR search: frozen-label character accuracy was 84.14%, while a disclosed correction to one erroneous human label raises it to 84.84%; both miss the 85% gate. End-to-end Recall@1/3/5 was only 59.09%/63.64%/63.64% against gates of 75%/90%/95%. CPU p95 was also 2.304 seconds per frame, above the 2-second gate.

No production route, model, sampling rule, Find Moments behavior, Whisper path, CLIP/RRF60 ranking, Q&A path or Ask Video flag changed. The result does not authorize Multimodal Evidence Index V1.

## Frozen evaluation

The manifest was frozen before validation at SHA-256 `a16b95c58ef3539bb8679cd6b50b0e228018a993ffb07ec61a234cf38bdb7d14`. Four development and four source-disjoint validation videos cover presentation/slides, coding/tutorial, UI/demo and ordinary real-world footage. The 22 validation events and natural-language queries were written from manual inspection before the selected engine saw validation data.

The validation set contains 89 exact production-format JPEGs across 426.81 seconds of source video. Sampling uses the production FFmpeg rule: first frame, then the first frame at least five seconds after the previous selected timestamp, scaled to 480 pixels wide and encoded as JPEG quality 3. The one-second frames are diagnostic ground truth only and were never indexed.

One real sampling miss was retained: `DRIVERS OF LONG LOW VEHICLES phone before crossing` is fully readable at 28–29 seconds, but the 30-second production JPEG crops most of the sign. Its empty five-second frame list counts against sampling and end-to-end retrieval, and it is excluded from recognition metrics.

## Candidate selection on development data

Exactly two local/free candidates were predeclared. RapidOCR won by the fixed ordering of detection recall, character accuracy and p95 CPU latency.

| Engine | Visible detection recall | Character accuracy | Recall@1/3/5 | Median / p95 per frame | Peak RSS delta | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidOCR 3.9.2 + ONNX Runtime 1.30 | 95.00% | 91.28% | 95.00% / 100% / 100% | 744 / 2,310 ms | 293.3 MiB | 17,894 |
| EasyOCR 1.7.2 English CPU | 60.00% | 65.75% | 65.00% / 70.00% / 70.00% | 1,246 / 3,365 ms | 522.5 MiB | 10,127 |

These are development measurements, not validation claims. Only the selected RapidOCR configuration ran on the frozen validation set.

## Frozen validation measurements

| Metric | Result | Gate | Pass |
| --- | ---: | ---: | :---: |
| Five-second sampling coverage | 21/22 = 95.45% | >= 80% | Yes |
| Detection recall, conditional on visible target | 19/21 = 90.48% | >= 80% | Yes |
| Character accuracy, conditional on visible target | 84.14% frozen / 84.84% label-audited | >= 85% | **No** |
| End-to-end OCR retrieval Recall@1 | 13/22 = 59.09% | >= 75% | **No** |
| End-to-end OCR retrieval Recall@3 | 14/22 = 63.64% | >= 90% | **No** |
| End-to-end OCR retrieval Recall@5 | 14/22 = 63.64% | >= 95% | **No** |
| Retrieval Recall@1/3/5, conditional on visible | 61.90% / 66.67% / 66.67% | Diagnostic | — |
| False text rate | 14/160 = 8.75% | <= 10% | Yes |
| Raw duplicate rate | 40.00% | Report | — |
| Residual duplicate rate after suppression | 0.00% | <= 5% | Yes |
| Median timestamp error on useful Top-1 hits | 0.0 s | <= 5 s | Yes |
| CPU frame latency, median / p95 | 521 / 2,304 ms | p95 <= 2,000 ms | **No** |
| End-to-end OCR wall time | 62.87 s | Report | — |
| Peak RSS delta | 356.8 MiB | <= 1 GiB | Yes |
| Serialized evidence index | 20,671 bytes | Report | — |
| Index growth | 2,906 bytes/source-minute | <= 50 KiB/min | Yes |

Timestamp error is conditioned on the 13 successful Top-1 retrievals. It shows that retained evidence points at the right sampled frame; it does not hide the nine Top-1 misses.

The false-text result comes from manual review of every OCR line on two predeclared validation frames per source. Fourteen outputs represented non-text controls/icons or unsupported glyphs. Garbled recognition of real visible text was counted as text-backed, then penalized through character accuracy. The full 160-line audit is in `OCR_EVIDENCE_V1_FALSE_TEXT_AUDIT.json`.

That audit also found one ground-truth transcription error: the production JPEG shows `1st 11:56 Salisbury`, while the frozen manifest says `11:59 Salisbury`. The immutable primary result retains the frozen label. `OCR_EVIDENCE_V1_LABEL_AUDIT.json` records the correction and its separately computed 84.84% accuracy. Retrieval, detection and the Decision B outcome do not change.

## Category behavior

| Category | Sampling | Visible detection | Character accuracy | End-to-end Recall@1/3/5 |
| --- | ---: | ---: | ---: | ---: |
| Presentation | 5/5 | 5/5 | 94.30% | 5/5, 5/5, 5/5 |
| Coding/tutorial | 5/5 | 4/5 | 69.93% | 2/5, 3/5, 3/5 |
| UI/demo | 5/5 | 5/5 | 97.60% | 4/5, 4/5, 4/5 |
| Real world | 6/7 | 5/6 | 79.23% frozen / 83.58% label-audited | 2/7, 2/7, 2/7 |

Large slide and UI text generalized. Dense 480-pixel-wide wiki source, distant/angled station text and short sign appearances did not. Exact text could still fail retrieval: `DUNBRIDGE Please drive carefully`, `MILL ARMS`, `Salisbury`, `A unified interface for LLMs` and `Publish changes...` were recognized but did not rank within the frozen target window at Top-5. This is an OCR-evidence ranking failure in addition to recognition loss.

## Resource interpretation

The run used Windows 11, Python 3.13.5 and an Intel Core i5-11400H CPU with 6 cores/12 logical processors and 16 GB system RAM. RapidOCR processed 89 frames in 62.87 seconds, about 0.706 seconds of OCR wall time per indexed frame and 8.84 seconds per source-video minute. Coding frames with dense small text drove the per-source median to 2.370 seconds and aggregate p95 over the gate. OCR index storage itself was negligible; model memory and dense-frame CPU time were the material costs.

## Recommendation

Keep OCR evaluation-only and preserve the production system unchanged. The dominant observed blocker is recognition/retrieval on dense small and environmental text, not five-second sampling overall. Denser global sampling would increase cost and storage while fixing only the one observed transient-sign miss; it would not repair the eight other Top-1 failures. Do not start Multimodal Evidence Index V1 from this result. Any future OCR milestone needs new source-disjoint evidence and should test a bounded high-resolution OCR branch plus an OCR-specific retrieval formulation, with the same CPU gate. This frozen validation set is now observation-only and unavailable for tuning.

Machine-readable metrics are in `reports/ocr-evidence-v1.json`; the protocol, manifest checksum and failure ledger preserve how every number was obtained.
