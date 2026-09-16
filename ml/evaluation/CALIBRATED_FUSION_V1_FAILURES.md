# Calibrated Fusion V1 Failure Analysis

The experiment failed at the calibration generalization gate, before fusion. Raw magnitudes and
query-local margins looked useful inside the three calibration sources, but did not preserve that
advantage on two new sources. A fusion model built from these probabilities would encode an
unstable confidence scale.

## Generalization failures

| Modality | Selected signal | Validation failure |
| --- | --- | --- |
| Visual | absolute CLIP raw score | Raw-only AUC 0.6863 is below rank-only 0.7012; Brier rises from 0.0901 to 0.1002. Source AUC ranges from 0.5624 to 0.7912. |
| Speech | rank, margins, robust query-relative score | AUC falls from rank-only 0.7644 to 0.6454; Brier nearly doubles from 0.0873 to 0.1721. |

The Visual result indicates source-dependent CLIP score scale. The Speech result is stronger:
even though raw BM25 scores separate labels in isolation, the calibration mapping learned from
three other videos is miscalibrated on the validation sources. Direct comparison of CLIP and BM25
raw values remains invalid.

## Existing RRF60 misses

The validation baseline has eight positive Top-5 misses:

| Mechanism | Queries | Count |
| --- | --- | ---: |
| Irrelevant consensus | `hfd-d-s01`, `hfd-d-s03`, `hfd-d-s05`, `hfd-d-a01`, `hfd-h-s03` | 5 |
| Weak required-modality rank | `hfd-d-v04` | 1 |
| Cross-modal agreement | `hfd-d-m03` | 1 |
| Retrieval miss | `hfd-d-m04` | 1 |

No remaining failure was reclassified as temporal fragmentation, rank/raw-score disagreement,
unsupported capability, or annotation ambiguity in this run. These labels describe the frozen
baseline; no after-count exists because there is no candidate.

## Why the experiment stopped

The protocol required score calibration to generalize before testing RRF plus confidence,
confidence fusion, or a tiny learned fusion model. Both modalities failed both gate measures.
Continuing would select weights around validation failures and invite holdout tuning. The 34-query
protected holdout was therefore not opened for candidate evaluation, and no second run was due.

Agreement presence and temporal distance remain analysis features only. Two-modality presence
cannot be rewarded safely without stable underlying confidence, because prior evidence found it
in failures and controls. Production remains uncapped RRF60.
