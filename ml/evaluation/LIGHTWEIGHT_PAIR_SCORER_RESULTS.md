# Lightweight pair scorer results

No tested method passes every product gate. UForm3-small ONNX is genuinely lightweight and stays under the CPU latency and memory ceilings, but its calibration-only no-match threshold abstains on 47.6% of held-out positives. The tiny learned scorer is far cheaper and worse at 71.4%. Production remains unchanged and no second run was performed.

## Frozen evidence

The new development expansion adds two independently sourced Commons files and 20 pre-inference annotations: 10 positive and 10 hard negative. It targets a phone, keys, bag and cup; holding, beside, behind and transfer relationships; and plausible absent actions. Together with Natural V2 calibration and the earlier verifier expansion, training/calibration has 66 visual queries (35 positive, 31 negative) across seven source groups. Final evaluation uses the unchanged 31-query, three-source Natural V2 held-out split. Media provenance, attribution and hashes are in `calibration_v2.json` and the experiment README.

## Frozen held-out comparison

R@K and MRR apply to the 21 held-out positive visual-path queries. FAR uses ten negatives. Positive false abstention (PFA) is the share of positives returning no candidate after thresholding. Raw systems deliberately return candidates for every negative, so their FAR is 100%.

| System | R@1 | R@3 | R@5 | MRR@5 | FAR@5 | PFA | Added median | Peak/additional memory | Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw CLIP baseline | 71.4% | 76.2% | 85.7% | 0.805 | 100% | 0% | 0 | 0 | existing |
| Current scalar CLIP calibration | 31.0% | 33.3% | 38.1% | 0.343 | 10% | 52.4% | negligible | negligible | existing |
| BLIP ITM rerank, unthresholded | 61.9% | 81.0% | 85.7% | 0.763 | 100% | 0% | 3.352 s | 1.396 GB peak delta | 2.050 s |
| BLIP ITM calibrated | 26.2% | 26.2% | 26.2% | 0.286 | 10% | 52.4% | 3.352 s | 1.396 GB peak delta | 2.050 s |
| UForm ONNX rerank, unthresholded | 71.4% | 73.8% | 85.7% | 0.793 | 100% | 0% | 202.1 ms | 163.1 MB isolated peak delta | 0.839 s |
| **UForm ONNX calibrated** | **50.0%** | **50.0%** | **50.0%** | **0.524** | **0%** | **47.6%** | **202.1 ms** | **163.1 MB isolated peak delta** | **0.839 s** |
| Logistic rerank, unthresholded | 64.3% | 83.3% | 85.7% | 0.790 | 100% | 0% | 0.066 ms | negligible | none |
| **Logistic calibrated** | **16.7%** | **19.0%** | **19.0%** | **0.190** | **0%** | **71.4%** | **0.066 ms** | **negligible** | **none** |

UForm beats current scalar calibration on R@1/3/5, MRR and FAR, and beats calibrated BLIP on the same quality measures while using about one seventh of BLIP's peak memory delta and one sixteenth of its latency. It still misses the required PFA by 27.6 percentage points. Its unthresholded ordering does not beat raw CLIP: R@1 ties, R@3 falls 2.4 points, R@5 ties, and MRR falls 0.012. The learned scorer improves unthresholded R@3 by 7.1 points but lowers R@1 and MRR; thresholding does not transfer.

| Held-out slice at K=5 | Raw CLIP R@5 | Scalar CLIP | UForm calibrated | Logistic calibrated |
| --- | ---: | ---: | ---: | ---: |
| Small object (3) | 100% | 0% | 0% | 0% |
| Relationship/compositional (4) | 100% | 75.0% | 62.5% | 50.0% |
| Temporal/action (3) | 100% | 66.7% | 100% | 0% |
| Object (6) | 100% | 33.3% | 50.0% | 16.7% |
| Scene (3) | 100% | 0% | 66.7% | 0% |

Small-object outcomes are the clearest regression: the raw candidate set contains all three answers, while every calibrated system abstains on all three. UForm preserves all action cases and improves scene acceptance over the scalar baseline. Its relationship PFA is 25%, still above the overall 20% gate.

## Runtime

Environment: Windows 11 build 26200; Intel64 Family 6 Model 141, six physical/twelve logical CPUs; Python 3.13.5; ONNX Runtime 1.30.0; UForm 3.1.3; PyTorch 2.14.0 CPU; four ONNX intra-op threads and one inter-op thread. Timings include loading five JPEGs, preprocessing, one text encoding, batched five-image encoding and cosine scoring.

Across 97 real SceneMind queries, UForm's warm top-five median is 202.1 ms, p95 229.1 ms, or 40.4 ms per pair. Its first measured top-five call is 156.8 ms; model/session load is 838.9 ms. With CLIP already resident, loading UForm adds 97.6 MB working set. A fresh-process load plus three top-five warmups peaks at 216.4 MB total, 163.1 MB above baseline.

| ONNX intra-op threads | Median top-five | p95 | Load |
| ---: | ---: | ---: | ---: |
| 1 | 385.2 ms | 415.4 ms | 0.824 s |
| 2 | 233.3 ms | 248.8 ms | 0.474 s |
| **4** | **172.4 ms** | **195.7 ms** | **0.475 s** |
| 8 | 233.7 ms | 272.7 ms | 0.509 s |
| runtime default | 187.0 ms | 226.2 ms | 0.543 s |

The thread sweep uses one fixed real five-frame batch for 20 repetitions after three warmups; the full evaluation spans varied images and text, which explains its higher 202.1 ms median. K=3 was not advanced: raw R@3 is 76.2%, 9.5 points below R@5 and therefore violates the two-point candidate-recall-loss gate. OpenVINO, PyTorch eager and further INT8 work were not run after quality failed. The shipped ONNX path was already below the resource gates; runtime optimization cannot reduce PFA.

The logistic scorer has eight trainable parameters, 300 candidate examples (94 relevant), 4.0 ms training time, 0.066 ms median and 0.084 ms p95 inference over five candidates in 1,000 repeats. Its fixed L2 strength is 1.0. The artifact stores all seven means, scales and weights, its threshold, three dataset hashes, model revision, seed, runtime and code version.

## Gates and decision

| Gate | Limit | UForm | Logistic |
| --- | ---: | ---: | ---: |
| Warm median | <=250 ms | 202.1 ms pass | 0.066 ms pass |
| Added peak memory | <=750 MB absolute; <=500 MB preferred | 163.1 MB pass | negligible pass |
| Negative FAR | <=10% | 0% pass | 0% pass |
| Positive false abstention | <=20% | 47.6% **fail** | 71.4% **fail** |
| Candidate R@5 loss | <=2 points | 0 points pass | 0 points pass |

Outcome **D** applies: no lightweight visual verifier works under all gates. Stop searching this dual-encoder/score-threshold architecture family for now. Build exactly one next capability: a small-object/object-detector branch evaluated over explicit object-bearing queries. It targets the only held-out slice where raw CLIP finds every answer and all three calibrated approaches return none. Relationship and temporal work remain evidence to revisit after that bounded branch.

Machine-readable evidence: `reports/lightweight-pair-scorer-v1.json`, `reports/lightweight-learned-scorer-v1.json`, `reports/lightweight-uform-scorer-v1.json`, and `reports/uform-runtime-options-v1.json`.
