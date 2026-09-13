# Object detector branch results

## Decision

YOLOX-Nano `0.1.1rc0` was rejected for production. The query-gated ONNX CPU branch is fast and small, but its calibration-only presence threshold abstains on every frozen small-object positive. Production remains the existing CLIP/BM25/RRF system. No second frozen run, reranking trial, or relationship proof of concept was warranted.

The committed machine report is [`reports/object-detector-v1.json`](reports/object-detector-v1.json). Its model-bound, non-promotable calibration artifact is [`../experiments/object_detector_branch/calibration_v1.json`](../experiments/object_detector_branch/calibration_v1.json). The 3,659,407-byte ONNX binary is ignored and identified by SHA-256 `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`.

## Protocol

- Reused the exact CLIP top-five timestamps from the frozen lightweight pair-scorer report; CLIP retrieval was not replaced.
- Verified all source-media checksums and all three parent manifest/report hashes before inference.
- Ran the official Apache-2.0 YOLOX-Nano ONNX release on 55 unique candidate frames with ONNX Runtime CPUExecutionProvider, four intra-op threads, 416 x 416 letterbox input, 0.01 evidence floor and class-aware NMS at 0.45.
- Mapped query terms to COCO labels with versioned explicit aliases. Unsupported attributes and relations remain visible in each row.
- Fitted the presence threshold on 66 calibration queries only. The selected `0.8569136262` threshold is just above the third-highest negative-query maximum, allowing 2/22 detector-triggered calibration negatives (9.1%).
- Applied the frozen threshold once to 31 held-out visual rows. Held-out results did not alter mapping, preprocessing, score floor, NMS or threshold.

## Main result

| Held-out slice/system | R@1 | R@3 | R@5 | MRR@5 | Negative FAR@5 | Positive false abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows, CLIP top five | 71.4% | 76.2% | 85.7% | 0.805 | 100% | 0% |
| All rows, query-gated presence | 47.6% | 57.1% | 61.9% | 0.567 | 50% | 23.8% |
| Detector-triggered, CLIP top five | 88.9% | 94.4% | 100% | 1.000 | 100% | 0% |
| Detector-triggered, presence | 44.4% | 44.4% | 44.4% | 0.444 | 37.5% | 55.6% |
| SMALL_OBJECT, CLIP top five | 83.3% | 83.3% | 100% | 1.000 | n/a | 0% |
| SMALL_OBJECT, presence | 0% | 0% | 0% | 0.000 | n/a | 100% |
| Small-object-class negatives, presence | n/a | n/a | n/a | n/a | 0% (0/2) | n/a |

Small-object recall averages interval coverage, so the bicycle query with two relevant intervals can contribute 0.5 at a given K. The small-object negative FAR is measured on the two held-out negative queries routed to `bicycle` or `sports ball`; the positive slice itself contains no negatives.

## Required slice breakdown at K=5

| Slice | Positives | Negatives | CLIP R@5 / MRR | Presence R@5 / MRR | CLIP FAR | Presence FAR | Presence false abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OBJECT | 6 | 0 | 100% / 1.000 | 50% / 0.500 | n/a | n/a | 50% |
| SMALL_OBJECT | 3 | 0 | 100% / 1.000 | 0% / 0.000 | n/a | n/a | 100% |
| RELATIONSHIP | 4 | 2 | 100% / 1.000 | 50% / 0.500 | 100% | 0% | 50% |
| COMPOSITIONAL | 4 | 0 | 100% / 1.000 | 50% / 0.500 | n/a | n/a | 50% |
| SCENE | 3 | 0 | 100% / 1.000 | 100% / 1.000 | n/a | n/a | 0% |
| ACTION_TEMPORAL | 3 | 0 | 100% / 1.000 | 100% / 1.000 | n/a | n/a | 0% |
| NEGATIVE | 0 | 10 | n/a | n/a | 100% | 50% | n/a |
| Non-detector | 12 | 2 | 75% / 0.658 | 75% / 0.658 | 100% | 100% | 0% |

The gate preserved non-triggered rows exactly, which confirms routing isolation. It did not make the triggered subset reliable: generic `person` evidence accepted three semantic negative queries, while the high calibration threshold removed five of nine triggered held-out positives.

## Resource measurements

The isolated subprocess loaded the model in 205 ms. A cold sequential top-five pass took 107 ms. After one warmup, 20 top-five passes measured a **90.5 ms median** and **99.6 ms p95**. Added peak RSS was **60,837,888 bytes (58.0 MiB)**. This passes the preferred 250 ms latency and 500 MB memory targets. The model is static batch one, so a top-five pass consists of five bounded calls.

## Gate

| Requirement | Result | Pass |
| --- | ---: | :---: |
| Small-object positive false abstention <=20% | 100% | No |
| Small-object negative FAR <=10% | 0% | Yes |
| Raw candidate R@5 loss <=2 percentage points | 0 pp | Yes |
| Warm top-five latency <=400 ms | 90.5 ms | Yes |
| Added memory <=750 MB | 60.8 MB | Yes |

The quality gate failed. The calibration presence trial was also not promising: detector-triggered calibration R@5 fell from 87.0% to 16.7% and positive false abstention reached 77.8% at 9.1% FAR. The predeclared sequence therefore stopped before fitting combined CLIP/detector weights. A box-only relationship heuristic was also skipped because it cannot repair missing object detections and the exact held-out holding frame does not show the ball.

## Next direction

Choose outcome **C: detector misses small objects**. Before another production proposal, add source-disjoint, frame-visible small-object examples and evaluate a higher-resolution detector on the same CLIP-candidate boundary. The next test must distinguish model-resolution misses from five-second sampling misses; higher detector resolution cannot recover an object absent from the sampled JPEG.
