# Small-object sampling and detector-resolution ablation

## Decision

Choose outcome **C: sampling and detector capacity both matter**, with sampling the dominant failure. Keep production unchanged. The next bounded product experiment should use a query-gated, cached **2-second secondary sample set with YOLOX-Nano 640** over bounded temporal windows. Do not promote RT-DETR-R18 as an all-frame CPU detector.

The machine report is [`reports/small-object-ablation-v1.json`](reports/small-object-ablation-v1.json). The visibility manifest and target-box review were frozen separately. Source media, sampled JPEGs, model files and raw timing runs remain ignored.

## Evidence set

Three checksum-verified Wikimedia Commons videos add bicycle, bottle and sports-ball evidence from source groups absent from the parent benchmark. The two calibration groups and one held-out group are disjoint; two held-out events share only the held-out Throwball source. No held-out label changed model choice, preprocessing, thresholds or gates.

A human reviewed 50 exact 1-second sampled JPEGs before detector inference: 18 visible, 27 not visible, 3 partially cropped and 2 too small to judge. Each 5/2/1-second assessment is derived from the exact JPEG membership. An interval timestamp never counts as visible evidence by itself.

## Sampling ablation

| Interval | Events with visible/partial evidence | Evidence recall | Frames | JPEG bytes | Embedding bytes | Extract time | CLIP image-index time | Combined ingest work | Warm CLIP search median / p95 | CLIP candidate R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 1/4 | **25%** | 47 | 1,008,678 | 96,256 | 5.559 s | 2.440 s | 8.000 s | 10.02 / 11.37 ms | 25% |
| 2 s | 4/4 | **100%** | 116 | 2,530,178 | 237,568 | 6.704 s | 4.375 s | 11.079 s | 10.30 / 11.42 ms | 25% |
| 1 s | 4/4 | **100%** | 230 | 5,078,291 | 471,040 | 7.550 s | 8.757 s | 16.308 s | 10.53 / 11.33 ms | 50% |

Relative to 5 seconds, 2 seconds creates 2.47x frames, 2.51x JPEG storage and 2.47x vector storage; measured extraction plus CLIP encoding grows 1.38x. One second creates 4.89x frames, 5.03x JPEG storage and 4.89x vector storage; combined work grows 2.04x. Exact FAISS query latency stays near 10 ms at this scale.

The retrieval result is distinct from evidence recall. Two-second sampling retained visible evidence for every event, but a fixed CLIP top five found a verified relevant frame for only one. One-second sampling reached two of four. Denser sampling adds distractors, so it cannot be treated as an automatic CLIP recall improvement.

## Detector ablation on the same visible frames

The denominator is exactly 18 human-verified visible frames. The best target-class box was overlaid and reviewed; boxes on another ball or on the return machine do not count. The main comparison uses the predeclared 0.01 evidence floor rather than a held-out-fitted threshold.

| Detector | Reviewed conditional recall | Calibration | Held-out | Mean confidence on matches | Mean matched box area | Median / p95 per frame | Added peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOX-Nano 416 | 12/18 = **66.7%** | 25.0% | 78.6% | 0.457 | 0.80% | 13.0 / 16.9 ms | 57.2 MiB |
| YOLOX-Nano 640 | 15/18 = **83.3%** | 50.0% | 92.9% | 0.607 | 0.78% | 24.5 / 32.5 ms | 81.9 MiB |
| YOLOX-Nano 768 | 15/18 = **83.3%** | 50.0% | 92.9% | 0.581 | 0.85% | 34.9 / 47.7 ms | 111.3 MiB |
| RT-DETR-R18 640 | 16/18 = **88.9%** | 50.0% | 100% | 0.721 | 0.76% | 350.7 / 362.8 ms | 552.5 MiB |

At a stricter fixed confidence of 0.25, recall is 44.4%, 61.1%, 61.1% and 77.8%, respectively. This exposes the low-confidence edge cases without fitting a cutoff on held-out data.

Resolution materially helps Nano: 640 adds 16.7 percentage points over 416 and passes the predeclared >=80% recall, >=10-point gain, <=100 ms p95 and <=500 MiB resource gates. Moving from 640 to 768 adds no recall and costs 47% more p95 latency and 36% more memory. RT-DETR adds only 5.6 points over Nano 640, takes about 11 times its p95 latency and exceeds the 500 MiB preferred memory gate.

Both passing Nano configurations were run a second time on the frozen frames. Detection rows were identical. Nano 640's second median/p95 were 29.6/33.7 ms with 82.4 MiB added peak RSS; Nano 768's were 35.8/48.2 ms with 113.0 MiB. No production promotion was made.

## Architecture recommendation

Five-second sampling loses three of four target events before detection, a 75-point evidence deficit. Nano 416 then misses or mismatches six of 18 visible targets. Sampling is therefore the larger measured bottleneck, while the 16.7-point gain at 640 shows detector resolution still matters.

For a future isolated branch, recognize detector-addressable small-object queries, generate and cache a 2-second secondary sample set, and run Nano 640 only over bounded temporal windows or as background work. Keep CLIP as the broad candidate system and measure whether a wider candidate budget or window proposal can reach the retained evidence; the current fixed top five does not. A full-video synchronous CPU rescan is not an interactive design. RT-DETR-R18 is viable for offline diagnostics, but its marginal recall gain does not justify its CPU latency and memory for SceneMind's local-first path.

This is a diagnostic set of four events and 18 visible frames, not a population estimate. It supports the next architecture experiment and rejects a stronger all-frame detector; it does not by itself justify changing the production sampler or search path.

Sources: [Dirt jump biking](https://commons.wikimedia.org/wiki/File:Dirt_jump_biking.webm), [Pantemaskin](https://commons.wikimedia.org/wiki/File:Pantemaskin.webm), [Throwball](https://commons.wikimedia.org/wiki/File:Throwball.webm), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), and [pinned RT-DETR-R18 artifact](https://huggingface.co/PekingU/rtdetr_r18vd/tree/6401be7fee8b49ee00b42fcc4e0064bba8061777).
