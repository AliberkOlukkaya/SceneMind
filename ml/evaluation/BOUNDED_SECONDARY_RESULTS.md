# Bounded secondary sampling results

## Decision

Choose **C: coarse candidate recall is the dominant remaining bottleneck**. Do not integrate the branch. Improve coarse candidate generation and semantic temporal diversification before another detector or production trial.

The machine report is [`reports/bounded-secondary-v1.json`](reports/bounded-secondary-v1.json). It is bound to the frozen 97-query parent report and Natural V2 manifest. Media, JPEG caches, embeddings and weights remain ignored.

## Selected policy

The calibration-only grid compared ±2/4/6-second windows, top 5/10/20 and either no spacing or six-second temporal spacing. The winner is **top 5, ±2 seconds, no spacing**. It produces 9.1 secondary frames per routed calibration query and reaches 90.7% routed calibration R@5.

Top 10 and top 20 do not improve R@5. Their best calibration settings both reach 88.9%. Six-second spacing reduces the selected top-5 policy from 9.1 to 6.1 frames per query and raises R@1 from 48.1% to 53.7%, but lowers R@5 from 90.7% to 88.9%. It improves efficiency and early rank, but does not win the predeclared R@5-first rule.

## Held-out visual quality

The current production visual path and global 5-second CLIP baseline are the same in this visual-only comparison. Speech and hybrid behavior are unchanged.

| System | R@1 | R@3 | R@5 | MRR@5 | Negative FAR | Positive false abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current production visual / global 5 s CLIP | 71.4% | 76.2% | **85.7%** | 0.805 | 100% | 0% |
| Hypothetical global 2 s CLIP | 57.1% | 69.0% | **76.2%** | 0.671 | 100% | 0% |
| Bounded 2 s, coarse order | 71.4% | 76.2% | **83.3%** | 0.805 | 100% | 0% |
| Bounded 2 s + secondary CLIP | 61.9% | 66.7% | **81.0%** | 0.721 | 100% | 0% |
| Bounded + CLIP + Nano 640 + abstention | 7.1% | 9.5% | **14.3%** | 0.105 | **0%** | **81.0%** |

The fusion grid tests object weights 0, 0.05, 0.10 and 0.20 on calibration only. **Zero wins**: detector evidence reduces calibrated recall. The selected score threshold is 0.2780004 and reaches 0% held-out FAR, but abstains on every small-object query and 81.0% of all held-out positives.

Small-object results are:

| System | R@1 | R@3 | R@5 | MRR@5 | False abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global 5 s | 83.3% | 83.3% | 100% | 1.000 | 0% |
| Global 2 s | 16.7% | 50.0% | 100% | 0.583 | 0% |
| Bounded coarse order | 83.3% | 83.3% | 83.3% | 1.000 | 0% |
| Bounded secondary CLIP | 16.7% | 16.7% | 66.7% | 0.417 | 0% |
| Final detector/abstention branch | **0%** | **0%** | **0%** | 0.000 | **100%** |

At K=5, final object-presence recall is 0%, relationship recall 50%, compositional recall 50%, scene recall 0%, action/temporal recall 0%, and negative FAR 0%. Scene/action queries use the unchanged coarse fallback; the global no-match threshold still removes them, showing that generic scalar abstention remains the quality failure.

## Actual visible-frame diagnosis

On the four human-reviewed small-object events, the selected local windows contain real visible evidence in 50% overall and 50% held-out. Secondary CLIP finds a verified visible frame at K=5 in 25% overall and **0% held-out**. The maximum tested ±6/top-20/diversified policy reaches 100% window recall, but uses 25–67 frames per event and still reaches only 50% overall and held-out secondary-CLIP recall. Global 2-second CLIP reaches 25% overall and 0% held-out on the same verified evidence.

This separates the failures: two events never enter the selected local windows; another reaches the window but is displaced by semantically similar frames. A stronger detector cannot repair either failure.

## Resources and cache

The selected run routes 57 of 97 queries and averages **8.98 secondary frames**. Across those queries it makes 512 frame requests: 128 cold misses and 384 reuse hits, a **75% cache hit rate**. The deterministic cache holds 128 JPEGs totaling 3,024,047 bytes. The configured ceiling is 512 MiB.

Measured component totals across routed queries are 18.11 seconds decoding, 5.37 seconds secondary CLIP image encoding and 2.78 seconds detector inference. Cross-query cache/model reuse gives a 27.8 ms median added latency, while cold cache misses drive **2.54 seconds p95** and 0.459 seconds mean. Added peak RSS is **84.3 MiB**. The warm median passes the preferred target; p95 is not production-interactive.

Global 5-second assets contain 98 frames, 1,931,324 JPEG bytes and 200,704 embedding bytes. Global 2 seconds contains 235 frames, 4,773,878 JPEG bytes and 481,280 embedding bytes, about 2.40–2.47 times the frame/index storage. Its measured extraction plus CLIP work is 22.28 seconds across the ten source videos.

Linear projections from that measured global-2-second run are labeled estimates:

| Video length | Global 5 s frames | Global 2 s frames | Est. 5 s work | Est. 2 s work | Bounded routed query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 min | 121 | 301 | 11.5 s | 28.5 s | 8.98 measured average, max 15 |
| 30 min | 361 | 901 | 34.2 s | 85.4 s | 8.98 measured average, max 15 |
| 60 min | 721 | 1,801 | 68.3 s | 170.7 s | 8.98 measured average, max 15 |

The bounded frame count is independent of video duration under top-5 ±2-second windows. Cold decode time still depends on cache history and source seek cost.

## Gates

| Critical gate | Result | Pass |
| --- | ---: | :---: |
| Small-object false abstention <=20% | 100% | No |
| Negative FAR <=10% | 0% | Yes |
| Candidate R@5 loss <=2 pp | 0 pp after selected zero-weight fusion | Yes |
| Added warm median <=400 ms | 27.8 ms | Yes |
| Added peak memory <=500 MiB | 84.3 MiB | Yes |

The quality gate fails, so no second frozen run is required and no feature flag or production code is added.
