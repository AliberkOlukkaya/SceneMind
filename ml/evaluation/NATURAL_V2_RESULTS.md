# Natural Video Benchmark V2 results

Run `2026-09-12T10:17:56.464911+00:00` at commit `8092ea30aae2e23bee9c6454b20246f6df523a21` with frozen manifest 
`3203fc545ced1ad2cfd9a335322af6b52101eb3a38408776ff94c172268088fd`. The visual cutoff `0.258063` was 
fit only on calibration visual negatives. Held-out labels were not used for tuning.

## Held-out metrics

| Path | State | R@1 | R@3 | R@5 | MRR@5 | P@5 | Negative FAR@5 | False abstention@5 | Warm median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual | raw | 71.4% | 76.2% | 85.7% | 80.5% | 20.0% | 100.0% | 0.0% | 20.4 ms |
| visual | calibrated | 31.0% | 33.3% | 38.1% | 34.3% | 8.6% | 10.0% | 52.4% | 20.4 ms |
| hybrid | raw | 66.7% | 77.8% | 77.8% | 83.3% | 20.0% | 100.0% | 0.0% | 25.9 ms |
| hybrid | calibrated | 33.3% | 44.4% | 44.4% | 50.0% | 13.3% | 33.3% | 33.3% | 25.9 ms |
| speech | raw | 60.0% | 80.0% | 80.0% | 90.0% | 24.0% | 0.0% | 0.0% | 4.9 ms |
| speech | calibrated | 60.0% | 80.0% | 80.0% | 90.0% | 24.0% | 0.0% | 0.0% | 4.9 ms |

## Held-out category breakdown

| Category | Positives | Negatives | Raw R@5 | Calibrated R@5 | Calibrated FAR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OBJECT | 8 | 0 | 100.0% | 25.0% | — |
| SCENE | 4 | 0 | 100.0% | 0.0% | — |
| COMPOSITIONAL | 5 | 0 | 100.0% | 80.0% | — |
| ACTION_TEMPORAL | 3 | 0 | 100.0% | 66.7% | — |
| NEGATIVE | 0 | 16 | — | — | 12.5% |
| SPEECH | 15 | 0 | 60.0% | 53.3% | — |

## Failure analysis

The calibrated held-out evaluation produced 23 query-path failures at K=5.
Each entry in the JSON report records the query, likely component, top timestamps, scores, transcript text, and hybrid evidence for review.

| Likely component | Count |
| --- | ---: |
| ASR transcript quality or lexical BM25 matching | 5 |
| modality mismatch: visual path cannot retrieve spoken content | 4 |
| open-set rejection / score calibration | 2 |
| single-frame sampling and temporal representation | 1 |
| visual composition binding | 1 |
| visual embedding or five-second frame sampling | 10 |

## Scope

This benchmark is a small diagnostic set of five openly licensed videos, with two calibration and three source-disjoint held-out videos. Repeated evaluation modes are separate query-path trials. The measurements characterize this pinned local configuration; they are not population accuracy estimates.
