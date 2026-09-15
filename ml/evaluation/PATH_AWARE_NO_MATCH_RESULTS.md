# English path-aware no-match results

The current CLIP/BM25/RRF ranked-list evidence cannot support a reliable no-match decision. Outcome **E** is selected. Production behavior remains unchanged, no feature flag was added, the failed held-out run was not repeated, and the frozen personal acceptance suite was not rerun.

## Protocol and data

The manifest was frozen before inference at SHA-256 `93e2bd28604e00dd217c8a7790aa07de7fa0e2ecf0dadd3a5afb27078621f95a`. It binds three calibration and three held-out Commons source groups, with no group overlap. Calibration has 36 positives and 36 negatives; held-out has 24 positives and 24 negatives. Visual has 12/12 held-out positive/negative queries, while Speech and Hybrid each have 6/6. All nine required hard-negative classes occur. Sources, licenses, attribution, and exact media checksums are in the frozen manifest.

One-threshold and two-threshold rules were enumerated only on calibration: 11,059 Visual, 2,737 Speech, and 37,540 Hybrid candidates. The objective first sought FAR and false-abstention at or below 15%, then minimized normalized gate violation. No held-out label selected a feature or threshold.

## Selected calibration rules

| Path | Rule | Calibration negative FAR | Calibration false abstention | Raw/accepted overall R@5 |
| --- | --- | ---: | ---: | ---: |
| Visual | CLIP top-1 >= 0.274567 | 16.67% | 16.67% | 83.33% / 75.00% |
| Speech | log(1 + matching BM25 segments) >= 0.895880 | 8.33% | 8.33% | 90.83% / 86.67% |
| Hybrid | speech concentration >= 0.123106 AND visual entropy strength >= 0.0000186 | 16.67% | 16.67% | 91.67% / 83.33% |

The calibration distributions already warn against Visual separation. Visual top-1 scores overlap: positives span 0.2250–0.3279 with median 0.2911, while negatives span 0.2202–0.2921 with median 0.2657. Speech matching-segment evidence separates most calibration examples, but one negative reaches a stronger value than many positives. Hybrid component distributions likewise overlap.

## Frozen held-out result

| Path | Negative FAR | False abstention | Accept precision | Coverage | Raw R@5 | Overall R@5 after abstention | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Visual | 8.33% | 83.33% | 66.67% | 12.50% | 100.00% | 16.67% | Fail |
| Speech | 33.33% | 16.67% | 71.43% | 58.33% | 66.67% | 66.67% | Fail |
| Hybrid | 50.00% | 16.67% | 62.50% | 66.67% | 83.33% | 66.67% | Fail |
| Overall explicit paths | 25.00% | 50.00% | 66.67% | 37.50% | 87.50% | 41.67% | Fail |

AUTO routed first and reached 95.83% route accuracy, above the 90% gate. With path decisions, AUTO had 25.00% negative FAR, 45.83% positive false abstention, 68.42% ACCEPT precision, and 39.58% coverage. Its useful Top-1/3/5 before rejection was 79.17%/87.50%/91.67%; after rejection the overall useful rates fell to 35.42%/43.75%/45.83%. Among the 13 accepted positives, useful Top-5 was 84.62%, but that conditional number hides eleven valid queries withheld from the user.

The decision itself costs 0.0019 ms median and 0.0021 ms p95. The serialized rules occupy 1,426 bytes, `tracemalloc` observed 770,756 bytes at peak during the timing loop, and no model memory was added. Resource gates pass; quality gates fail decisively.

## Gate and product decision

The target was at most 15% negative FAR and 15% positive false abstention per path with no material retrieval loss. No path passed all three quality requirements. Visual appears safe on negatives only because it rejects ten of twelve real positives. Speech preserves retrieval but accepts two of six hard negatives. Hybrid accepts three of six hard negatives and also loses a positive.

There was no eligible second frozen run. The personal English subset stays at its frozen baseline: 4/9 (44.44%) negatives were non-misleading, and positive useful Top-5 was 8/9 (88.89%). There are no after values because using personal acceptance after a failed development gate would violate the protocol.

SceneMind v1.0 should keep ranked results and explicit Visual/Speech/Hybrid modes, preserve AUTO as the default, and avoid copy that presents a returned timestamp as a confirmed answer. It should not hide results behind these rules. No further no-match model milestone is justified before release; product polish should communicate that search returns likely moments and let the user inspect them. The remaining release blockers are conservative result wording, a valid 30–60 minute audio-video acceptance run, and ordinary product/UI hardening already tracked in the plan.
