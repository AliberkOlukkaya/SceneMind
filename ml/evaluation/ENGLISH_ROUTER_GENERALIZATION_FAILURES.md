# English AUTO router generalization failures

The unchanged 54-parameter production router misses 45 of 90 frozen source-card queries. It sends 26 of 30 Speech queries to Visual and recovers only 11 of 30 Hybrid queries. Its high confidence on these mistakes shows weak feature coverage rather than useful uncertainty calibration.

| Diagnostic slice | Queries | Baseline failures | Baseline accuracy | Candidate failures | Candidate accuracy |
|---|---:|---:|---:|---:|---:|
| Natural question phrasing | 34 | 17 | 50.00% | 1 | 97.06% |
| Long queries | 18 | 9 | 50.00% | 1 | 94.44% |
| Technical terms | 30 | 13 | 56.67% | 1 | 96.67% |
| Indirect Speech intent | 26 | 25 | 3.85% | 3 | 88.46% |
| Indirect Visual intent | 18 | 0 | 100.00% | 0 | 100.00% |
| Mixed modality | 30 | 19 | 36.67% | 1 | 96.67% |
| Ambiguous wording | 12 | 7 | 41.67% | 0 | 100.00% |
| Lexical-overlap traps | 7 | 3 | 57.14% | 1 | 85.71% |
| Contains unseen source vocabulary | 90 | 45 | 50.00% | 4 | 95.56% |

The character candidate's four misses are three Speech→Hybrid decisions and one Hybrid→Visual decision. Three involve indirect Speech intent, one is a mixed-modality long question, and one is a lexical-overlap trap. Validation-selected Hybrid fallback causes two additional frozen-test errors overall: raw accuracy is 97.78%, compared with 95.56% after fallback.

The main failure is evidence quality. The 450 queries are unique, balanced, naturally phrased, and grouped into independent source scenarios, but no reviewer verified them against actual video/audio evidence. Route labels are internally consistent because they were authored from declared evidence requirements; they do not demonstrate generalization to how users describe moments in real videos. Promoting on this result would repeat the earlier mistake of trusting a narrow routing benchmark.

Production therefore remains on `query_router_v1.json`. No candidate vocabulary, weights, fallback threshold, feature flag, or retrieval change is shipped. The protected final acceptance evidence remains failed and untouched.
