# Candidate-list ranking results

The frozen experiment did not pass either quality gate. Production search remains unchanged.

| System | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Raw CLIP top-20/top-50 | 71.43% | 76.19% | 85.71% | 0.8048 |
| Normalized score | 71.43% | 76.19% | 85.71% | 0.8048 |
| Score margin | 71.43% | 76.19% | 85.71% | 0.8048 |
| Temporal support, top-20 | 71.43% | 76.19% | 83.33% | 0.7952 |
| Temporal support, top-50 | 71.43% | 76.19% | 85.71% | 0.8048 |
| Calibration-selected logistic, top-20 | 66.67% | 73.81% | 76.19% | 0.7302 |
| Logistic, top-50 | 66.67% | 76.19% | 83.33% | 0.7579 |
| Existing explicit channel routing | 85.71% | 92.86% | 95.24% | 0.9762 |

Oracle correct-interval recall is 85.71% at 5, 92.86% at 20, and 100% at 50. Top-50 therefore contains useful headroom, but these cheap list statistics do not identify it across sources. Top-20 and top-50 produce the same raw first five; expanding the logistic input to 50 does not improve R@1 and remains below baseline R@5.

The query-level logistic threshold is 0.581239, chosen only from calibration under the 10% FAR limit. Calibration FAR is 9.68% and positive false abstention is already 42.86%. Held-out FAR is 0%, precision among accepted queries is 100%, and positive false abstention is 90.48%: 2 of 21 positives are accepted and 29 of 31 total queries are rejected. The FAR gate passes and the <=20% positive-false-abstention gate fails.

The selected scorer preserves held-out object, scene, compositional and small-object R@5, but speech R@5 falls from 40% to 0%, a 40-point category drop. The ranking gate fails its 85.71% R@5 floor, the preferred R@1/MRR improvements, and the five-point category-drop limit. Explicit correct-channel routing uses the already measured BM25 path for speech and reaches 95.24% R@5; this is an audit of an existing user-selected mode, not an automatic query classifier.

Added list feature, ranking, and no-match inference costs 0.441 ms median and 0.961 ms p95 on the measured Windows CPU. The 29 float64 parameters occupy 232 bytes, well within the 250 MB added-RAM gate. The full process RSS includes resident CLIP and is not attributed to the scorer. Resource gates pass, but resource efficiency cannot rescue failed quality.

1. The input pool is production five-second CLIP; production search was not modified.
2. Calibration has 66 queries from seven sources; held-out has 31 queries from three disjoint sources.
3. All learned weights, hyperparameters, and the no-match threshold use calibration only.
4. Raw held-out R@1/R@3/R@5 is 71.43%/76.19%/85.71%; MRR@5 is 0.8048.
5. Oracle@5/@20/@50 is 85.71%/92.86%/100%.
6. Score normalization and margin transforms cannot change order and exactly match raw ranking.
7. Calibration-selected temporal support at top-20 lowers R@5 to 83.33%.
8. The same temporal formulation at top-50 matches raw R@5 but adds no gain.
9. The selected top-20 logistic scorer reaches 66.67%/73.81%/76.19% R@1/3/5.
10. Its MRR@5 is 0.7302, below the 0.8048 baseline.
11. The top-50 logistic variant reaches 83.33% R@5 and still fails the floor.
12. Object, scene, compositional, action, speech, small-object, and negative slices are stored in the machine report; no hybrid-positive row exists in the frozen set.
13. The selected scorer's largest category drop is speech at 40 points.
14. No-match calibration FAR/PFA is 9.68%/42.86%.
15. No-match held-out FAR/PFA is 0%/90.48%; accepted precision is 100% on only two accepts.
16. Real cross-path evidence records CLIP, BM25, and RRF best scores/result counts where those runs exist.
17. Existing explicit routing reaches R@1/R@3/R@5 of 85.71%/92.86%/95.24% and MRR@5 0.9762.
18. Added median/p95 is 0.441/0.961 ms; parameter storage is 232 bytes.
19. Both quality gates fail, so no second frozen run or feature flag is allowed.
20. Decision E applies: routing is the dominant measured failure; keep raw CLIP order and explicit modes, collect source-disjoint speech calibration before testing automatic routing, and postpone a stronger semantic reranker until the route is correct.

For a 30–60 minute lecture or podcast today, SceneMind can be trusted for exact spoken terms when transcription is ready and the user selects Speech; paraphrases remain weak because BM25 is lexical. For a software demo, Visual is useful for broad objects and scenes and Speech for narrated terms; precise UI text and short actions remain untrusted. For ordinary video, broad visible objects/scenes are useful candidate search, while absence claims, brief actions, tiny objects, and cross-modal questions are untrusted. Hybrid is useful when both evidence types may apply, but it is not a calibrated confidence signal.

Exact rows, feature/model parameters, split selection evidence, slice metrics, path evidence, resource timing, and gates are in `reports/candidate-list-ranking-v1.json`.

