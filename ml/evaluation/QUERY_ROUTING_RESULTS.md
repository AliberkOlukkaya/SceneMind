# Query routing results

The frozen learned AUTO router passes the product gate and exactly matches the explicit-route Oracle's held-out search quality. Decision C promotes the 54-parameter classifier while preserving every explicit mode.

| Strategy | Routing accuracy | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous UI default / always Hybrid | 3.23% | 80.95% | 88.10% | 90.48% | 0.9286 |
| Always Visual | 77.42% | 71.43% | 76.19% | 85.71% | 0.8048 |
| Always Speech | 19.35% | 14.29% | 19.05% | 19.05% | 0.2143 |
| Oracle explicit route | 100% | 85.71% | 92.86% | 95.24% | 0.9762 |
| Heuristic | 70.97% | 85.71% | 92.86% | 95.24% | 0.9762 |
| Learned AUTO | 96.77% | 85.71% | 92.86% | 95.24% | 0.9762 |
| Learned plus confidence fallback | 96.77% | 85.71% | 92.86% | 95.24% | 0.9762 |

The previous product UI default was Hybrid; the backend's omitted-mode default remains Visual for API compatibility. On videos without a transcript, Hybrid retains the available Visual results. Always Speech returns no results where no transcript path was measured.

The selected held-out confusion matrix is:

| Actual → / Predicted ↓ | Visual | Speech | Hybrid |
| --- | ---: | ---: | ---: |
| Visual | 24 | 0 | 0 |
| Speech | 0 | 6 | 0 |
| Hybrid | 1 | 0 | 0 |

The sole routing error is the negative query “Where does the speaker demonstrate cooking?”, labelled Hybrid and predicted Visual. It cannot reduce positive recall or a category slice. Relative to explicit Oracle, Object, Scene, Compositional, Action/Temporal, and Speech R@5 drops are all zero.

1. Speech calibration size is 961.304 seconds with 205 real Whisper-tiny segments.
2. Calibration uses three independent Commons source groups.
3. It contains 36 frozen queries: 12 Visual, 12 Speech, and 12 Hybrid.
4. Held-out heuristic routing accuracy is 70.97%; calibration accuracy is 94.44%.
5. Learned held-out accuracy is 96.77%; calibration fit is 100% and leave-one-source-out accuracy is 94.44%.
6. The selected confusion matrix is 24/24 Visual correct, 6/6 Speech correct, and 0/1 Hybrid correct.
7. Always-Visual R@5 is 85.71%.
8. Always-Speech R@5 is 19.05%.
9. Always-Hybrid and the previous UI default reach 90.48% R@5.
10. Oracle explicit-mode R@5 is 95.24%.
11. AUTO R@1/R@3/R@5 is 85.71%/92.86%/95.24%.
12. AUTO MRR@5 is 0.9762.
13. Hybrid fallback also reaches 95.24%; calibration selects threshold zero, so it adds no fallback decisions.
14. Median routing latency is 0.031 ms.
15. Routing p95 is 0.042 ms.
16. The >=93% R@5, <=5-point category drop, latency, and artifact gates pass.
17. The second identical frozen run passes with the same accuracy, confusion matrix, and retrieval metrics.
18. AUTO is integrated behind `SCENEMIND_AUTO_ROUTING_ENABLED` and is the UI default.
19. Explicit Visual, Speech, and Hybrid modes remain available and bypass AUTO.
20. The blocker before SceneMind v1.0 is a private, frozen acceptance run on real 30–60 minute personal lecture/demo/podcast/ordinary videos, including transcript readiness and user override behavior.

For a real 30–60 minute upload today, AUTO can select the right engine for query forms represented by this small English benchmark, provided the visual index and transcript have been built. Its frozen accuracy is 96.77% and its search quality matches explicit routing. This does not yet prove reliability for Aliberk's own vocabulary, Turkish queries, long-video ASR errors, or different domains. SceneMind v1.0 should provisionally default to AUTO while keeping the explicit selector visible; the personal acceptance milestone is required before calling that default reliable for daily use.

The global no-match threshold remains unresolved and disabled. SceneMind returns ranked evidence instead of repeating the previous 90.48% positive false-rejection failure.
