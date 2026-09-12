# Next model recommendation

Implement a compact image-text matching reranker with an explicit no-match score as the next AI capability, after expanding the calibration set. Apply it only to the current CLIP top candidates. Preserve CLIP and BM25 as fast candidate generators.

Natural V2 shows that candidate generation is not the main visual bottleneck: raw held-out R@5 is 100% for every non-speech category, including the three `ACTION_TEMPORAL` query-path trials and five `COMPOSITIONAL` trials. The current scalar CLIP cutoff is the bottleneck. It reduces visual negative FAR@5 from 100% to 10%, but raises positive false abstention to 52.4%. It also accepts the absent “gradient descent” query because presentation context is visually plausible. A candidate-level matching model can assess the query and frame jointly and expose a no-match logit for calibration, which is closer to the measured need than replacing the high-recall retriever.

The first experiment should remain isolated from product code:

1. Add more source-disjoint calibration videos and freeze them before model comparison. Keep Natural V2 held-out unchanged.
2. Shortlist one CPU-capable pretrained image-text matching model with a documented revision and license. Do not fine-tune it.
3. Rerank only the top five CLIP frames and calibrate its no-match or match score on the expanded calibration split.
4. Require held-out visual negative FAR@5 at or below 10%, positive false abstention at or below 20%, and raw candidate R@5 loss no greater than two percentage points. Record CPU latency and peak memory; use 250 ms added warm median latency per query as the initial local-product ceiling.
5. Promote it only if a second frozen rerun passes the same regression gate.

Do not implement a temporal action model from this result. The present temporal cases are easy and all reach raw R@1 and R@5 of 100%; the one calibrated action failure has the correct frame at rank 1 but a score just below the cutoff. Do not enlarge Whisper yet either: speech already gives 0% negative FAR, 0% positive false abstention, and 80% R@5. Its remaining errors call first for interval-aware result diversification and a documented timestamp tolerance.

Automatic query-path routing and evidence-aware hybrid fusion are useful later. They should be evaluated after the visual verifier because the current hybrid path inherits the visual false accept and the cutoff's domain shift.
