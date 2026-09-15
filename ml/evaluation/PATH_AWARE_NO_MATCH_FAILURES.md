# English path-aware no-match failure analysis

The failure is score non-separation, not latency or routing. AUTO selects the expected path for 46 of 48 held-out queries. Once on that path, the cheap evidence cannot simultaneously retain positives and reject hard negatives.

## Visual

The calibration top-score cutoff transfers poorly across source/domain appearance. It falsely abstains on 10/12 held-out positives, including buses, a bicycle, the persistent intersection, pedestrians crossing, moving traffic, the man, yellow ball, park path, man holding the ball, and the throw. These include both easy persistent scenes and small/action content, so a higher CLIP score is not a stable existence probability. The rule also accepts the absent red tram because the street scene contains visually related vehicles and traffic context.

## Speech

The matching-segment count retains five of six positives but accepts two of six negatives. “Tokyo airport” matches real airport segments despite the wrong city, and “downloading the article” collects in-domain lexical evidence even though that action is never explained. The rejected FindLink introduction exposes an ASR/tokenization weakness: exact or near-exact named entities can appear too sparsely to satisfy a segment-count rule. Increasing the threshold would reject more legitimate single-mention facts; lowering it raises FAR.

## Hybrid

The selected conjunction accepts three of six negatives: a red screen attribute mismatch, a Wikidata-logo mismatch, and a download-plus-city-map compositional mismatch. RRF and list summaries can show that each modality has related evidence, but they cannot establish that all requested attributes and relations co-occur. The FindLink introduction is again falsely withheld. Requiring both calibrated path decisions was evaluated through explicit state features but did not provide the best calibration tradeoff and does not solve attribute grounding.

## Why production is unchanged

The prior global threshold failed by rejecting 90.48% of positives. This path-aware attempt improves the aggregate figure to 50%, but still misses the 15% maximum by a wide margin and drops overall positive R@5 from 87.50% to 41.67%. Speech-only promotion is also unsafe because its held-out FAR is 33.33%. A second run would only confirm deterministic arithmetic on a configuration that already fails; it cannot repair external validity.

No uncertainty API/UI state or feature flag was added because the promotion gate failed. The safe v1.0 response is editorial: describe outputs as likely moments, retain manual path overrides, and avoid a definitive “not present” or confirmed-match claim. The full per-query features, top-five evidence, decisions, and failures remain in the machine report.
