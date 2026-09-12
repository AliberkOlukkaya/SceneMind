# Image-text verifier results

The tested verifier is `Salesforce/blip-itm-base-coco` at revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`. It was evaluated only as an isolated second stage over CLIP's top five. Production remains unchanged.

The calibration pool contained 46 visual queries over five source-disjoint videos: 25 positives and 21 negatives. The locked match threshold was `0.43383792042732244`, the lowest threshold allowing no more than 10% calibration negative accepts. At that threshold calibration FAR was 9.5%, but positive false abstention was already 44%. No calibration threshold simultaneously reached FAR ≤10% and false abstention ≤20%; when false abstention was constrained to ≤20%, the best observed FAR was 28.6%.

## Frozen Natural V2 held-out results

| System | R@1 | R@3 | R@5 | MRR@5 | Negative FAR@5 | Positive false abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Raw CLIP | 71.4% | 76.2% | 85.7% | 80.5% | 100.0% | 0.0% |
| B. Current scalar CLIP calibration | 31.0% | 33.3% | 38.1% | 34.3% | 10.0% | 52.4% |
| C. CLIP top five + BLIP reranking | 61.9% | 81.0% | 85.7% | 76.3% | 100.0% | 0.0% |
| D. CLIP top five + BLIP reranking and threshold | 26.2% | 26.2% | 26.2% | 28.6% | 10.0% | 52.4% |

BLIP preserved candidate R@5 because it only reordered the same five frames, satisfying the ≤2 percentage-point candidate-recall-loss gate with zero loss. It reduced R@1 by 9.5 points and MRR@5 by 4.1 points. Its calibrated result is worse than the current calibrated CLIP result on recall and MRR while producing identical held-out FAR and false abstention.

Added warm median verifier latency was 3.352 seconds per batch of five, versus the 0.250-second ceiling. A fresh-process memory measurement recorded 1,491,443,712 bytes peak working set and a 1,395,855,360-byte increase after loading and running BLIP. The cached full-run model load took 2.05 seconds; parameter count was 223,744,258. These measurements are Windows CPU in-process results, not network latency.

## Promotion decision

The verifier failed two primary gates:

- Positive false abstention: 52.4%, limit 20%.
- Added warm median latency: 3.352 seconds, limit 0.250 seconds.

It met the 10% negative FAR boundary and preserved candidate R@5, but that is insufficient. No second held-out run was performed because the promotion rule requires a first-run pass. No production integration, feature flag, dependency, or runtime configuration was added.

Machine-readable artifacts are `reports/verifier-blip-v1.json` and `reports/verifier-blip-v1-calibration.json`. The report contains all candidate ranks, CLIP scores, verifier probabilities, timings, manifests, revisions, category slices, and resource measurements.
