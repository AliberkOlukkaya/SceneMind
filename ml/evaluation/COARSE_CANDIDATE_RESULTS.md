# Coarse candidate diversity results

## Protocol

The frozen 66-query calibration split chose every radius, pool depth, MMR weight and visual-change threshold. The 31 held-out rows and two held-out human-reviewed small-object events did not affect selection. All methods use real pinned CLIP embeddings and exact FAISS inner-product retrieval. Negative scores are diagnostic only; no no-match threshold was fitted.

The selected policy retrieves a top-20 pool from a multi-scale index, retains the five-second base frames, admits two-second frames whose adjacent CLIP embedding change is at least 0.20, then applies temporal NMS with a five-second radius. Final K is five. A fixed 50 ms tolerance handles observed VFR timestamp jitter.

## Held-out retrieval

| Method | Configuration chosen on calibration | R@1 | R@3 | R@5 | MRR@5 | Reviewed small-object visible R@5 |
|---|---|---:|---:|---:|---:|---:|
| A. Current 5 s raw CLIP | top 5 | 71.43% | 76.19% | 85.71% | 0.8048 | 50% |
| B. Global 2 s raw CLIP | top 5 | 57.14% | 69.05% | 76.19% | 0.6706 | 0% |
| C. Temporal NMS | top 20, radius 5 s | 57.14% | 66.67% | 69.05% | 0.6524 | 0% |
| D. Embedding MMR | top 20, lambda 0.90 | 57.14% | 71.43% | 76.19% | 0.6762 | 50% |
| E. Temporal + embedding | top 20, radius 2 s, lambda 0.90 | 57.14% | 71.43% | 71.43% | 0.6667 | 50% |
| F. Scene-aware grouping | top 20, CLIP-change 0.10 | 57.14% | 57.14% | 57.14% | 0.6190 | 0% |
| Selected multi-scale diagnostic | top 20, change 0.20, radius 5 s | 71.43% | 78.57% | 80.95% | 0.7952 | 50% |

The selected result preserves held-out object, compositional, scene, action/temporal and benchmark small-object R@5 at 100%. Speech-label queries on the visual path fall from 40% to 20%, a 20-point category regression. Their intended product path remains transcript search.

## Raw-pool coverage and redundancy

Correct-region recall before final selection is 92.86%/100% for the five-second top-20/top-50 pools and 90.48%/97.62% for the two-second pools. On the reviewed small-object evidence, the two-second top-20 reaches 50% held-out visible recall and top-50 reaches 100%; the five-second pools remain at 50% because one event has no visible five-second JPEG.

The five-second top five averages 3.26 unique five-second temporal regions, 5.0-second minimum spacing and 69.0% candidates with a neighbor within five seconds. Global two-second top five falls to 2.94 regions and 2.0-second spacing while redundancy rises to 77.4%. The selected multi-scale policy restores 3.26 regions, raises median minimum spacing to 10 seconds and measures 0% five-second-radius redundancy, but semantic recall still trails the baseline.

Global two-second indexing underperforms because denser adjacent frames compete for the same five slots. The three positive queries found by five-second top five and missed by two-second top five are all speech-label visual-path diagnostics. Every correct region remains in the two-second top 20, and every failing top five contains neighboring competitors. The higher density therefore changes CLIP rank order and dilutes distinct temporal coverage; it is not evidence that the relevant moments vanished from the index.

## Resources and gates

Five-second storage is 98 JPEGs, 1.84 MiB of JPEG data and 196 KiB of embeddings. Global two-second storage is 235 JPEGs, 4.55 MiB and 470 KiB: 2.40x the frame/index size and 2.47x the JPEG size. Its parent run measured 22.28 seconds for warm-model extraction plus encoding. The historical five-second ingestion plus indexing total is 24.40 seconds but includes a different model/cache state, so those wall times are not a paired speed ratio.

FAISS top-50 median/p95 is 0.042/0.069 ms at five seconds and 0.048/0.099 ms at two seconds. Selected diversification adds 0.005 ms median and 0.012 ms p95. Including unchanged CLIP text encoding, the selected query path measures 17.31 ms median and 24.78 ms p95. Added measured index RSS is 0.79 MiB. The ranking logic itself is cheap; dense ingestion and storage are the resource cost.

Promotion fails: R@5 is 80.95% versus the required 85.71%, reviewed small-object R@5 does not materially improve, speech drops by 20 points, and index growth is 2.40x. Latency gates pass. No second frozen run was required and production was not modified.

The final choice is **F: the larger candidate pool is high recall, but final selection is not promotable**. Keep production five-second CLIP unchanged. The next bounded capability is a calibration-only candidate-list ranking/no-match experiment over a bounded raw pool. Video RAG remains deferred until both final ranking and rejection pass.
