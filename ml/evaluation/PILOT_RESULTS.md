# Licensed scene-disjoint pilot — measured 2026-09-12

Source: Big Buck Bunny, Blender Foundation, CC BY 3.0; attribution and checksum in `bunny.json`. Two disjoint 40-second animated clips; eight queries each, including four negatives. Reviewed five-second frame labels were frozen before retrieval. This is a single-film appearance-retrieval pilot, not broad real-video accuracy, motion understanding or speech evaluation.

| Held-out metric | Raw | Calibrated |
| --- | ---: | ---: |
| Recall@1 | 0.375 | 0.375 |
| Recall@3 | 0.875 | 0.875 |
| Recall@5 | 1.000 | 1.000 |
| MRR@1 | 0.750 | 0.750 |
| MRR@3 / @5 | 0.8333 | 0.8333 |
| Negative false accepts | 4/4 | 2/4 |
| Positive-query abstention | 0/4 | 0/4 |

Threshold: 0.21446362137794497, fitted only on calibration negatives. Calibration false accepts were 0/4 by construction. The remaining held-out false matches show that the threshold does not establish semantic absence, especially for compositional queries. Keep it opt-in. Do not promote these counts to a population accuracy claim.

The repeated run matched the first run's quality metrics and passed the frozen regression gate. Warm median across all pilot queries was 28.791 ms in the recorded final run; local browser/model checks ran concurrently, so timing is descriptive only. The synthetic baseline rerun also passed its two positive queries at K=1 while returning a false match for its one negative.

Full raw measurements: `reports/bunny-v1.json`. Frozen threshold and split-specific metrics: `reports/bunny-v1-calibration.json`. No media, weights or embeddings are committed.

## Reproduce from the repository root

Use the repository virtual-environment Python (`.venv/Scripts/python` on Windows):

```text
python scripts/prepare_benchmark.py
python -m ml.evaluation.run --manifest ml/evaluation/bunny.json --k 5 --repeats 3 --output data/bunny-report.json
python -m ml.evaluation.calibrate data/bunny-report.json --output data/calibration.json
python -m ml.evaluation.regression data/calibration.json ml/evaluation/reports/bunny-v1-calibration.json
python -m ml.evaluation.run --synthetic --output data/synthetic-report.json
```

To add natural footage: provide a licensed local video path, attribution/source, split, source group, and reviewed query intervals using the same schema. Keep source-related footage in the same split. Include ordinary and compositional negatives. Freeze annotations before inference. The runner validates interval bounds against video duration. Calibration refuses missing positives/negatives, held-out fitting, cross-split groups and identical video content. Regression never rewrites the baseline automatically. Speech/hybrid queries are supported by the runner but require their own labels and calibration study; this visual pilot does not validate them.
