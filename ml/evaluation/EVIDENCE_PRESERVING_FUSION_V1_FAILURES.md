# Evidence-Preserving Hybrid Fusion V1 Failure Analysis

The frozen one-slot modality quota is a useful diagnostic, but it is not a safe general replacement
for RRF60.

## What improved

- One Speech failure, `talk-s-languages`, moves from outside Top-5 to rank 1.
- Speech Top-1/3/5 each gain one query.
- All 21 baseline Top-5 successes remain inside Top-5.
- Explicit-lane displacement and irrelevant-consensus counts each fall from one to zero.
- Top-1, Top-3, Top-5 and MRR all improve in aggregate.

## Why it still fails

The entire gain comes from six Speech positives on one validation source family. Visual is already
saturated at 12/12 and contributes no evidence that the rule improves another category. Hybrid
Top-K is unchanged, while its MRR falls from 0.7500 to 0.7222. In `talk-h-airport`, a correct rank-1
Hybrid result moves to rank 3 because the reserved single-lane bucket receives priority over the
strong shared RRF winner.

This is precisely the accuracy tradeoff the cross-category gate was designed to reject: preserving
one lane can rescue displaced Speech evidence, but fixed reservation can also demote strong genuine
agreement. Development already showed the same warning: quota 1 improved Top-3/5 but reduced Top-1
from 10 to 8.

## Negative queries

The candidate changes the first returned bucket for `talk-ns-tokyo`, `talk-ns-download` and
`talk-nh-red`. All requested content is absent. The interface remains conservative and no no-match
threshold was added, but the annotations do not establish that the new candidates are less or more
misleading. Negative behavior therefore cannot support promotion.

## Stopping decision

Validation failed before protected evaluation. Hybrid Holdout V1 and Final Acceptance V2 were not
ranked by the candidate. Production remains uncapped RRF60. Per protocol, do not respond by trying
another quota, reranker, VLM or calibration method on these validation outcomes. Choose the next
product milestone separately.
