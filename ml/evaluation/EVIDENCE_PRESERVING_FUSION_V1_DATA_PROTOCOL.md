# Evidence-Preserving Fusion V1 Data Protocol

This protocol was fixed before candidate evaluation. It separates method selection from
source-disjoint validation and preserves all acceptance/holdout evidence.

## Dataset classification

| Dataset | Class | Sources / queries | Allowed use in this experiment |
| --- | --- | ---: | --- |
| Hybrid Fusion development V1 | **A — development** | 2 / 32 | Design the finite rank-only family and select one configuration. The manifest explicitly marks it eligible for future tuning. |
| Path-Aware No-Match heldout split | **B — validation** | 3 / 48 | One source-disjoint validation of the frozen candidate. It was previously holdout for a different no-match question, but never selected or tuned this fusion family. |
| Hybrid Fusion Holdout V1 | **C — protected holdout** | 2 / 34 | Use only if every validation gate passes. |
| Final English Acceptance V2 | **C — protected acceptance** | 1 / 30 | Isolation and leakage checks only; never tune or evaluate this candidate on it. |
| Final English Acceptance V1 | **C — protected acceptance** | 1 / 30 | Historical product acceptance only. |
| Personal Acceptance V1 | **C — protected acceptance** | 3 / 54 | Historical user evidence only. |
| Natural Video Benchmark V2 heldout | **C — protected historical evaluation** | 3 heldout videos within 5 total / 47 total annotations | Preserve original split and claims; not reused for selection. |
| Hybrid Ranking Failure Analysis V1 | **D — diagnostic only** | 19 failures + 12 controls | Motivates the hypothesis. Its inspected holdout/acceptance examples are not tuning rows. |
| Calibrated Fusion V1 | **D — historical diagnostic** | 3 calibration sources/72 queries; 2 validation sources/32 queries | Demonstrates that raw-score calibration did not generalize. No raw-score feature is reused. |
| Hybrid cap/refinement reports | **D — historical diagnostic** | Development plus failed 2-source holdout | Preserve as rejected evidence; do not tune another cap. |
| Query Routing / English Router / Human-Grounded Router | **D — different task** | Frozen train/validation/test datasets | Router evidence only; not fusion data. |
| Turkish Compatibility | **D — different language/task** | 7 videos, 72 queries | Not used. |
| Small-object, detector, sampling, verifier and pair-scorer suites | **D — different task** | Multiple frozen source groups | Not used. |

## Selected protocol

Development uses all 32 Hybrid Fusion development queries: 28 positives and four negatives over
two sources. It contains production-authoritative Top-50 Visual/Speech traces. Eight total
configurations are declared: RRF60, two strict quotas, two round-robin orders and three protected
rank depths. No neural model, score calibration or raw-score comparison is allowed.

Validation uses all 48 path-aware heldout queries from `street-traffic`, `ball-throwing` and
`find-link-talk`: 24 positives and 24 negatives. Existing local frozen assets reconstruct complete
candidate lists. Production RRF60 timestamps and scores must match all 48 historical Top-5 rows
before candidate measurement.

The designated protected holdout is Hybrid Fusion Holdout V1, checksum
`dcddffb0495cb7c397249ccd13d5818967127b6c62d01173944d0235030b64b5`. It remains unopened for
candidate ranking unless every validation gate passes. Final Acceptance V2 is additionally
protected and is never candidate evaluation data.

## Leakage audit

Source SHA-256 sets are disjoint across development, validation and the protected Hybrid
Holdout/Acceptance V2 inventory. Query text is case-folded, tokenized on word characters and
whitespace-normalized. Pairwise checks found:

- development vs validation: **0 exact**, **0 token-Jaccard ≥ 0.80**
- development vs protected: **0 exact**, **0 token-Jaccard ≥ 0.80**
- validation vs protected: **0 exact**, **0 token-Jaccard ≥ 0.80**

No data was removed automatically. Query phrasing remains tied to independently sourced media;
there was no same-video/source overlap or annotation reuse across the selected splits. The 19
manually inspected ranking failures remain diagnostic only.

## Freeze and stopping rules

Development selects by Top-5, then Top-3, Top-1, MRR@5, lower displacement and stable config ID.
The candidate JSON, finite search space, grouping and tie rules are hashed before validation. If
any validation gate fails, protected holdout is not consumed. Negative queries retain the normal
possible-match UX; no threshold or absence claim is added.
