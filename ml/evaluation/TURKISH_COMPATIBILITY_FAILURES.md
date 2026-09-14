# Turkish compatibility failure analysis

The frozen outcome is **E — Turkish compatibility still fails acceptance**. The cheap candidate passes resource and English-regression constraints but fails held-out Turkish routing and AUTO retrieval quality.

## Router generalization

The production router reached 53.33% held-out Turkish accuracy. The selected 14-feature linear candidate improved this to 76.67%, still below the fixed 90% gate. Seven of 30 held-out queries were misrouted:

- ela-07: Visual to Speech
- ela-09: Hybrid to Visual
- ela-10: Hybrid to Speech
- brv-11: Hybrid to Visual
- brv-12: Hybrid to Visual
- jmp-02: Visual to Hybrid
- jmp-08: Visual to Hybrid

The errors span all three held-out source groups and concentrate around short visual wording, ambiguous context and Hybrid phrasing. Adding rules for these individual queries would tune on held-out evidence and is prohibited. More independent source groups are needed before choosing another router.

## Retrieval compatibility

The cheap AUTO candidate missed Top-5 on six rows: ela-03, ela-07, ela-09, brv-06, brv-07 and brv-12. Two failures follow directly from wrong routing into a path without usable evidence. Two Brave Wikipedian visual failures show that translating or expanding Turkish terms can damage direct CLIP retrieval that already works well.

The stage comparison separates three effects:

1. Visual language mismatch is modest. Direct Turkish Visual R@5 is 93.33%, only 6.67 percentage points below paired English queries. Cheap lexical adaptation makes it worse at 86.67%.
2. Speech is limited by lexical alignment and ASR errors. Cheap Turkish normalization improves R@5 from 75.00% to 87.50%. English queries against Turkish ASR collapse to 12.50%, so global Turkish-to-English translation is the wrong architecture for Turkish transcripts.
3. Hybrid inherits router and path weaknesses. Cheap Hybrid R@5 remains 71.43%, so fusion does not repair missing or weakened evidence.

## Semantic diagnostic limits

The pinned multilingual MiniLM diagnostic shows that semantic text matching could improve held-out Speech R@5 to 100% and Hybrid R@5 to 85.71%. It does not solve routing: total AUTO R@5 is 86.67% while router accuracy remains 76.67%.

It also adds about 254 MiB RSS, 458 MiB of cache, 9.995/14.378 ms warm median/p95 query latency and a 46.06-second cold first load. This diagnostic was intentionally isolated and is not eligible for production promotion.

## Negative queries and acceptance

Global no-match behavior is outside this milestone. Rejection thresholds were not tuned and negative behavior was not remeasured. The personal acceptance labels and manifest remain frozen and unchanged. Since the development quality gates failed, no personal acceptance after-run exists.

## Remaining blockers

- Turkish intent routing does not generalize across source groups: 76.67% versus the 90% gate.
- Cheap lexical adaptation reaches only 80% held-out AUTO R@5 and can harm direct Turkish CLIP retrieval.
- Semantic Speech retrieval is promising but has significant CPU memory/cache cost and cannot compensate for the failing router.

Another ML promotion milestone should not begin immediately. First collect independent Turkish source groups and freeze a new router-validation split. Only then reconsider the compact semantic Speech branch. The current held-out set must remain untouched for model and rule selection.
