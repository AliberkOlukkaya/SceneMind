# Temporal Video Retrieval V1 — work in progress

Development is complete. The fixed selection rule chose 4s/2s/eight frames. Frozen validation,
architecture decision and historical replay have not run. These results do not
support production promotion or a final A/B/C/D/E decision.

| Development, 66 queries | R@1 | R@3 | R@5 | R@10 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Production CLIP | 48.48% | 63.64% | 66.67% | 75.76% | 55.98% |
| X-CLIP 4s / stride 2s | 13.64% | 25.76% | 30.30% | 39.39% | 19.60% |
| X-CLIP 8s / stride 4s | 9.09% | 15.15% | 24.24% | 39.39% | 13.81% |

The 4-second candidate regresses R@5 by 36.36 percentage points. Combined
action/interaction/event R@5 is 33.33%, versus CLIP's 55.56%; combined static
R@5 is 20%, versus 80%. Its cost is 42.55 indexing seconds/source-minute versus
1.98 for CLIP, 3,087,037 versus 24,648 index bytes/source-minute, and query
median/p95 0.312/1.099 seconds versus 0.032/0.104 seconds. Absolute process-tree
peak RSS is 1,239,293,952 versus 1,243,295,744 bytes. Model loads are 10.83 versus
15.56 seconds, including local cache/library overhead. These are host measurements,
not a validated 4-vCPU deployment. Raw per-source/category details are in the JSON.

Three CLIP raw misses are useful results just outside coarse integer ground-truth
endpoints. Labels and metrics remain frozen; see the failure report. This cannot
explain the large observed candidate regression, but limits small-delta inference.

One candidate was predeclared before retrieval: MIT X-CLIP base/patch32, revision
`a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`. Native video-conditioned text scoring
is retained. This is an exhaustive feature-cache scorer, not a shared-query
FAISS index. The two development settings are 4s/2s/8frames and 8s/4s/8frames.

Engineering smoke (synthetic pixels; not benchmark quality):

| Measurement | Observed |
|---|---:|
| Parameters | 196,585,729 |
| Video embedding | 512 |
| Extra prompt features/window | 49 x 512 FP32 |
| Cached/native cosine absolute error | 0.0 |
| Model load, cached local weights | 2.364 s |
| One synthetic eight-frame window | 0.470 s |
| One-window text encoding + scoring | 0.069 s |
| Windows process peak working set | 1,110,564,864 bytes |

This single smoke does not measure p95, ingestion throughput, model-download
time, resource variability, source generalization or whole-service deployment.
Host has 16,905,977,856 bytes RAM; Torch2.14.0+cpu, Transformers4.57.6; no GPU.

Five development sources span 48.38 minutes, including the 24.43-minute
*One Week*, 14.64-minute pottery documentary, 4.50-minute street parade,
85-second cycling newsreel and 3.39-minute solar-oven tutorial. All 66 queries
received chronological overview and dedicated dense wording/boundary/repeat
review before retrieval. This is agent visual review, not independent human
sign-off. Reviewed timestamps have approximately one-second precision; the
supplementary pottery repeat audit uses two-second frames. Very brief repeats
between overview samples may remain unobserved.

Historical provenance audit found no source URL/original-credit alias matches
against 87 historical media URLs across 52 tracked files containing URLs.
All media decoded sequentially. Parade decoded duration is 269.931 seconds,
versus 270.08 nominal. Cycling was losslessly remuxed to MKV; both arms receive
the same file. No visual source was transcoded for the candidate.

Development manifest SHA-256:
`c4e3e79ad3fa72e0c07f7ddec2bbadc8a0b8627d691be1d237d6eec954308006`.
Validation sources have not been selected or acquired. All three arms completed once. Selection checksum is recorded in
`reports/temporal-development-selection.sha256`.

Full regression including all 18 temporal tests: **303 passed, 1 skipped**.
Ruff passed. A focused test invocation without a local basetemp encountered a
Windows Temp-directory permission error; subsequent checks use ignored local
basetemp directories. Frontend untouched; no frontend rebuild claimed.

Next: acquire and audit a completely new validation set before its freeze. No promotion or
model-quality recommendation is supported yet.
