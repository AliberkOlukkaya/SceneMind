# Temporal Video Retrieval V1 — work in progress

No development retrieval, frozen validation, architectural decision or historical
replay has run. R@1/3/5/10, MRR@5, category deltas and long-video performance
are **not measured**. The current work is infrastructure and source annotation.
It is not Decision A, B, C, D or E.

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
Validation sources have not been selected or acquired. Development execution
has not started at this checkpoint.

Full regression including all 18 temporal tests: **300 passed, 1 skipped**.
Ruff passed. A focused test invocation without a local basetemp encountered a
Windows Temp-directory permission error; subsequent checks use ignored local
basetemp directories. Frontend untouched; no frontend rebuild claimed.

Next: commit development freeze, run both temporal settings and the exact
production Visual path, then collect a new validation set. No promotion or
model-quality recommendation is supported yet.
