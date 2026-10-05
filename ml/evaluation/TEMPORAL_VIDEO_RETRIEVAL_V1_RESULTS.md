# Temporal Video Retrieval Backbone V1 — Decision C

The frozen candidate reaches 17/66 R@5 versus 40/66 for production CLIP, a 34.85 percentage-point regression. All five sources fail the 60% source floor; temporal/action and static gates also fail. No generalizable benefit is demonstrated. Stop this model branch and do not tune on observed validation.

**Do not replace production CLIP. Stop further models/tuning in this milestone.**
This is a rejection of the tested X-CLIP configuration for SceneMind, not a proof
that every temporal architecture is ineffective. No production promotion, V2
integration or release tag was created. Ask Video remains disabled.

## Frozen comparison

| Validation: 66 queries | R@1 | R@3 | R@5 | R@10 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Production CLIP, 5s frames | 36.36% | 51.52% | 60.61% | 68.18% | 45.73% |
| X-CLIP, 4s / stride 2s / 8 frames | 13.64% | 21.21% | 25.76% | 37.88% | 17.90% |

Candidate R@5 delta: **-34.85 percentage points**.
The primary criterion is the displayed timestamp (CLIP frame / X-CLIP window
midpoint) inside any frozen valid interval. Window overlap alone earns no credit.

| Query category | n | CLIP R@5 | X-CLIP R@5 | Delta, pp |
|---|---:|---:|---:|---:|
| STATIC_OBJECT | 9 | 77.78% | 0.00% | -77.78 |
| SCENE | 6 | 83.33% | 33.33% | -50.00 |
| HUMAN_ACTION | 13 | 69.23% | 46.15% | -23.08 |
| OBJECT_INTERACTION | 19 | 47.37% | 15.79% | -31.58 |
| TEMPORAL_EVENT | 10 | 40.00% | 20.00% | -20.00 |
| SMALL_OBJECT_DETAIL | 6 | 66.67% | 50.00% | -16.67 |
| MULTIPLE_SIMILAR_MOMENTS | 3 | 66.67% | 33.33% | -33.33 |
| combined_temporal | 42 | 52.38% | 26.19% | -26.19 |
| combined_static | 15 | 80.00% | 13.33% | -66.67 |

Combined temporal = human action + object interaction + temporal event;
combined static = static object + scene. These small category samples are
descriptive; no population-level confidence claim is made.

| Validation source | Minutes | Queries | CLIP R@5 | X-CLIP R@5 |
|---|---:|---:|---:|---:|
| river | 31.931 | 18 | 77.78% | 33.33% |
| dumplings | 23.506 | 18 | 38.89% | 16.67% |
| log-stool | 2.941 | 8 | 25.00% | 37.50% |
| wedding | 9.761 | 12 | 83.33% | 8.33% |
| fitness | 1.893 | 10 | 70.00% | 40.00% |

Candidate catastrophic sources (at least eight queries, R@5 below 60%): dumplings, fitness, log-stool, river, wedding.

## Resources and long video

| Measurement | CLIP | X-CLIP |
|---|---:|---:|
| Model load, seconds | 31.766 | 10.241 |
| Preprocessing, seconds/source-minute | 0.848 | 17.078 |
| Total indexing, seconds/source-minute | 1.578 | 40.384 |
| Embedding-stage frames/second | 16.462 | 10.290 |
| Processed frames | 842.000 | 16,784.000 |
| Processed frames/source-minute | 12.023 | 239.662 |
| Clips/source-minute | 0.000 | 29.958 |
| Persisted index bytes/source-minute | 24,632.449 | 3,090,610.089 |
| Query median, seconds | 0.025 | 0.938 |
| Query p95, seconds | 0.047 | 1.493 |
| Whole-run process-tree peak RSS, bytes | N/A — sampler failed | N/A — sampler failed |

The longest source is **river, 31.931 minutes**.
CLIP indexes 383 JPEGs in 50.046s, storing 784,512 feature bytes; X-CLIP indexes 957 clips / 7,656 processed frames in 1057.236s, storing 98,747,922 bytes.
Per-video indexing RSS observations: CLIP 1,006,866,432 bytes, X-CLIP 1,174,552,576 bytes.
Long-video query median/p95: CLIP 0.022/0.065s; X-CLIP 1.056/1.492s.

CPU FP32, four Torch threads, Torch 2.14.0+cpu, Transformers 4.57.6;
Windows workstation with 16,905,977,856 bytes RAM; no GPU benchmark.
CPU execution is possible but slower and substantially larger than the
baseline. This is not a dedicated 4-vCPU/8-GB deployment certification.
The workstation experienced memory pressure from shared activity.

Baseline outer RSS sampling failed with Windows error 1455; its incomplete
whole-run number is deliberately N/A. Per-video samplers continued. All
five sources and 66 queries completed, but the shell reported exit 1 with
the sampler exception. No second baseline attempt was made. Candidate
execution was interrupted after 44 queries. Those exact rows and three
indexes were preserved; only the two wholly unqueried sources were finished
after verifying the unchanged freeze. Partial unpersisted wedding ingestion
was lost. Full-attempt candidate ingestion cost and whole-run RSS are N/A;
table indexing throughput sums completed stages only and excludes that
unknown lost work. Additional recovery model load is recorded in JSON.
Lightweight evidence-sheet decoding overlapped later ingestion stages;
these shared-workstation times may include CPU/I/O contention.

Load time is first model initialization in each process using already cached
weights, including import overhead; download time is excluded. Query times
include persisted index reload, with the model resident. Query-result caches
are not used; OS disk cache is uncontrolled. Candidate preprocessing is
decode time; image normalization is inside its embedding stage. Baseline
preprocessing includes production ingest and JPEG extraction. Index sizes
exclude media/JPEGs: baseline embeddings.npy versus the complete candidate
NPZ vectors, prompt patches and metadata. Processed candidate frames count
overlap repeats, not unique source frames.

## Promotion gates

| Gate | Pass? |
|---|---|
| overall_r5_at_least_80 | FAIL |
| overall_gain_at_least_10pp | FAIL |
| temporal_r5_at_least_75 | FAIL |
| temporal_gain_at_least_10pp | FAIL |
| static_regression_at_most_5pp | FAIL |
| no_catastrophic_video | FAIL |
| candidate_rss_at_most_6_gib | N/A |
| indexing_at_most_120_seconds_per_source_minute | N/A |
| long_video_query_p95_at_most_2_seconds | PASS |

All quality gates and resource feasibility are required for A. A would
authorize only a later V2 integration milestone, not direct replacement.
This run does not authorize that milestone. No further model, threshold,
sampling, prompt or fusion tuning is permitted on the observed sets.

## Candidate and development selection

`microsoft/xclip-base-patch32`, X-CLIP cross-frame vision transformer with
multiframe integration and video-conditioned text prompts. Revision:
`a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`; MIT model card; implementation
Transformers 4.57.6, Apache-2.0. Measured 196,585,729 parameters; 512 video
dimensions plus 49 x 512 FP32 prompt-patch features per window. Eight
ordered RGB frames, native resize/center-crop 224, ImageNet normalization;
paired tokenizer, at most 77 tokens. Kinetics-400 pretrained, no fine-tuning.
Native conditioned cosine is retained, so scoring is exhaustive over a
per-video feature cache, not a shared-text FAISS nearest-neighbor query.

Selected before quality testing for a pinned available checkpoint, explicit
license, established local runtime and CPU support. GPU placement is
supported by the framework but unmeasured. ViCLIP-B and MobileViCLIP were
considered; unresolved checkpoint/base-weight license/runtime questions
excluded them before downloads or experiments. Only X-CLIP was tested.

Cached/native scoring matched exactly on the synthetic smoke and within
4.48e-8 on one middle development window from each source using fixed
non-benchmark text. This rules out that cache decomposition as the cause
of the large observed regression; it does not certify model generalization.

Development: five new sources, 48.384 minutes, 66 queries across all seven
categories: One Week (24.43 min), pottery (14.64), parade (4.50), cycling
(1.42), solar oven (3.39). Two predeclared configurations only:

| Development | R@1 | R@3 | R@5 | R@10 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Production CLIP | 48.48% | 63.64% | 66.67% | 75.76% | 55.98% |
| X-CLIP 4s/2s | 13.64% | 25.76% | 30.30% | 39.39% | 19.60% |
| X-CLIP 8s/4s | 9.09% | 15.15% | 24.24% | 39.39% | 13.81% |

The fixed maximum-R@5 rule selected **4s window / 2s stride / eight frames**
before validation source research. No validation-based configuration choice.

## Data provenance, annotation and freeze

Validation uses five new licensed/public-domain sources totaling 70.032
minutes: The River (US government public domain), Homemade Dumplings 3 Ways
(CC BY 3.0), log-stool woodworking (CC BY-SA 4.0), Bandhan wedding film
(CC BY 3.0), and Sante et Bien-etre fitness (CC BY-SA 4.0). Canonical
source/original-credit aliases and media hashes were excluded against all
development sources and 87 historical URLs from 52 tracked files. One
initial pasta nomination was rejected for historical overlap before use.
Unknown pretraining-data overlap cannot be excluded.

Before retrieval the agent reviewed 191 chronological and targeted contact
sheets, including dense subsecond transitions where needed, wording,
boundaries, repeated occurrences, ambiguity and visual sufficiency. This
was actual pixel inspection, not continuous playback or independent human
sign-off. Brief occurrences between overview frames can remain unobserved.
Validation difficulty was recorded uniformly as medium; difficulty-stratum
generalization is not supported. The 66 English queries do not evaluate
Turkish. Labels were never repaired after retrieval.

Validation manifest SHA-256:
`54f006a26f23e50a4e213182ab3b6bb1402baa6da5e1457c187ff74c6b7f462e`.
Manifest and all source/model/code/dependency hashes were committed and
pushed in `b729546` before either validation arm. Both arms used identical
source bytes. Each query was evaluated once per arm; candidate execution
spanned the original process and a disclosed checkpoint recovery. Raw
complete rankings, metrics and resource
observations are in `reports/temporal-validation-baseline.json` and
`reports/temporal-validation-candidate-4.json`.

Development manifest SHA-256:
`c4e3e79ad3fa72e0c07f7ddec2bbadc8a0b8627d691be1d237d6eec954308006`.
Development selection SHA-256:
`f153155c82d968c97cf2a9502b52ec23d6d2a272a1740288f3cb51c0f8c06f23`.
Protected V1 final manifest remains:
`d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196`.

## Failure analysis and historical replay

See `TEMPORAL_VIDEO_RETRIEVAL_V1_FAILURES.md` for every missed query and
sampling-versus-ranking diagnosis. Decision is frozen separately in
`reports/temporal-validation-decision.json` before any historical replay.
Historical results are diagnostic only and cannot change this decision.

Historical replay and final checks are pending; no completion claim yet.
