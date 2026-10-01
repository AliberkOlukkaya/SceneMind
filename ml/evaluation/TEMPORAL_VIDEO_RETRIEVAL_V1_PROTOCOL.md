# Temporal Video Retrieval Backbone V1

Status: predeclared, before development retrieval. Production baseline commit:
`8115baebcc1b72546d601ab624b37e99f2a55cb7`. This is an isolated V2 experiment,
not a production migration. No decision or quality measurements exist yet.

## Candidate selection

One candidate: `microsoft/xclip-base-patch32`, revision
`a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`, MIT model-card license.
The established Transformers implementation supplies cross-frame attention and
a multiframe integration transformer, eight RGB frames at 224 square pixels,
512-dimensional video features, and a video-conditioned text prompt generator.
The checkpoint was supervised on Kinetics-400. Parameter count and CPU memory
will be measured, not inferred from its rounded model-card size. FP32 CPU,
four Torch threads, batch one for video, no training or augmentation. GPU is
optional but not used for the primary result. Download and cold-load costs are
separate from steady inference. Weight memory alone is not peak RSS.

**Preserve the trained score.** Plain `get_video_features`/`get_text_features`
cosine omits X-CLIP's prompt generator. Instead cache the normalized video
vector AND its 49 x 512 projected patch features; at query time apply the
original prompt generator and normalize the resulting text vector per window.
Rank by exact cosine, excluding the positive constant logit scale (which does
not affect order). Compare this cached path against native `forward` before
evaluating media. This is an exhaustive, video-conditioned scorer over a
persistent feature index, **not** one shared text vector queried with FAISS.
Report this architectural and query-cost limitation explicitly. It remains a
bounded pretrained temporal representation, without a learned extra reranker.

Research exclusions, not additional benchmark candidates: ViCLIP-B-16 has a
well-matched joint spatiotemporal dual encoder and an official public checkpoint,
but its HF B checkpoint currently has neither a license file nor license
metadata; the upstream code's Apache license does not settle checkpoint terms.
MobileViCLIP has a newer, less mature standalone implementation and additional
base-weight licensing questions. Neither is downloaded or evaluated. No model
selection will use any SceneMind holdout results.

Primary sources (checked 2026-09-28):
- [X-CLIP model card](https://huggingface.co/microsoft/xclip-base-patch32)
- [Transformers X-CLIP implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/x_clip/modeling_x_clip.py)
- [Original X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP)
- [ViCLIP architecture and official checkpoint links](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)
- [ViCLIP-B checkpoint](https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf/tree/8484a9cb5b1b86e43c3ded53abe7485f52d8b789)
- [MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)

## Development and freeze

Only two configurations: 4-second windows / 2-second stride / 8 frames and
8-second windows / 4-second stride / 8 frames. Windows start at zero, then
integer stride multiples; stop after the first window reaching the end.
The last window is truncated, not extended past the source. Short videos use
one window; sampling uses eight equal-bin midpoints. Nearest decoded frame
timestamps are retained; repeated frames in very short clips are disclosed.
No single frame is represented as temporal evidence. Center-crop preprocessing
is the pinned processor's native transform. No scene segmentation or fusion.

Development requires >=5 new diverse sources, >=60 visual positives, >=15
HUMAN_ACTION/TEMPORAL_EVENT combined and >=10 OBJECT_INTERACTION, all seven
categories, and a >=20-minute source. Select maximum development R@5, then
combined action R@5, then lower measured indexing time as deterministic ties.
Collect validation only after that selection: >=5 further source-disjoint
videos, >=60 positives, all categories and a >=30-minute source. At least
eight positives per source. Source provenance and hashes must be checked
against all previous benchmark manifests, not merely filename differences.
Independence from unknown model training footage cannot be guaranteed.

Annotations precede retrieval. Audit wording, every valid occurrence, boundaries,
ambiguity and visual sufficiency over the source timeline. Review dense frame
sequences for motion, not only contact sheets. Record actual review coverage,
reviewer identity/type and limitations; agent review is not independent human
sign-off. Ambiguous or incompletely reviewed queries cannot enter final freeze.
No truth derived from ranked output. Never narrow a repeated-event query to one
convenient occurrence. Audit baseline sampled-frame visibility and candidate
sampled-sequence sufficiency independently of rank where feasible.

Freeze selected config, source IDs/licenses/media hashes, queries, categories,
difficulty, ALL valid intervals, annotation audits, code and dependency hashes,
both model revisions, preprocessing, index, normalization and metric definitions.
Commit the checksum before the single validation run. Refuse a second run and
any freeze/code/media mismatch. A failed/incomplete attempt remains disclosed.
No tuning on validation. V1 final checksum stays
`d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196`.

## Comparison and metrics

Baseline: unchanged production ingest, visual index and Visual endpoint at
5 seconds / 480-pixel JPEGs, pinned production CLIP, uncalibrated exact FAISS.
Use isolated runtime directories, never production indexes. Same queries and
source bytes. Return Top-10, no temporal suppression or neighbor expansion.

Primary timestamp criterion is the displayed/clicked timestamp: baseline frame
timestamp; candidate window midpoint. It must fall in ANY acceptable interval.
No extra tolerance, no credit for a long window merely overlapping an event.
Also report candidate window overlap only as a secondary diagnostic, never as
primary success. Preserve full ranked windows and actual sampling times.
R@1/3/5/10 and MRR@5 (zero when first relevant rank >5), per category, source,
long video and combined action/static groups. Percentage-point deltas use the
same query denominator. Categories: STATIC_OBJECT, SCENE, HUMAN_ACTION,
OBJECT_INTERACTION, TEMPORAL_EVENT, SMALL_OBJECT_DETAIL,
MULTIPLE_SIMILAR_MOMENTS.

For every Top-5 miss, visually review sampled evidence: timestamp overlap alone
does not prove evidence exists. Separate SAMPLING_MISS and
TEMPORAL_COVERAGE_MISS from semantic/ranking failures. Other primary codes:
STATIC_SEMANTIC_MISS, ACTION_SEMANTIC_MISS, OBJECT_RELATION_MISS, SMALL_OBJECT,
MULTIPLE_SIMILAR_MOMENTS, BACKGROUND_BIAS, QUERY_AMBIGUITY, ANNOTATION_ISSUE,
OTHER. Do not repair frozen truth after seeing rankings.

Report measured load time, preprocessing and embedding time/source-minute,
clips and processed frames/source-minute, on-disk index bytes/source-minute,
peak absolute process-tree RSS, and query median/p95 (including load of the
persisted per-video index, excluding one-time model load, no warmed query cache).
Separate cold and warm costs. Report machine/threads and distinguish a local
measurement from validation of a rented 4-vCPU/8-GB deployment.

## Gates and decision

All A gates: validation R@5 >=80%; delta >=10pp; combined HUMAN_ACTION +
OBJECT_INTERACTION + TEMPORAL_EVENT R@5 >=75% AND delta >=10pp (predeclared
meaning of material); STATIC_OBJECT + SCENE regression <=5pp; no source with
>=8 queries below60%; plausible resources. Engineering guide for an isolated
4-thread CPU process: RSS <=6 GiB, indexing <=120 s/source-minute, and query
p95 <=2 s on the >=30-minute source. Meeting these budgets is provisional CPU
practicality, not proof of whole-service concurrency on 8 GB. Exceeding budgets
requires an explicit feasibility decision, never silent threshold relaxation.

A: all gates pass; mark V2_VISUAL_CANDIDATE only, integration separate.
B: positive generalization gain insufficient. C: no meaningful generalization.
D: quality passes but resources unacceptable. E: actual implementation/model
availability blocker prevents a fair experiment. Missing evidence is N/A, never
zero or a fabricated rejection. B/C/D stop further models and tuning.
Historical V1 visual replay only after the new decision is frozen, clearly
diagnostic, never grounds for selection or promotion. Production, speech, Smart,
RRF60, OCR, Q&A and disabled Ask Video remain untouched. No release-tag changes.
