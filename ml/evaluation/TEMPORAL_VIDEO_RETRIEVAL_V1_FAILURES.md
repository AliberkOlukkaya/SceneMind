# Temporal Video Retrieval V1 — final failure record

Decision C is frozen in commit `6796df1`, before historical replay.
Validation: CLIP misses 26/66; X-CLIP misses 49/66 under the same frozen
timestamp criterion. All validation misses received actual-pixel review.
No intervals, queries, scores or parameters were changed after evaluation.

## Development baseline (retained diagnostic)
The production CLIP baseline missed 22/66 under frozen timestamp labels.
All 22 misses were reviewed against actual production JPEGs, including the
frames around every annotated interval and top-ranked images:

- 5 SAMPLING_MISS: short hoop action, two bar-scene queries, crop-field cutaway,
  and the girl's approach to the picnic table have no relevant sampled JPEG.
- 14 evidence-present semantic/relation/action misses: relevant image evidence
  exists but ranks below Top-5. One train-crash example preserves the passing
  train and wreckage but not the exact impact; evidence is partial.
- 3 ANNOTATION_ISSUE: useful sticker-contact Top-1 (225.051s), chocolate Top-2
  (175.189s), and bite Top-1 (190.204s) fall just beyond integer-valued frozen
  endpoints. These are not representation failures. Raw scores remain unchanged;
  no labels were repaired after retrieval. This is a material precision limitation
  for interpreting small development gains. Future independent annotations need
  subsecond transition inspection before freezing, without changing the metric.

Per-query observations and review-sheet checksums are in
`reports/temporal-development-baseline-failures.json`. They are agent visual
review, not independent human sign-off. The development candidate raw rankings
are retained; its full per-miss visual audit was not performed. Final validation
candidate misses are all audited below.

Resolved pre-freeze engineering issues: use the pinned X-CLIP image processor
explicitly and request return_dict=True from multiframe integration. Synthetic
cached/native cosine matches exactly. Sequential decode established parade's
269.931-second visual duration instead of the 270.08-second nominal container
length. Cycling was losslessly remuxed from OGV to shared MKV; neither arm uses
a different visual transcode. No production implementation changed.

The V1 observed acceptance remains frozen; the final diagnostic replay is
reported separately after the new decision, never used for model selection.

## Frozen validation: baseline pixel audit

Production CLIP misses 26 of 66 queries at Top-5. All 26 were reviewed against
the actual indexed JPEGs near the annotated occurrences and the top-ranked
results. Twelve are **SAMPLING_MISS**, fourteen have sampled evidence present
but a semantic/action/relation ranking miss. The latter comprise seven
OBJECT_RELATION_MISS, four ACTION_SEMANTIC_MISS, two STATIC_SEMANTIC_MISS and
one SMALL_OBJECT. These are agent judgments, not independent human labels.
The two causes are both material; semantic failures slightly outnumber sampling.

Examples of genuinely absent sampled actions: water poured into the frying pan,
lifting its lid, tipping and lifting the wooden stool, stair ascent, drinking
from a bottle and clinking glasses. A frame showing the state after an event
does not establish that the action was sampled. Query tval-032 is especially
important: the 1061.066-second JPEG falls inside its frozen interval, but the
tray is already inside the refrigerator. Interval overlap would incorrectly
classify that as visible insertion evidence. Its raw metric is not changed.

Evidence-present misses include dough cutting, pouring broth through a sieve,
lining a steamer with cabbage, the yellow square wrapper, and the bride's hug.
Search commonly ranks another view of the same actor/object instead of the
specified interaction. A tree-fall still provides weaker motion evidence than
the reviewed sequence; visibility is not proof that an image encoder can infer
the action. Baseline visibility refers to the indexed JPEG before native model
preprocessing. Model cropping and representational limitations remain within
the evidence-present failure group.

Per-query observations, evidence timestamps and sheet checksums:
`reports/temporal-validation-baseline-failures.json`. Frozen intervals, wording,
categories and ranked outputs are unchanged. Near-event useful clicks can still
fail the predeclared strict timestamp metric; no post-hoc tolerance is added.

## Measurement incident

During baseline model startup the outer process-tree RSS sampler terminated
with Windows OSError 1455 (paging file too small). All five ingests and all 66
queries completed and the durable run status is complete, but the shell reported
exit 1 with that thread exception on stderr. The reported whole-run RSS
maximum is **invalid/incomplete**, not a valid low-memory result. Per-video
samplers continued; those observations are reported separately. The run was
not repeated. Host RAM was shared with other workstation activity, so timing
is a local observation, not a dedicated 4-vCPU/8-GB server certification.

## Interrupted candidate process and bounded recovery

On resumption the original candidate Python process and tool session were
absent. Its last durable record contained river, dumplings and log-stool
indexes plus all 44 queries for those sources. Wedding ingestion had logged
200/292 clips but had not persisted its index or executed any wedding queries;
fitness had not begun. The exact process-exit cause is unknown.

`scripts/temporal_resume.py` verifies the original manifest/code/dependency/media
freeze, snapshots `interrupted-run.json`, retains the original `started.json`,
and creates an exclusive recovery marker. It refuses partially queried sources
and completed runs. Only the two wholly unqueried sources are processed; the
44 completed query records are preserved exactly. The new orchestration does
not change the encoder, sampler, score, ground truth or primary metric.

Unpersisted partial wedding ingestion was lost. Its cost and the first process's
whole-run RSS cannot be recovered, so full-attempt ingestion cost and whole-run
candidate RSS are N/A. Completed-source indexing times and resumed-stage memory
observations remain separately reportable. This is an interrupted execution
with disclosed checkpoint recovery, not an uninterrupted single-process run
or a second pass over observed query outcomes. No query is selectively rerun.

## Frozen validation: candidate pixel audit

All **49** Top-5 misses were inspected with the actual eight sampled frames
and the returned Top-1 sequence. Saved index sample timestamps were checked
against the decoded frames. Raw images and center-crop geometry were reviewed;
the crop preview does not reproduce exact processor interpolation/normalization.
Relevant evidence is present in at least one reviewed sequence for all 49,
although the departing wedding car and some tray/bottle crops are partial.
Evidence visibility alone is not proof that an encoder can infer an action.

- **2 temporal timestamp-coverage misses**: tval-051 (tipping the stool) and
  tval-065 (picking up an orange). Neither frozen interval contains a candidate
  midpoint on the two-second grid. The action pixels were sampled. For the
  orange, Top-1 is already the useful action window, but its 84s midpoint is
  outside 84.2–85.6s. Do not call this a semantic error or absent pixels.
- **46 semantic/ranking/localization misses with evidence present**: 23 object
  relations, 10 actions, 8 static concepts, 3 small-object/detail cases, 1
  background bias and 1 boundary-context localization. tval-004's Top-1 at
  818s is just before the 819s interval; the valid 820s window ranks eighth.
  This boundary case makes 45 clearer semantic cases plus one localization case.
- **1 ambiguity case**: tval-001. The annotated rolling-bale sequence exists,
  but another bale/conveyor scene returned first may also fit the query.
  Alternate-interval completeness is uncertain; its frozen raw miss remains.

The counts above partition 49 raw misses; they do not replace benchmark labels.
Examples of sampled but badly ranked evidence include hanging garlands, folding
a henna cone, stretching on a railing, jumping rope, lifting a pan lid and
putting a tray into the refrigerator. Many wedding queries collapse onto the
same group-dancing or interview window. Several fitness queries collapse onto
a fist bump. This is broader than merely failing to represent motion: static
object R@5 is zero (0/9), and scene R@5 is only 2/6.

For tval-047, seven adjacent context sheets were reviewed because the first
representative sequence was insufficient. The decorated car becomes a small,
partly occluded receding rear with tail lights behind foreground guests. A
single inspected window would have incorrectly suggested absent evidence.

Per-query evidence, exact sampled timestamps, failure labels, first relevant
ranks and reviewed-sheet checksums are in
`reports/temporal-validation-candidate-failures.json`. Agent interpretation is
not independent human review. Contact sheets can miss brief between-frame
occurrences, and no claim of complete continuous-playback annotation is made.

## Diagnosis and limits

Baseline failures are mixed: **12 sampling versus 14 evidence-present ranking**.
Candidate failures are dominated by ranking/representation, not missing pixels;
only two have no representable midpoint. Denser sampled evidence did not create
useful retrieval with this pretrained model. The cache implementation agrees
with native X-CLIP forward on real development clips within 4.48e-8. This checks
implementation fidelity, not the suitability of the pretrained representation.
Training-domain mismatch is a plausible explanation, not a measured causal result.

Even giving every candidate raw miss the most favorable treatment for the two
coverage cases, one ambiguity and one boundary-context case would produce only
21/66 = 31.82% R@5, far below CLIP's unchanged 60.61%. This is an upper-bound
sensitivity illustration, not an alternate benchmark or annotation repair.
Stop this candidate branch. No threshold, prompt, window, fusion or model
follow-up is authorized by these results. Production remains V1.

## Historical diagnostic after decision freeze

Six short historical visual sources / 42 queries were replayed only after
Decision C was committed. The paired subset drops from 31/42 V1 CLIP hits to
15/42 X-CLIP hits at Top-5. This is diagnostic, not a second holdout for
selection. Detailed per-query rankings and reviewed-sheet checksums are in
`reports/temporal-historical-diagnostic.json`.

- **Aikido**: raw Top-5 falls from 3/8 to 1/8. The candidate repeatedly ranks
  a 166–169s pin for kneeling, resetting and later throw prompts. Reviewed
  annotated windows at 6, 32, 72, 126, 132 and 244s show kneeling, a throw,
  a pin, resetting, a downed partner and three artists respectively. Several
  similar pins and downed-partner moments are repeated, however, so the old
  single intervals can mark useful alternate clicks wrong. The upside-down
  aspect of F069 was not established by the one reviewed sequence; no stronger
  pixel claim is made. F119's technical arm-lock identity is uncertain.
- **Manchester**: raw Top-5 falls from 8/8 to 3/8. Candidate Top-1 often lands
  at 28–30s. The annotated children near 72/78s and later tram near 96s are
  clearly sampled but ranked poorly. For broad opening street/tram queries,
  the 28s scene can itself be visually relevant; these old labels may omit
  alternatives. Those raw misses remain unchanged, so the precise 62.5 pp
  regression is not a certified user-success delta.
- **OpenRefine/PPE**: the opening title, reconciliation dialog, yellow gown,
  respirator fitting and hand sanitizer before PPE have sampled evidence but
  the candidate often returns related yet different screens/actions. The
  OpenRefine facet on a side panel is partly lost in the model's center crop.

These findings agree directionally with the new source-disjoint validation;
only that new validation determined the decision. Agent review was selective
and no independent human annotation sign-off exists.
