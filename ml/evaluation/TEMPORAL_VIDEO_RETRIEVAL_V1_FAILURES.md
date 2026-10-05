# Temporal Video Retrieval V1 ? development failure record

The development diagnostics below are retained unchanged. Validation baseline
has now completed once; the candidate is still running. No final decision yet.
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
review, not independent human sign-off. Candidate failures are pending.

Resolved pre-freeze engineering issues: use the pinned X-CLIP image processor
explicitly and request return_dict=True from multiframe integration. Synthetic
cached/native cosine matches exactly. Sequential decode established parade's
269.931-second visual duration instead of the 270.08-second nominal container
length. Cycling was losslessly remuxed from OGV to shared MKV; neither arm uses
a different visual transcode. No production implementation changed.

The V1 observed acceptance remains frozen and has not been replayed.

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
