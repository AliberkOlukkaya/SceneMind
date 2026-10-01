# Temporal Video Retrieval V1 ? development failure record

Final validation has not run. These are development diagnostics, not a decision.
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
