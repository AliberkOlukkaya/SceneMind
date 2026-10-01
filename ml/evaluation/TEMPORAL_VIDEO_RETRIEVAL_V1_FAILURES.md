# Temporal Video Retrieval V1 — failure record

Retrieval failure counts are not available: no annotated retrieval evaluation
has run. Do not interpret missing counts as zero failures.

Engineering issues found and resolved before any quality run:
- Transformers4.57.6's combined processor call with `videos=` produced no
  pixel tensor. Use its pinned image processor and tokenizer explicitly.
- The multiframe integration module defaults to tuple output. Request
  `return_dict=True` explicitly. Native/cached synthetic cosine now matches.

Data-preparation issue, still under review: the street-parade container's
reported duration is 270.08s, but seeking to 270.00s yielded no decodable frame.
The overview review was completed through 268s. Verify actual media timestamps
and final-frame handling before freezing; do not paper over decode failures
with fabricated/repeated evidence. Cycling's OGV container also needs a shared,
lossless remux to a production-supported container before either arm runs.

Annotation limitations: overview contact sheets do not by themselves verify
motion, precise boundaries or all repeated occurrences. The provisional query
draft is ineligible for retrieval. Dense review and source-exclusion audit are
pending; no independent human reviewer has signed off.

Final analysis will classify each baseline/candidate Top-5 miss using the
predeclared sampling, coverage, semantic, relation, small-object, repeated-moment,
background, ambiguity and annotation taxonomy. The V1 observed benchmark remains
frozen and has not been replayed during development.
