# Personal acceptance failures

Status: **measured; outcome C — NOT YET ACCEPTED** (2026-09-15).

Failure counts are multi-label counts, so one query may appear in more than one category.

| AUTO failure category | Count | Evidence and implication |
| --- | ---: | --- |
| no-match / false-confidence problem | 14 | Fourteen of 18 unsupported/negative searches returned plausible but unrelated moments. A normal user receives no reliable absence signal. |
| unsupported query | 14 | Temporal/action, mixed spoken/visual, or absent-content requests exceeded the current evidence model. |
| Turkish language mismatch | 13 | Turkish cues did not align with the English-trained router, English transcript BM25, or CLIP text-image space consistently. |
| AUTO routing failure | 12 | Every mismatch was Turkish: speech intents often routed to Visual, while several visual UI searches routed to Hybrid. |
| CLIP semantic retrieval failure | 3 | The requested visible content existed, but returned frames did not locate it usefully. |
| ASR transcription failure | 2 | Two English lecture AUTO searches reached a failed transcript state because the supplied tutorial has no audio. |
| UI/product friction | 2 | The API told the user to transcribe after transcription had already failed on an audio-less source. |
| hybrid fusion failure | 2 | Relevant modality evidence did not survive combined ranking for modifier/camera-style mixed requests. |
| speech lexical retrieval failure | 1 | “Status bar” missed because the narrator used “display.” |

The current labels record no separate temporal-sampling, timestamp-granularity, small-object, OCR, or temporal/action-reasoning failure on a positive AUTO query. Unsupported negatives still expose the product’s inability to reason about absence; this should not be mistaken for proof that those capabilities work.

## Product and evidence failures

The original 439,295,727-byte tutorial was rejected with HTTP 413 against the 262,144,000-byte production cap. Preserving its content required an external H.264 transcode:

```text
ffmpeg -y -v error -i <original> -map 0:v:0 -c:v libx264 -preset veryfast -crf 27 -pix_fmt yuv420p -an -movflags +faststart <derived.mp4>
```

The derived file was 187,399,730 bytes; the original was preserved. The operation took about 309 seconds. This is a product workflow failure even though subsequent ingestion succeeded.

The designated lecture is 22:49 and silent, the designated demo is 11:12, and the street video is 35 seconds. They therefore do not supply the requested 30–60 minute audio lecture, 15–60 minute demo, or long-form real-world evidence. No claim about smooth 30–60 minute use can be made. The production 1,800-second duration cap also prevents testing most of the requested upper range without a deliberate product decision.

The production metadata reader reported the street file as 1000 fps while FFmpeg identified the actual stream as about 30 fps. Frame timestamps and duration were correct, so retrieval was not affected, but the displayed/stored metadata is misleading product friction.

The automated browser surface was unavailable in this execution environment. Backend and frontend were run, and production API ingestion/search behavior was exercised end to end, but a fresh click-through of the rendered UI could not be completed. Existing frontend checks still verify the implemented UI behavior; this acceptance report does not claim a new manual UI pass.

## Required next action

Run one focused Turkish compatibility milestone over source-disjoint calibration material, covering router features and the existing CLIP/BM25/RRF retrieval paths without adding a new model first. Keep this acceptance manifest frozen and repeat it only after the candidate change passes its own held-out gate. In parallel product planning, define an honest unsupported-query response and a deliberate 30–60 minute upload policy. Final acceptance then needs new, valid-duration audio media; those new videos must receive their own labels before any retrieval output is viewed.

Do not promote a change from this acceptance set alone. Do not claim v1.0 acceptance until the repeat and valid long-form run pass the frozen gates.
