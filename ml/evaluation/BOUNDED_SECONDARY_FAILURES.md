# Bounded secondary sampling failure analysis

The branch fails before detector capacity becomes the deciding factor.

## Verified-visible events

| Event | Selected-window result | Primary failure |
| --- | --- | --- |
| Dirt bicycle jump | No reviewed visible 2-second frame enters top-5 ±2-second windows | Coarse CLIP never selects the correct region |
| Bottle insertion | Window reaches visible evidence and secondary CLIP retrieves it | Success before abstention |
| Early throwball rally | Window reaches visible evidence; secondary CLIP top five misses it on held-out | Semantic ranking miss among similar sports frames |
| Late throwball rally | Selected windows contain no reviewed visible target frame | Coarse region/window miss |

The maximum ±6/top-20 policy reaches all four events, but it approaches global dense work and secondary CLIP still misses two. Larger windows therefore mask candidate-generation weakness without solving semantic ranking.

## Benchmark buckets

- **Coarse region absent or window too narrow:** directly observed on two of four visible events under the selected policy.
- **Secondary frame still misses the object:** retained as a possible frame-level failure, but the reviewed event analysis separates it from interval overlap; it is not counted without a visibility label.
- **Detector misses a visible object:** established by the preceding Nano-640 ablation, but not dominant here because the selected fusion weight is zero.
- **Detector class unsupported:** safely falls back to coarse results; no unsupported query is forced through YOLOX.
- **Semantic mismatch despite an object hit:** object confidence raises calibration-negative scores and every positive detector weight loses to zero.
- **Relationship unsupported:** boxes prove class presence, not holding/riding/near relationships; final relationship R@5 is 50%.
- **Action needs multiple frames:** action/temporal queries remain on the coarse fallback; this experiment adds no action inference.
- **Duplicate candidates:** six-second spacing cuts secondary frames by about one third and improves calibration R@1, but costs 1.85 points R@5.
- **Threshold/calibration failure:** the <=10% calibration FAR threshold causes 81.0% overall and 100% small-object held-out false abstention.

## Next capability

Improve coarse candidate generation before adding another detector. The next experiment should retrieve temporally diverse regions using more than raw top-score greediness while preserving a bounded query budget. It should explicitly optimize coverage of brief visible evidence and prevent many visually similar moments from monopolizing the candidate set. Keep object detection as optional supporting evidence after candidate recall is repaired. This result does not justify OCR, action recognition, tracking, a larger VLM or Video RAG.
