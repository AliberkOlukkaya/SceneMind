# Frozen small-object failure inventory

This inventory was frozen before detector selection and inference. It derives from the unchanged Natural Video Benchmark V2 manifest and the committed lightweight pair-scorer report. The machine-readable record is [`failure_inventory_v1.json`](failure_inventory_v1.json).

| Query | Slice membership | CLIP top-5 | Exact candidate review | Detector implication |
| --- | --- | --- | --- | --- |
| `street-object-bicycle` | OBJECT, SMALL_OBJECT | Interval-relevant frames at ranks 1, 2 and 4 | Bicycle is visibly present at 25.0 s (about 3% of frame, partly cropped) and 15.0 s (about 1%); it is no longer visible at the interval-relevant 30.0 s sample | A COCO bicycle detector can provide valid presence evidence on two candidates. |
| `throw-object-ball` | OBJECT, SMALL_OBJECT | Interval-relevant frame at rank 1 | The exact 5.007 s candidate JPEG crops the hand below the frame and contains no visible yellow ball | A detector cannot recover evidence absent from the sampled frame. The frozen interval and sampled visual evidence disagree at this timestamp. |
| `throw-compositional-holding` | SMALL_OBJECT, RELATIONSHIP, COMPOSITIONAL | Interval-relevant frame at rank 1 | Person is visible at 5.007 s; ball and hand are outside the sampled JPEG | Presence requires person plus sports ball, and holding requires a spatial heuristic. Neither can succeed on this candidate from boxes alone. |

All three failures have a CLIP result inside the frozen interval, which gives raw small-object Recall@5 of 100%. Only one of the three queries has a manually confirmed frame in its top five that visibly contains every requested object. This distinction prevents interval-based recall from being mistaken for detector-addressable evidence.

The two object categories are standard COCO classes (`bicycle` and `sports ball`). This supports testing a small closed-set detector first. The broader product still accepts vocabulary outside COCO, so the detector branch must remain optional and preserve CLIP fallback.

No benchmark interval or query changed during this review. Approximate area ratios are descriptive measurements and are not detector labels or new evaluation ground truth.
