# Object detector failure analysis

The frozen preselection review is in [`../experiments/object_detector_branch/FAILURE_INVENTORY.md`](../experiments/object_detector_branch/FAILURE_INVENTORY.md). The real run confirms that detector accuracy and frame sampling are separate constraints.

| Bucket | Representative evidence | Consequence |
| --- | --- | --- |
| Detector missed small object | `street-object-bicycle`: YOLOX-Nano finds a bicycle at 15.0 s with 0.6085 confidence and 1.14% box area, but misses the larger, partly cropped rank-1 bicycle at 25.0 s. | Even the best bicycle evidence is below the 0.8569 calibration threshold; the query abstains. |
| Object too small / low confidence | `throw-object-ball`: the only sports-ball detection is 0.0174 at 10.007 s with 0.61% area, outside the relevant interval. | The evidence floor retains it for diagnosis, but it is far below any safe presence threshold. |
| Frame sampling missed evidence | The exact rank-1 5.007 s JPEG for both ball queries contains the man but crops his hand and yellow ball below the frame. | CLIP receives interval credit, while no detector can prove ball presence in that candidate. This is a benchmark sampling/interval mismatch, not a detector false negative on a visible ball. |
| Relationship cannot be inferred from boxes | `throw-compositional-holding` requires person, sports ball and holding. The relevant candidate has person detections but no ball; the weak ball detection appears only at 10.007 s. | Presence fails before a proximity heuristic could run. No holding claim is possible. |
| Wrong/general alias mapping | `street-negative-swimming`, `talk-negative-gradient` and `talk-negative-cooking` route through `person`/`speaker` evidence and pass the threshold despite lacking the requested event or topic. | Detector-triggered held-out FAR is 37.5%. A noun hit alone is insufficient for semantic negatives. The frozen router was not changed after seeing held-out results. |
| Unsupported class or attribute | `yellow`, `black shirt`, `podium`, `projected slide`, `riding` and `holding` are recorded as unsupported. COCO detection can confirm a person, bus or generic ball, but not these attributes and relations. | The closed-set branch is evidence only and cannot act as a complete query verifier. |
| Detector finds an object but CLIP candidate is wrong | The 10.007 s ball detection is outside the ball and holding intervals. | Detection cannot create a correct temporal candidate after CLIP/sampling misses it. |
| Multiple similar objects | No held-out failure was attributable solely to choosing among multiple same-class instances. | No instance-selection rule is justified by this run. |
| Temporal context needed | Swimming, cooking, riding and holding cannot be established from generic person/object presence in one frame. | Keep action/temporal queries off a presence-only detector route in any future design. |

The failure inventory contains only three small-object positives and two class-matched held-out negatives. These counts support rejection of this configuration, not a broad estimate of detector quality.
