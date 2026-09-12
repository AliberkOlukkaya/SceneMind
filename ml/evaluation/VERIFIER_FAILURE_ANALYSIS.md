# Image-text verifier failure analysis

BLIP improves some individual negative decisions but does not separate positives and negatives reliably enough for transferable abstention. The calibration score distributions overlap: forcing calibration FAR below 10% abstains on 44% of calibration positives, then 52.4% of held-out positives.

## Representative cases

| Required case | Evidence | Interpretation |
| --- | --- | --- |
| CLIP correct, verifier rejects | `street-object-bus`: CLIP's relevant top candidate scores 0.268; BLIP's best pair probability is 0.130, below the 0.434 threshold. | Correct object evidence receives a low verifier score. This is consistent with domain shift or model calibration, but the example alone cannot distinguish them. |
| Answer in candidates, verifier ranks incorrectly | `throw-compositional-holding`: CLIP ranks the labeled 5.007 s frame first. BLIP ranks the out-of-interval 0.007 s frame first at 0.936 and gives the labeled frame 0.189. | The verifier is highly confident but does not respect the frozen relationship interval. The boundary may also expose annotation sensitivity. |
| Plausible negative correctly rejected | `talk-negative-gradient`: current CLIP calibration accepts the presentation context at 0.270. BLIP's maximum is 0.268, below its threshold. | Joint scoring can reject a plausible but absent presentation topic in this case. |
| Absent query incorrectly accepted | `talk-negative-bicycle`: BLIP assigns 0.519 to a presentation frame even though no bicycle is present. | An image-text matching head still produces contextual false positives; it is not an absence oracle. |
| Compositional result succeeds | `street-compositional-cyclist-bus`: the relevant 15 s frame stays rank one with probability 0.591 and passes the threshold. | BLIP can preserve a visible relation when both subjects are sufficiently salient. |
| Compositional result fails | `throw-compositional-holding`: the correct frame remains in the candidate set but is demoted and filtered. | Relation wording does not reliably improve ordering across domains. |

Other complete false abstentions include the bicycle and intersection in street footage, the yellow ball and park in throwing footage, and the projection screen and presentation room in the talk. Several objects are small, scenes differ from COCO-like captions, or the query describes a whole scene rather than a salient foreground pair. These are correlations with the observed cases, not proven causes.

The action/temporal examples remain inconclusive. BLIP scores independent frames and adds no motion representation. Its rejection of moving traffic and pedestrian crossing therefore does not justify a temporal model; the current candidate generator already contains relevant frames.

The experiment also confirms a resource mismatch. Batched verification avoids five separate model calls, yet the median remains 3.35 seconds and peak working set is about 1.49 GB. Dynamic quantization might reduce resource use, but it cannot repair the observed positive/negative overlap without new evidence. It was not pursued after the quality gate failed.

## Engineering decision

Keep production CLIP, current opt-in calibration, BM25, and hybrid fusion unchanged. Do not test the larger BridgeTower checkpoint in this milestone: BLIP already exceeds the latency ceiling by more than 13×, and BridgeTower's larger checkpoint does not offer evidence that it will solve calibration overlap.

The next work should improve the evaluation before another model download: add more small-object and relation-focused held-out sources, define timestamp tolerance before annotation, and evaluate lightweight non-generative alternatives that expose a true pair score under the CPU budget. A smaller distilled ITM model or an ONNX/quantized verifier is worth considering only when its published artifact, license, and CPU path are clear. Do not fine-tune until a substantially larger labeled development set exists.
