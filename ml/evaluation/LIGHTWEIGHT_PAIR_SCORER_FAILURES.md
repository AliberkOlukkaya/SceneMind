# Lightweight pair scorer failure analysis

UForm solves the resource problem and repeats the calibration problem. Its calibration cutoff separates all ten held-out negatives, but the positive score distribution shifts downward enough to reject ten of 21 positives. The logistic boundary shifts further and rejects fifteen. Neither score is a probability of semantic presence.

## Representative failures

| Failure bucket | CLIP | BLIP ITM | UForm ONNX | Logistic scorer | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Small object not accepted | Raw candidates find bicycle, yellow ball and man-holding-ball at R@5=100%; scalar threshold accepts none | Earlier calibrated verifier also rejects bicycle/ball cases | All three score below 0.3302; small-object R@5 becomes 0% | All three fall below 0.6778; R@5 becomes 0% | Absolute similarity remains driven by broad scene content; calibration does not prove a small object is present. |
| Correct candidate reranked badly | `throw-compositional-holding` starts with the labeled 5.007 s frame | BLIP demotes it behind 0.007 s with high confidence | UForm keeps candidates but does not overcome its low query maximum of 0.2938 | Learned rank features move correct answers within the five and lower held-out R@1 | Candidate-level score transformations can harm ordering even when Stage 1 already succeeded. |
| Semantic scene similarity | Raw CLIP accepts every negative | BLIP rejects `talk-negative-gradient` but accepts `talk-negative-bicycle` | Threshold rejects all ten held-out negatives | Threshold rejects all ten | Negative rejection is possible, but only at a high positive cost. Presentation, street and park context overlap with many absent requests. |
| Positive domain shift | Scalar calibration has 52.4% PFA | BLIP repeats 52.4% PFA | PFA is 47.6%; five throwing-video visual queries and all five speech-worded visual queries are rejected | PFA is 71.4%, including most street, throwing and conference positives | Calibration video identity and score scale remain stronger signals than semantic presence. |
| Relationship mismatch | Raw relationship R@5 is 100% | BLIP preserves some relations and badly demotes `man holding a yellow ball` | Calibrated relationship R@5 is 62.5%, PFA 25% | R@5 is 50%, PFA 50% | UForm is a dual encoder and has no image-conditioned token interaction; shared training layers do not provide per-pair cross-attention at inference. |
| Temporal information required | Static CLIP candidates happen to cover all three held-out action intervals | BLIP adds no motion and rejected some actions | UForm keeps all three action cases after threshold | Logistic rejects all three | UForm's success on this tiny slice does not establish motion understanding; all systems still score individual frames. |
| Text-image incompatibility | Speech-worded queries are poor visual prompts | BLIP rejected some speech-topic queries | UForm rejects all five held-out speech-worded positives after threshold | Logistic accepts two of five | BM25 remains the correct evidence path for spoken topics. Visual failure here is expected and must not trigger larger visual models. |

## Exact UForm abstentions

The ten held-out positive query maxima below the 0.3302116394 threshold are: bicycle (0.3097); man in black shirt (0.2790); yellow ball (0.3013); park path (0.3049); man holding yellow ball (0.2938); and five conference speech-topic queries ranging from 0.2875 to 0.3249. There are no held-out false accepts. This improves the scalar and BLIP calibrated FAR from 10% to 0%, but fails PFA by a wide margin.

UForm changes the first result for 19 of 31 held-out queries. It preserves R@1 overall but lowers R@3 and MRR slightly. Examples include moving the bicycle from 25 s to the labeled 15 s, but also moving the projection-screen query from a labeled 190 s frame to 25 s. These mixed changes support the aggregate conclusion: it supplies another similarity geometry rather than reliable pair verification.

The logistic scorer overfits source-dependent retrieval statistics despite only eight parameters and L2 regularization. Calibration reranking MRR@5 rises from 0.852 to 0.924, while held-out MRR@5 drops from 0.805 to 0.790. Its calibrated PFA worsens from 48.6% on calibration to 71.4% held out. Small sample size, correlated five-candidate rows and only seven source groups make that risk material.

## What the experiment rules out

More CPU optimization is not the answer: UForm already passes latency and memory. A different cutoff cannot satisfy both observed FAR and PFA without violating the frozen calibration protocol. K=3 loses too much candidate recall. Further dual-encoder similarity models are unlikely to add the missing explicit object/relation evidence, and larger SigLIP/Jina variants would spend more memory on the same score family.

The next bounded experiment is a small-object/object-detector branch for explicit object queries. It should verify named object presence on CLIP's candidates and retain CLIP fallback. This recommendation follows the 0% calibrated small-object R@5 shared by scalar CLIP, UForm and the learned scorer, against 100% raw candidate R@5.
