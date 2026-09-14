# Query routing failure analysis

Learned AUTO makes one held-out routing mistake: the negative query “Where does the speaker demonstrate cooking?” is labelled Hybrid and sent to Visual. This is a Hybrid-over-specialization error. Both Visual and Hybrid return only irrelevant evidence, so the mistake does not alter positive Recall/MRR. It still shows that a 31-query held-out set with one Hybrid target cannot establish broad Hybrid classification quality.

The heuristic obtains the same end-to-end R@5 despite only 70.97% routing accuracy. It sends eight Visual targets to Hybrid. Those videos either lack a transcript or retain strong visual candidates through RRF/fallback, masking routing errors. Search quality is therefore primary, while the confusion matrix remains necessary for diagnosing avoidable compute and unavailable-transcript behavior.

There are no held-out speech-to-visual errors, visual-to-speech errors, positive transcript-unavailable failures, weak selected-path BM25 failures, weak selected-path CLIP failures, or selected Hybrid fusion failures. Eight queries are lexically ambiguous to the heuristic. The benchmark still contains known temporal and OCR limits from earlier experiments; this routing milestone does not add those capabilities.

The learned probabilities are route scores, not correctness confidence and not no-match confidence. Calibration selected a zero Hybrid-fallback threshold. Global rejection remains disabled because the previous no-match experiment falsely rejected 90.48% of valid queries.

The next test must use Aliberk's uncommitted 30–60 minute videos, freeze queries and intervals before retrieval, include English/Turkish wording where personally relevant, and record transcript/index availability. A populated copy of `personal_video_acceptance_template.json` must stay private and ignored.
