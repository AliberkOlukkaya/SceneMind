# Natural Video Benchmark V2 failure analysis

The held-out results isolate two different behaviors. Raw visual retrieval finds at least one relevant interval by K=5 for every non-speech positive, but it accepts every negative because nearest-neighbor search has no natural rejection. The calibration-only cutoff reduces visual negative FAR@5 from 100% to 10%, while false abstention rises to 52.4% and R@5 falls from 85.7% to 38.1%. The threshold therefore separates the two calibration videos but does not transfer cleanly across held-out domains.

Speech BM25 has the best rejection behavior: held-out negative FAR@5 is 0% and positive false abstention is 0%. Its positive R@5 is 80%. Several reported speech failures are partial-interval misses rather than complete misses because one query can label multiple discussion intervals.

## Representative reviewed cases

| Case | Observation | Diagnosis |
| --- | --- | --- |
| `street-object-bicycle`, visual | Raw rank 1 is a relevant 25.0 s frame at score 0.2389; the 0.2581 cutoff removes it. | Correct candidate, domain-shifted score; rejection calibration failure. |
| `throw-action-ball`, visual | Relevant 5.01 s and 10.01 s frames rank first and second at 0.2530 and 0.2491, just below the cutoff. | The current frame encoder can retrieve this simple action cue; the cutoff creates the failure. This case does not justify an action model yet. |
| `talk-object-speaker`, visual | The speaker is visible throughout, but the best score is 0.2184 and every result is rejected. | Presentation footage has a lower score distribution than calibration footage. |
| `talk-negative-gradient`, visual | The absent spoken concept returns the presentation at score 0.2701, above the cutoff. | CLIP associates the broad presentation context with the text and cannot verify semantic absence. The hybrid path inherits the same visual false accept. |
| `talk-speech-changing-link`, speech | Relevant transcript segments at 90.85 s and 105.85 s rank first and second; no retrieved segment covers the second labeled discussion at 197–205 s. | Good lexical retrieval for one occurrence, incomplete recall across repeated intervals. |
| `talk-speech-airport`, speech | Results at 191.8 s and 181.98 s cover the second interval; the first result begins at 56.95 s, 0.05 s before the frozen 57.0 s boundary. | One miss is boundary-sensitive. Future benchmark expansion should define and freeze a timestamp tolerance before inference. V2 remains unchanged. |

At calibrated K=5, the report records 23 query-path failures: 14 complete false abstentions, seven partial interval misses, and two negative false accepts. Ten false abstentions involve ordinary visual objects or scenes, one compositional query, one temporal query, and two visual evaluations of speech queries. The raw and accepted top-three evidence for every failure is stored in `reports/natural-v2.json`.

## Cheap calibration experiment

A calibration-only top-1 versus top-2 score-margin rule was tested offline. The cutoff was frozen one floating-point step above the largest calibration-negative margin, 0.02586697. On held-out visual rows it produced 0% FAR but also 100% positive false abstention and 0% R@1/3/5. This rejects the hypothesis that a simple within-query margin normalizes the observed domain shift. It was not added to production.

The main constraint is open-set semantic verification and transferable confidence, followed by multimodal ranking of repeated speech intervals. The evidence does not support Action Recognition, OCR, RAG, fine-tuning, or a larger Whisper model as the next implementation.
