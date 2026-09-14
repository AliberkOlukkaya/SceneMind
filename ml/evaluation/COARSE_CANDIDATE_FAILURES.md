# Coarse candidate diversity failures

The selected multi-scale policy removes temporal duplicates but still loses two held-out queries whose correct region is present in the raw top 20. One further speech-label region is absent from that pool. This separates pool coverage from final selection failure.

## Measured buckets

- `talk-speech-languages`: correct region absent from the two-second raw top 20.
- `talk-speech-changing-link`, `talk-speech-bug`: correct region exists in the raw top 20 but final diversity selection drops it in favor of visually plausible frames.
- `talk-speech-kansas-city`, `talk-speech-bug`, `talk-speech-airport`: five-second top-five successes displaced in the raw two-second top five by neighboring frames; all recover within the two-second top 20.
- `dirt-bicycle-pass`: a human-verified visible bicycle frame exists in the two-second top 20, but neither raw nor diversified top five retains it. The CLIP score is too weak relative to distractors.
- No selected held-out compositional or action/temporal failure remains. No OCR-dependent failure was identified. Three remaining misses are speech queries that the transcript path should handle.

MMR recovers held-out reviewed small-object R@5 to the 50% baseline, but cannot improve it. Simple scene grouping reduces held-out R@5 to 57.14%. The selected change-aware multi-scale method reaches 80.95%, closer to the baseline than the other diversity variants, yet its high-change additions still require scanning the dense two-second stream and do not solve score separation.

The raw two-second top-50 pool reaches 97.62% benchmark correct-region recall and 100% visible small-object coverage. The next failure is choosing five candidates from that pool without category regressions. Another detector, denser global index or scene model is not justified by this result.
