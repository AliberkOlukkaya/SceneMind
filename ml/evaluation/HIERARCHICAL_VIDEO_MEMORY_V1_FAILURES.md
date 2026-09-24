# Hierarchical Video Memory V1 failures

## Outcome

The candidate failed nine promotion checks: Evidence Recall@5, minimum-sufficient recall, fully complete evidence, Answer Correctness, Core User Success, Citation Recall, false abstention, per-video catastrophic protection, and material BM25 improvement. Grounding, citation precision, unsupported-claim safety, hard-negative abstention and false-answer safety passed.

## Failure taxonomy

The 14 hierarchical answer failures comprise six false abstentions, three generation/completeness errors, two section misses, two local-evidence misses and one wrong-section answer. The dominant product effect is excessive abstention and incomplete local evidence, not hallucination.

The 22-minute GM source exposed the worst interaction. Two questions missed the relevant section, one selected a different but topically plausible section, and several questions reached the correct region but either omitted required facts or abstained. Core User Success was 3/11. The 40-minute ACIP source had perfect Section R@3 yet still failed three answers, demonstrating that successful navigation alone does not guarantee complete evidence or generation.

The hierarchy's final evidence completeness was 12.12 percentage points below flat semantic and 12.12 points below the 90% gate. Compared with BM25 it lost 12.12 points of evidence completeness and 21.21 points of Core User Success. False abstention rose from BM25's 18.18% to 30.30%.

## Summary and safety findings

Extractive memory did not introduce summary hallucinations. All 46 validation summaries map exactly to original transcript units, and every topic label is source-attested. Memory text never entered the evidence package and no section ID could resolve as a citation. All answered claims were grounded, citation precision was 100%, all 12 hard negatives abstained and there were no false answers or unsupported claims.

## Stop rule

This frozen set is now observed and cannot be used for tuning. Do not change K, section thresholds, topic weights, evidence budgets, prompt text, or abstention rules against these failures. Do not automatically create a V2. A future architecture decision must choose between a genuinely new local evidence/completeness design on new sources and postponing Ask Video.
