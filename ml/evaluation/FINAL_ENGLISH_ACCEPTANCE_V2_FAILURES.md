# Final English Acceptance V2 failures

The frozen Smart Search run has one PARTIAL and three FAIL outcomes among 26 positives. Only PASS-level results count in retrieval metrics. Diagnostic explicit-mode results were collected after judgment and do not alter the primary outcome.

| Query | Primary outcome | Primary evidence | Diagnostic evidence | Dominant cause |
| --- | --- | --- | --- | --- |
| S03 — why Internet data is split into packets | PARTIAL | Hybrid rank 5 opens at 243.78 s, where the speaker concludes that data is split into 1,500-character blocks. The reason was explained earlier at 179.78–216.78 s. | Speech rank 1 is the same late conclusion; rank 5 reaches packet terminology at 216.78 s. | Hybrid ranking plus passage-boundary precision. The click is related but incomplete. |
| S08 — why a lower-level address is needed before an IP address | FAIL | No Hybrid Top-5 result reaches 1177–1218 s; closest is 1078.78 s and lacks the explanation. | Speech rank 1 at 1210.78 s states that the lower Ethernet level is required; rank 5 at 1211.78 s completes it. | Hybrid fusion/ranking. Speech retrieval contains direct evidence. |
| S09 — how pinging 9.9.9.9 diagnoses DNS | FAIL | Hybrid ranks five earlier DNS-definition moments at 1278.16–1345.16 s and misses 1443–1510 s. | Speech rank 1 at 1489.16 s says to ping 9.9.9.9; ranks 3–4 cover the always-up address and DNS-failure conclusion. | Hybrid fusion/ranking. Lexically related visual/speech context crowds out the direct passage. |
| V04 — slide with three home-network boxes | FAIL | Hybrid Top-5 shows individual switches, routers, or unrelated diagrams; nearest timestamp is 126.78 s outside the interval. | Visual Top-5 also returns switch/router frames at 425–1100 s and never reaches 271–302 s. | CLIP visual retrieval. The target is visibly present in production-sampled frames, so this is not temporal sampling. |

S07 and V06 are low-ranked but PASS-level: S07 rank 5 opens immediately before the exact Wireshark selection advice, and V06 rank 4 visibly contains the named people, arrows, and teacher. They reduce Top-1/Top-3 but count at Top-5. S06 starts a continuous 56-second MAC-address passage and was judged useful at rank 1; its diagnostic Speech result at 576.78 s confirms the later uniqueness clause but does not replace the primary result.

No failure is attributed to ASR. The reviewed production transcript contains the relevant wording for all three weak Speech cases, and explicit Speech retrieves direct evidence for S08 and S09. No failure is attributed to five-second sampling: the V04 target slide is visible in sampled JPEGs, and every requested visual state was independently verified before search. OCR, a new model, routing changes, and acceptance-video tuning are outside this evaluation.

The four negatives are not retrieval successes. They are acceptable only because the UI presents candidates conservatively and the returned content is distinguishable from the absent request. SceneMind still has no reliable global no-match detector.

The result supports one bounded next investigation: use new source-disjoint development material to inspect Hybrid candidate fusion and protect strong Speech evidence from visually generic displacement. The acceptance video and manifest remain held out and must never become development data.
