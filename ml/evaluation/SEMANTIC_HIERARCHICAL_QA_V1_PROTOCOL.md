# Semantic Hierarchical Q&A V1 protocol

## Question

Can local semantic transcript retrieval plus hierarchical evidence selection improve answer completeness and false abstention without weakening the frozen grounded-answer safety contract?

## Isolation and data

Development uses 45 questions over three newly acquired licensed English sources: HTML basics, amyloidosis, and reuse of freely licensed media. Validation uses 45 questions over three different sources: RAIDZ expansion, 1Lib1Ref, and a 44-minute Wiki Education Foundation conference talk. Each split has 33 supported answerable and 12 hard-negative questions. Sources are disjoint between splits and from all earlier Q&A milestones. Questions, key facts, minimum sufficient evidence, intervals, media hashes, configuration, and sources were frozen at SHA-256 `e99fecc0c0c941afe830eb1d8ff8072212b18226096ead0bcd0b54343d068ba1` before validation.

Previous Q&A sets were not used for development. The six Final Ask Video Core Acceptance failures were replayed only after the decision froze.

## Frozen candidate

The encoder is `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0, 384 dimensions. It runs locally on CPU with mean pooling and L2 normalization. Fine units greedily group adjacent Whisper segments without overlap, capped at 30 seconds or 600 characters. Context units are non-overlapping groups of three fine units.

FAISS inner-product indexes retrieve ten fine and five context candidates. A deterministic rank-only selector combines fine semantic rank, context membership, and a half-weight BM25 fine rank. It deduplicates, caps the package at five fine units and 3,600 characters, then restores chronological order. Only fine units may be cited.

The existing `gpt-5.4-mini-2026-03-17` prompt, strict schema, claim evidence contract, citation resolver, generation settings, and abstention behavior remain unchanged. BM25 Top-5 is the paired baseline. Validation runs once; no validation result may change the frozen candidate.

## Gates

The fixed gates are the requested 95% Evidence Recall@5, 90% minimum-sufficient and fully-complete evidence, 90% correctness and user success, 95% grounding and citation precision, 90% citation recall, at most 5% unsupported claims and false answers, at least 95% correct abstention, at most 10% false abstention, no catastrophic per-video failure, and meaningful paired improvement without safety regression.
