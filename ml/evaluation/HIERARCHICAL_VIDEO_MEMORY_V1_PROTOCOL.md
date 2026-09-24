# Hierarchical Video Memory V1 protocol

This frozen experiment asks whether persistent discourse-level navigation can improve grounded spoken-video Q&A, especially on long videos. It does not change production. `SCENEMIND_QA_ENABLED=false`; Find Moments, CLIP, visual FAISS, RRF60, five-second sampling, Whisper, Speech Search, URL ingestion and production BM25 remain unchanged.

## Data and freeze

Development contains three new licensed English videos and 45 questions (33 answerable, 12 hard negatives), including a 40-minute source. Three source-disjoint validation videos supply another frozen 45 questions with the same balance; one is 40 minutes and another is 22 minutes. No source occurs in any earlier Q&A set. Every supported question records answer facts, minimum sufficient timestamp intervals, category and the semantic sections intersecting those intervals.

Only development selected among three deterministic section configurations. The selected method starts from non-overlapping 30-second/600-character L1 transcript units. A new section begins after at least 35 seconds when an adjacent pause is at least 3 seconds or adjacent pinned MiniLM cosine is below 0.40, with a hard 180-second cap. A short tail merges only when the cap remains satisfied. Section retrieval is fixed at three.

L3 memory is local and extractive: up to two L1 units nearest the section centroid, capped at 420 characters, plus source-frequency topic terms. Every summary unit, topic, section, L1 unit and Whisper segment retains provenance. Summaries only navigate. The generator receives only original timestamped L1 transcript text.

Within the selected sections, original L1 units receive MiniLM cosine plus a small positive-BM25 rank bonus. The five best units, capped at 3,600 characters, become the chronological evidence package. The comparison arms are unchanged BM25 Top-5 and the frozen previous flat semantic candidate. All arms use the same existing grounded generator, prompt, resolver and abstention contract.

The manifest and configuration are frozen at [`hierarchical_video_memory_v1_manifest.json`](hierarchical_video_memory_v1_manifest.json). Validation runs once. Previous Q&A failures may be replayed only after the decision is recorded.

## Metrics and gates

Stage A reports relevant-section Recall@1/@3, wrong-section rate, section counts and durations, including the >=40-minute source. Stage B reports Evidence Recall@1/3/5, minimum-sufficient recall, package completeness and fully complete packages for all three arms. Stage C reports answer correctness, Core User Success, grounding, citation precision/recall, unsupported claims, correct abstention, false answers and false abstention.

Promotion requires the exact gates in the mission: evidence R@5 >=95%; minimum-sufficient and fully complete >=90%; correctness and Core User Success >=90%; grounding and citation precision >=95%; citation recall >=90%; unsupported claims <=5%; correct abstention >=95%; false answers <=5%; false abstention <=10%; and every video's Core User Success >=70%. It must also materially improve BM25 completeness and correctness, user success, or false abstention without a safety regression. A result below 70% Core User Success on any video is catastrophic and blocks promotion.

Memory persistence is checksum-bound to the source transcript, section/summary configuration, pinned embedding revision and schema. Local extractive preprocessing has zero API tokens and cost. Query generation token use and cost are measured separately.
