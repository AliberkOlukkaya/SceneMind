# Decisions

## 041 - Keep Ask Video disabled after structured-question validation failure

Add a lightweight answerability contract without changing evidence retrieval: deterministic question constraints for count, temporal and selected relation forms; strict claim-level citations; explicit missing requirements; backend-derived citations; and safe abstention for malformed, uncited, incomplete-count or provider-declared insufficient responses. Pin `gpt-5.4-mini-2026-03-17`. Use 30 new development questions over two licensed sources to select one of three bounded prompt/contract designs. Freeze two further source-disjoint videos and 30 questions at SHA-256 `26646d2bf977981950e68888898522e348e6019cc39acac417a8a28acca34255` before one validation run. Do not use the observed Grounded Video Q&A V1 validation for development.

Choose decision D. Validation reaches 94.44% Evidence Recall@5, 77.78% Answer Correctness, 100% Grounded Answer, 100% Citation Precision/Recall, 0% Unsupported Claims, 100% Correct Abstention, 0% False Answers and 11.11% False Abstention. Frozen `LIST_COUNT` correctness is 0/2 and `TEMPORAL` correctness 1/2, so answer correctness and no-catastrophic-category gates fail. Historical replay of the old temporal and mobile-data patterns safely abstains but is excluded from metrics. Keep `SCENEMIND_QA_ENABLED=false`; do not rerun or tune either Q&A validation. The next eligible experiment is Q&A Structured-Question Evidence V1 on wholly new development sources, with BM25 Top-5 and Find Moments frozen.

## 040 - Keep transcript-grounded Ask Video evaluation-gated after abstention failure

Add a Q&A-specific evidence layer without changing production search: deterministic 45-second/900-character transcript chunks with one-segment overlap, existing BM25 Top-5 retrieval, an extensible `AnswerGenerator`, strict OpenAI Responses JSON schema, pre-provider abstention on no lexical evidence, and server-side evidence-ID-to-timestamp resolution. Reject unknown evidence IDs and answerable responses without citations. Send only the question and selected transcript text with `store: false`; never send video bytes, full transcripts, source metadata, paths, or secrets.

Choose decision D. The one checksum-frozen real-provider run reaches 100% Evidence Recall@5, but one of three unanswerable questions receives a substantive response. Answer Correctness is 88.89%, Grounded Answer 80.00%, Citation Precision 94.44%, Unsupported Claim 8.70%, Correct Abstention 66.67% and false-answer 33.33%; all miss their gates. Keep `SCENEMIND_QA_ENABLED=false`, do not promote Ask Video, preserve Find Moments unchanged, and develop abstention safety only on new development evidence. Never tune or rerun against this frozen validation set.

## 039 - Promote bounded URL ingestion through the existing local pipeline

Add `URLIngestProvider` with Direct Media and YouTube implementations. Accept only HTTP(S), reject embedded credentials and every non-global resolved IPv4/IPv6 address, revalidate redirects and DNS address sets, stream direct media with authoritative byte counting, and wrap pinned yt-dlp in a monitored subprocess with a 720p ceiling. Queue acquisition as a bounded durable job, retain provenance in the existing atomic video manifest, classify retryability in migration 003, and hand validated local media to the unchanged ingest, Whisper, CLIP/FAISS, BM25 and RRF60 paths. Exact provider plus external-ID duplicates return the existing video.

Choose decision A. A real 35,611,959-byte DirectMedia import and local upload produced identical 219.443-second media, 44 frame timestamps, 60 Whisper segments, 44×512 indexes and all nine frozen Visual/Speech/Smart Search Top-5 timestamp lists. A real public YouTube run acquired the official CC BY 3.0 Big Buck Bunny source at a bounded format, produced 127 frames, completed speech and visual stages, returned visual search results and left no partial files. Keep retrieval unchanged. The next milestone is Grounded Video Q&A V1, designed separately over local timestamped evidence.

## 038 - Reject evidence-preserving quota fusion on cross-category validation

Compare eight deterministic rank-only configurations on the two-source, 32-query Hybrid Fusion development set. Select and checksum-freeze `quota_1`, which preserves the first unique bucket from each modality and fills the remaining Top-5 positions in production RRF order. Validate it once on 48 queries from three source-disjoint sources after reproducing all historical RRF60 Top-5 outputs.

On 24 positives, the candidate raises useful Top-1/3/5 from 19/21/21 to 20/22/22, raises MRR@5 from 0.8333 to 0.8681, retains 21/21 baseline Top-5 successes, eliminates the one measured explicit-modality displacement, and recovers one Speech query. Reject promotion because every aggregate gain comes from Speech, Visual is unchanged, and Hybrid MRR falls from 0.7500 to 0.7222. Choose decision B. Do not open the protected 34-query Hybrid Holdout V1, run another fusion variant, or change production. Keep Smart Search on uncapped RRF60 and choose any later product milestone independently.

## 037 - Freeze the search core and prepare v1.0.0-rc1

End search-core ML research for v1.0 after the calibrated fusion generalization failure. Keep Smart Search mapped directly to uncapped RRF60 Hybrid, retain explicit Speech and Visual modes, and leave AUTO outside the normal interface. Present the system as a local single-operator portfolio release candidate with documented held-out failures and limitations. Add simple version metadata, release-focused documentation and clearer real pipeline state labels; do not change models, sampling, retrieval, ranking or API search contracts. Move OCR, action understanding, RAG, multilingual work, stronger deployment and ranking research to Future Work that requires new evidence.

## 036 - Reject raw-score fusion before candidate construction

Use 72 legitimate calibration queries from three sources and 32 queries from two source-disjoint development-validation sources to test per-retriever confidence calibration. Calibration-only selection chooses Visual raw-score logistic and Speech query-relative logistic mappings, but both lose to rank-only on validation AUC and Brier. Choose outcome C. Stop before building a fusion formula or opening the 34-query protected holdout, mark the artifact non-production, and keep uncapped RRF60. A second-stage reranker is not justified. Collect at least three new source-disjoint calibration sources with full Top-50 Visual/Speech traces and freeze a new validation split before reconsidering confidence-aware fusion.

## 035 - End diagnosis with confidence-aware ranking as a design class

Reconstruct every available Hybrid ranking failure from development, frozen Holdout V1, and safely reproducible historical Acceptance V2 evidence. Keep `app.hybrid.fuse` authoritative and verify all 30 reconstructed V2 outputs against the historical artifact. Compare 19 failures with 12 balanced controls. Fourteen failures are primarily irrelevant consensus; 16 contain explicit required-modality Top-5 evidence displaced outside Hybrid Top-5, and 11 reverse a stronger same-retriever raw-score advantage. Two-contribution winners occur in 18/19 failures and 12/12 controls, while same-thumbnail Speech duplicates are discarded, so overlap or accumulation alone is not causal. Temporal attachment/context explains two cases and weak required-modality rank explains two.

Conclude that fixed RRF60 is insufficient for the observed tail, while remaining the safest production baseline because Cap 1.50× failed frozen holdout. Rank calibrated confidence plus explicit agreement features first by evidence coverage, followed by second-stage reranking and temporal representation. Implement none of them here. Do not search another alpha, tune on Holdout V1, optimize on Acceptance V2, or start Acceptance V3. Any next ranking experiment requires new development evidence and a future untouched holdout. Production remains uncapped RRF60.

## 034 - Reject the 1.50× overlap cap on frozen holdout evidence

Freeze 34 queries over two new source-disjoint Wikimedia sources before retrieval: 11 Speech, eight Visual, nine Multimodal, and six negative. Process both through normal streamed upload, durable ingestion, Whisper, CLIP, and persistence. Compare only production-equivalent RRF60 with the fixed 1.50× shared-contribution cap; do not use Final English Acceptance V2 or search another parameter.

Baseline positive Top-1/3/5 is 12/19/21 with MRR@5 0.5560. Cap reaches 15/19/20 and 0.6083. Speech Top-5 falls from 8/11 to 7/11; Jimmy Wales falls from 6/12 to 5/12 while RUN stays 15/16. There are zero rescues and one newly broken query. Strong-candidate retention is unchanged at 18/32, all 28 positives have required Top-50 input evidence, and one negative becomes more plausibly misleading. Choose **REJECTED** because the predeclared Top-5, Speech, rescue/break, source-robustness, and negative-ordering criteria fail. Do not run a second frozen evaluation, tune the holdout, or promote the cap. Keep production `app.hybrid.fuse` at uncapped RRF60.

The approved Jimmy media also exposed a final-segment boundary defect: Whisper audio can extend slightly beyond OpenCV's playable video duration. Validate starts, clip stored ends to playback duration, and discard fully out-of-range tails. Preserve model, sampler, search contracts, and ranking.

## 033 - Carry the 1.50× overlap cap only to holdout validation

Freeze the existing two-video, 32-query diagnostic development evidence and establish exact offline parity with production `app.hybrid.fuse` before evaluating alternatives. Compare three predeclared rank-only caps and three independent list-normalized rank formulas. Do not use raw CLIP/BM25 scores, learned weights, modality quotas, Final English Acceptance V2, or production code.

The 1.50× cap raises Top-1/3/5 from 10/16/20 to 11/16/21, MRR@5 from 0.4827 to 0.5155, Speech Top-5 from 6/11 to 7/11, and strong retention from 13/28 to 14/28. Visual remains 7/8 and Multimodal 7/9. It rescues one query and breaks none; neither source loses Top-K, although one supplies the Top-5 gain and Multimodal MRR declines slightly. Stronger caps or normalized ranks introduce Visual failures. Skip the optional modality floor. Designate 1.50× a provisional development candidate, keep production uncapped, and require one new source-disjoint frozen holdout before any promotion.

## 032 - Diagnose exact-thumbnail overlap before changing Hybrid

Use two existing source-disjoint CC BY 4.0 English development videos and freeze 32 queries before retrieval. Keep Final English Acceptance V2 fully isolated. Add evaluation-only tracing that calls production `app.hybrid.fuse` as the authoritative ranker and proves reconstructed IDs, timestamps, scores, and ordering are identical. The normal API and frontend remain unchanged.

On 28 positives, unchanged Hybrid reaches interval-evidence Top-5 on 20; seven failures retain relevant evidence in an input list and one is candidate recall. Thirteen relevant Speech Top-5 and eight relevant Visual Top-5 candidates are displaced. Exact-thumbnail dual contributions occupy 135/160 final slots. Because `2/(60+50)` exceeds `1/(60+1)`, every shared top-50 bucket outranks every single bucket under the current limits. Choose diagnosis-only outcome C: production remains unchanged. Next pre-register a minimal per-modality preservation or capped-overlap experiment on development sources and require a new frozen source-disjoint evaluation before promotion.

## 031 - Fail Final English Acceptance V2 on Hybrid ranking

Freeze 30 natural English queries only after full independent review of a new source-disjoint 30:29 CC BY-SA presentation, then run every primary request through unchanged Smart Search / Hybrid. Normal durable ingestion succeeds in 244.393 seconds with 366 frames, 576 Whisper segments, zero retry/failure/residue, and 1.94 GiB peak worker-tree RSS. PASS-level useful Top-1/3/5 is 76.92%/76.92%/84.62%, so Top-3 and Top-5 miss their frozen 85%/90% gates. Multimodal is 100% at rank 1, Visual Top-5 is 87.50%, and Speech Top-5 is 70.00%. All four negatives remain understandable under conservative wording.

Choose C. Explicit post-judgment Speech diagnostics recover both primary Speech FAILs at rank 1; the sole PARTIAL also moves to rank 1 but remains incomplete. Explicit Visual still misses the remaining failure. Hybrid fusion/ranking of strong Speech evidence is the dominant subsystem, with one secondary CLIP visual-retrieval miss. Do not tune, relabel, or rerun this held-out source and do not change production in this milestone. The English-first v1.0 search core is not frozen. Any next work must use new source-disjoint development evidence for a bounded product-level Hybrid fusion investigation; no new model milestone starts automatically.

## 030 - Make Hybrid-backed Smart Search the v1.0 default

Remove automatic route classification from the normal SceneMind v1.0 path after real-video evidence measured 48.33% accuracy for the production router and 60.00% for the best lightweight candidate. Present three product modes: Smart Search maps directly to existing Hybrid retrieval and is the default; Spoken Content maps to Speech; Visual Content maps to Visual. Preserve the `auto` API value, router implementation, artifact, feature flag, tests, and historical reports for compatibility and internal work, but do not expose or recommend AUTO in the standard frontend.

This decision changes only frontend labels, default state, and the request mode. CLIP, Whisper, BM25, RRF, FAISS, five-second sampling, ranking, indexing, workers, response data, conservative candidate wording, and click-to-seek behavior remain unchanged. Final English Acceptance V2 will use one new independent 30-60 minute English video and evaluate Smart Search / Hybrid useful Top-1/3/5, latency, timestamps, pipeline completion, and honest negative-query UX. It will not score AUTO; explicit Speech and Visual modes are diagnostics only.

## 029 - Reject the character router after human-grounded validation

Review actual five-second frames and local Whisper transcripts for ten independent CC BY/CC BY-SA English videos, then freeze 360 balanced route queries by video source. Use four train videos, four validation videos, and two untouched final-test videos. Preserve the protected failed-acceptance source without opening or reusing its evidence. Evaluate production first and test the previously fixed character 3-5 gram TF-IDF plus linear-softmax configuration with raw argmax and no confidence fallback.

The production router reaches 48.33% frozen accuracy and 100.00%/0.00%/45.00% Visual/Speech/Hybrid recall. The character candidate reaches 60.00% and 50.00%/40.00%/90.00% recall at 1.626/1.928 ms median/p95. Twenty-four of 60 candidate routes are wrong; only two errors are on annotations flagged ambiguous before evaluation. Choose outcome C because overall, Visual, and Speech gates fail materially. Do not run a formal second frozen evaluation, package the experimental artifact, promote the router, or schedule Final English Acceptance V2. Keep production AUTO, explicit modes, and retrieval unchanged, and do not automatically start another model experiment.

## 028 — Reject source-card router promotion despite passing numeric gates

Freeze 450 balanced English route queries across 15 independently authored source scenarios, with nine train, three validation and three test source groups. Protect the failed final acceptance set from exact query overlap and never use its media, transcript, labels, failures, results or wording. The unchanged 54-parameter router reaches 50.00% frozen accuracy and 100.00%/13.33%/36.67% Visual/Speech/Hybrid recall. Validation selects a character 3–5 gram TF-IDF linear model with a low-confidence Hybrid fallback; frozen accuracy is 95.56%, recall is 100.00%/90.00%/96.67%, median/p95 is 2.7706/3.3677 ms, and the second run is identical. The fallback itself fails to generalize, lowering raw 97.78% accuracy to 95.56%. Choose outcome E because source-card labels are not grounded in independently reviewed real videos. Do not promote the 731,122-byte artifact, fallback or vocabulary. Preserve production AUTO, explicit modes, dependencies and retrieval. Next collect new real-video annotations with source-disjoint splits; after a later promotion, require unrelated media for Final English Acceptance V2.

## 027 — Fail final acceptance on AUTO routing generalization

Freeze 30 English queries before retrieval over one independently reviewed, CC BY 4.0, real continuous 54:11 technical presentation. Run normal HTTP upload, durable FFmpeg ingest, Whisper, CLIP, persistence, and production AUTO without tuning or changing retrieval. Processing succeeds in 579.258 seconds with no retry or residue. Search does not: routing is 56.67% against the frozen 90% gate, and positive useful Top-1/3/5 is 50.00%/65.38%/65.38% against preferred 70% and required 85%/90%. Negative conservative UX passes 4/4. Explicit diagnostics recover six weak cases through their frozen Speech or Hybrid route. Choose decision C and name one dominant blocker: AUTO routing generalization on natural English interrogative technical queries. Do not tune or relabel this acceptance source, do not change production in this milestone, and do not broaden model research. The next bounded milestone uses source-disjoint routing evidence; any later acceptance uses new media and newly frozen ground truth.

## 026 — Ship conservative candidate wording; block final acceptance on real media

Keep the validated CLIP/Whisper/BM25/RRF/AUTO core unchanged. Present ranked output as “Most relevant moments,” explain once that possible moments may appear without an exact match, hide raw scores, and preserve timestamp, transcript, evidence, and click-to-seek. Do not infer absence from an empty or weak list. Inventory local media before final acceptance: the sole 30–60 minute source is a synthetic repeated stress fixture, while the longest real source is 1,369.633 seconds and silent. Choose decision D and do not fabricate routing or Top-k metrics. Prepare an English-only checksum-bound validator and protocol; resume only when a real continuous English-speaking 1,800–3,600 second video with usage rights and no tuning history is available.

## 025 — Stop no-match model experimentation for English v1.0

Freeze 72 balanced calibration and 48 balanced held-out English queries across six source-disjoint Commons groups. Evaluate separate CLIP-list, BM25/transcript, and RRF/dual-path rules after AUTO routing. AUTO passes at 95.83%, but held-out Visual/Speech/Hybrid FAR is 8.33%/33.33%/50.00% and false abstention is 83.33%/16.67%/16.67%. Overall R@5 falls from 87.50% to 41.67%. Choose outcome E. Do not add a threshold, feature flag, API uncertainty state, or personal-acceptance rerun. For v1.0, retain ranked results and explicit modes and use conservative wording. Further rejection-model work is not justified by current score separation.

## 024 — Support one-hour local video with streamed upload and bounded resources

Raise configurable defaults from 250 MiB/30 minutes to 1 GiB/60 minutes only after preserving direct request streaming, adding early and counted size rejection, reserving 512 MiB free disk plus 25% processing headroom, atomically staging frames and embeddings, and cleaning disposable artifacts at both stage and supervisor boundaries. Use the existing durable queue and expose its real queued/running stage instead of fake percentages. Replace the single 15-minute worker deadline with bounded per-kind defaults: 30 minutes ingest, two hours speech and one hour visual. An unmodified 439,295,727-byte tutorial reaches ready ingest and visual states; a 45-minute audio stress fixture reaches ready Whisper and CLIP states in 245.4 seconds with 3.02 GB peak worker-tree RSS and no remaining temporary files. Search models and five-second sampling stay unchanged. This proves infrastructure support only; 30-60 minute retrieval usefulness still requires personal acceptance evidence.

## 023 — Do not promote the Turkish compatibility candidate

Freeze 72 natural Turkish queries across three calibration and three held-out source groups, with checksum leakage protection against personal acceptance. The production router reaches 53.3% held-out Turkish accuracy. A calibration-selected 14-feature linear router improves this to 76.7%, while cheap path-specific lexical adaptation reaches 80.0% AUTO R@5; both miss the fixed 90% routing and 85% retrieval gates. Direct Turkish CLIP already reaches 93.3% forced-Visual R@5 and lexical adaptation lowers it to 86.7%. A pinned multilingual MiniLM diagnostic raises Speech R@5 to 100% and AUTO R@5 to 86.7%, but routing remains below gate and the component adds 254 MiB RSS, about 458 MiB cache and 9.995/14.378 ms median/p95 query latency. Choose outcome E. Keep production unchanged, skip the second and personal-acceptance reruns, and avoid a Turkish reliability claim for v1.0. Add independent Turkish sources and freeze a new router-validation split before reconsidering a semantic Speech branch.

## 022 — Do not accept v1.0 after the first personal-video run

Freeze 54 queries over the three exact supplied videos before retrieval, then use the production HTTP ingestion and search paths without tuning or model changes. AUTO reaches 77.8% routing accuracy and positive-query useful R@1/3/5 of 52.8%/66.7%/83.3%; all frozen quality gates fail. English Top-5 is 88.9% and Turkish Top-5 is 77.8%, while all 12 route mismatches are Turkish. Fourteen of 18 negatives return plausible but unsupported moments. Search latency passes at 23.25/32.02 ms median/p95, but the 419 MiB tutorial is rejected and requires an external transcode. The supplied lecture/demo durations also do not establish the requested 30–60 minute use. Choose outcome C, keep production unchanged, and make a bounded Turkish compatibility study with the current models the next ML milestone. Treat no-match disclosure and long-video policy as product work; require held-out evidence and a frozen repeat before promotion.

## 021 — Block personal acceptance rather than reuse benchmark media

Inventory the ignored workspace before testing. Only fixtures, public benchmarks and earlier calibration sources are present; there is no Aliberk-selected lecture, demo, ordinary video or populated private manifest. Do not convert benchmark accuracy into a personal-use claim. Complete the checksum-bound manifest validator and human-usefulness aggregation, including Turkish, negative queries and all requested scenario/category slices, while leaving production unchanged. Record the current 1,800-second duration limit as an acceptance constraint: a video over 30 minutes is rejected. Resume evaluation only after real personal media is selected and annotations are frozen.

## 020 — Promote tiny learned AUTO routing; retain explicit modes

Freeze 36 balanced route queries over three new, source-disjoint Commons lecture/tutorial/demo videos with 205 real Whisper-tiny segments. Compare readable rules and a class-balanced 54-parameter softmax model using 18 lexical features. The learned router wins calibration, reaches 96.8% held-out routing accuracy, and matches explicit-route Oracle R@1/3/5 at 85.7%/92.9%/95.2% with MRR 0.9762 and zero category loss. Median/p95 is 0.031/0.042 ms; an identical second frozen evaluation matches. Choose outcome C. Add AUTO behind `SCENEMIND_AUTO_ROUTING_ENABLED`, default the UI to Auto, preserve Visual/Speech/Hybrid overrides, and keep no-match disabled. A private 30–60 minute personal-video acceptance run is the blocker before v1.0 reliability claims.

## 019 — Keep explicit paths; reject cheap list ranking and no-match

Evaluate five-second CLIP top-20/top-50 candidates with score normalization, margins, temporal support, visual change, embedding agreement and calibration-only logistic models. The selected list scorer lowers held-out R@5 from 85.7% to 76.2%, while Oracle@20/50 is 92.9%/100%. A 10%-FAR calibration threshold reaches 0% held-out FAR only by falsely abstaining on 90.5% of positives. Resources pass at 0.44/0.96 ms median/p95 and 232 parameter bytes, but both quality gates fail. Existing explicit visual/speech/hybrid routing with real BM25/RRF evidence reaches 95.2% R@5. Choose outcome E: routing is dominant. Keep raw five-second CLIP and explicit modes, skip the second run, and collect new speech calibration sources before automatic routing. A stronger semantic reranker is justified only after the route is correct.

## 018 — Keep raw five-second retrieval; stop cheap diversity promotion

Measure raw top-20/top-50 redundancy and compare temporal NMS, embedding MMR, combined selection, CLIP-change grouping and multi-scale sampling on the frozen 66/31 split. Calibration selects a top-20 multi-scale pool with 0.20 embedding-change threshold and five-second NMS. Held-out R@5 is 81.0% versus the 85.7% baseline; reviewed small-object R@5 remains 50%, speech falls 20 points, and dense indexing grows 2.40x. The two-second top-50 pool still reaches 97.6% correct-region and 100% reviewed visible-object recall, locating the failure in final score separation. Choose outcome F, reject promotion and skip the second run. Keep production unchanged; next specify a bounded candidate-list ranking/no-match experiment before any Video RAG work.

## 017 — Reject bounded secondary search; repair coarse candidate generation

Evaluate ±2/4/6-second windows, top 5/10/20, six-second temporal spacing, secondary CLIP, Nano 640 and calibration-only fusion over the frozen 66/31 calibration/held-out split. Calibration selects top-5 ±2 seconds without spacing and about nine secondary frames per routed query. On four human-reviewed events, selected windows reach only 50% of held-out visible evidence and secondary CLIP reaches 0% at K=5. Detector weights above zero hurt calibration, while the <=10% FAR threshold causes 100% small-object held-out abstention. Warm median and memory pass, but cold misses produce 2.54-second p95. Choose outcome C and leave production unchanged. Improve coarse candidate generation and bounded temporal coverage before reconsidering object evidence.

## 016 — Use sampling-first, query-gated Nano 640 as the next experiment

Freeze four small-object events from three new Commons sources and review the exact sampled JPEGs before detector inference. Five-second visible-evidence recall is 25%; 2 and 1 seconds both reach 100%, making sampling the dominant measured failure. On the same 18 visible frames, manually box-reviewed YOLOX-Nano recall improves from 66.7% at 416 to 83.3% at both 640 and 768. RT-DETR-R18 reaches 88.9%, but its 362.8 ms per-frame p95 and 552.5 MiB added RSS are disproportionate to the 5.6-point gain over Nano 640. Choose outcome C. Keep production unchanged; next test a query-gated, cached 2-second secondary sample path with Nano 640 over bounded windows and a wider candidate strategy. Both passing Nano sizes produced identical detection rows on second frozen runs.

## 015 — Reject YOLOX-Nano; separate detection resolution from frame sampling

Freeze the exact three-query small-object failure inventory before model selection. Test the official Apache-2.0 YOLOX-Nano `0.1.1rc0` ONNX artifact only on frozen CLIP top-five frames with explicit COCO aliases. Its 90.5 ms warm median, 99.6 ms p95 and 58.0 MiB added peak RSS pass resources, and class-matched negative FAR is 0%, but the calibration-only 0.8569 threshold causes 100% small-object false abstention. One ball-positive candidate does not visibly contain the ball, and the detector's best bicycle confidence is only 0.6085. Skip reranking, relationship rules and the second run; keep production unchanged. Next collect frame-visible source-disjoint small-object evidence and test a higher-resolution detector while evaluating sampling independently.

## 014 — Stop dual-encoder threshold search; test explicit small-object evidence

Freeze two more Commons calibration sources with 20 phone/key/bag/cup and relationship queries, then evaluate pinned `unum-cloud/uform3-image-text-english-small` through ONNX and an eight-parameter logistic scorer over CLIP retrieval statistics. UForm passes local resources at 202.1 ms warm median, 229.1 ms p95 and 163.1 MB isolated peak delta. It preserves candidate R@5 and reaches 0% held-out FAR, but abstains on 47.6% of positives. The logistic scorer costs 0.066 ms and abstains on 71.4%. Both fail the 20% limit, so skip the second run and keep production unchanged. All calibrated methods score 0% R@5 on the three small-object cases while raw CLIP scores 100%. Stop this similarity-threshold family and evaluate one explicit small-object/object-detector branch next.

## 013 — Reject BLIP ITM production promotion

Expand verifier calibration with three source-disjoint Commons videos and 30 frozen relation/context annotations, then test pinned `Salesforce/blip-itm-base-coco` only over CLIP's top five. The calibration-only 0.433838 threshold reaches 10% held-out negative FAR but causes 52.4% positive false abstention and 26.2% R@5. Verifier-only reranking preserves R@5 but reduces R@1 and MRR. Batched CPU latency is 3.352 seconds and peak working set is about 1.49 GB. This fails the accuracy and 250 ms resource gates. Keep the experiment isolated, skip a second held-out run, and leave production unchanged. BridgeTower is larger; SigLIP lacks the joint ITM architecture being tested. Seek a materially smaller non-generative pair scorer only after expanding frozen held-out evidence.

## 012 — High-recall candidates before an open-set verifier

Natural V2 freezes 47 annotations over five independently sourced natural videos and evaluates visual, speech and hybrid paths without held-out tuning. Raw CLIP finds a relevant K=5 candidate for every held-out non-speech positive, while the calibration-only scalar cutoff causes 52.4% visual positive false abstention to reach 10% negative FAR. A calibration-only score-margin experiment rejects every held-out positive. Preserve CLIP as the candidate generator. Expand calibration sources, then test a compact pretrained image-text matching reranker with an explicit no-match score over the top five. Action Recognition, OCR, RAG, fine-tuning and a larger speech model remain unjustified by this evidence.

## 009 — Licensed, frozen scene-disjoint calibration pilot

Two disjoint scenes from CC BY 3.0 Big Buck Bunny have checksum-verified media and reviewed sampled-frame labels written before inference. Raw scores and environment are recorded; calibration refuses cross-split content/group leakage. The threshold is just above the largest calibration-negative cosine score and remains opt-in. It reduces held-out false accepts from four to two but is not probability calibration or broad accuracy. Natural-footage/source-disjoint and speech/hybrid labels remain future data work. Existing evidence does not justify OCR/action/RAG.

## 010 — One durable supervisor and reusable inference child

Keep FastAPI, local asset manifests and SQLAlchemy/SQLite or PostgreSQL. A SQL queue provides unique active stage keys, serialized capacity checks, compare-and-set claims, persisted retries/backoff and explicit failed-job retry. OS locks enforce a single-host supervisor and file writer. A reusable spawned child caches models; deadlines and parent-death monitoring terminate its process tree. Redis/Celery, modality services and Kubernetes are unnecessary at this scale. At-least-once replay is explicit; no distributed/HA claim. Inline mode remains for V1 compatibility.

## 011 — Operator auth and portable validation

Optional Basic/Bearer authentication protects APIs and media; Origin checks reject unexpected cross-origin writes. Browser-managed credentials keep passwords out of frontend storage. This is one operator, not RBAC or tenant ownership. Linux backend tests and disposable PostgreSQL migration/queue checks are executed through Docker. Compose binds loopback and requires passwords; public deployment and container ML remain outside verified scope.

## 001 — Local development first
Use Python 3.13, FastAPI and Next.js, with no external services in the foundation. Python 3.14 is installed but ML wheel support favors 3.13. Alternatives: container-only setup or mandatory PostgreSQL. Consequence: fewer initial dependencies; persistence and deployment arrive with concrete requirements.

## 002 — UI direction
Visual thesis: a quiet charcoal editing workspace with warm white typography and one lime accent. Content: library first, empty-state guidance, then upload/workspace as ingestion arrives. Interaction: focus feedback and short hover transitions, respecting reduced motion. No fabricated videos or search results.

## 003 — Portable media tools and bounded local jobs
Use imageio-ffmpeg for a packaged FFmpeg executable and OpenCV for basic metadata. System FFmpeg/ffprobe was absent. Alternatives: require installation or use PyAV throughout. Consequences: easy local setup, approximate VFR duration; FFmpeg licenses matter if binary redistribution is introduced.

## 004 — Video metadata manifests before relational speech storage
Phase 1 uses atomic per-video JSON manifests, keeping source assets and processing output together. A single-process lock bounds jobs and startup detects interruption. Alternatives: premature queue/Redis or SQLite schema immediately. Consequences: no multi-worker deployment; Phase 2 introduces SQLAlchemy/Alembic when transcript querying requires relational data. These manifests are local runtime artifacts, never committed.

## 005 — Optional tiny speech model and scoped relational persistence
Use faster-whisper tiny with CPU INT8; install via the speech extra and download on explicit transcription. Alternative: larger Whisper or mandatory transcription on every upload. Consequence: ingestion stays lightweight and model accuracy is limited. Actual smoke inference is required beyond mocked unit tests.

SQLAlchemy/Alembic now own transcript jobs and segments. Keep Phase 1 video manifests as asset metadata instead of migrating them without a query requirement. This refines decision 004: no duplicate authoritative video database. SQLite is the verified local path; PostgreSQL deployment remains future work.

## 006 — Pinned CLIP baseline and exact local retrieval
Use CLIP ViT-B/32 at a pinned Hub revision with Transformers 4.x and CPU PyTorch. Its ~600 MB download is justified by the first cross-modal feature; no larger model is needed. Compare normalized embeddings with FAISS IndexFlatIP. Alternatives: OpenCLIP/MobileCLIP or approximate indexes. Consequences: readable baseline and exact sampled-frame retrieval, bounded CPU batches, limited temporal understanding and no confidence calibration. Actual color-scene smoke passed; broader quality remains unmeasured.

## 007 — Rank fusion before learned ranking
Use BM25 (k1=1.2, b=0.75) for lexical speech and RRF (constant 60) for hybrid ranking. Alternatives: uncalibrated score addition or a learned reranker without labeled data. Consequence: interpretable evidence and reasonable untuned defaults; nearest-frame merging can conflate moments and lexical search misses synonyms. Preserve modality-specific scores and document candidate limits.

## 008 — Report synthetic evidence honestly
The first benchmark is a generated two-color video with two positives and one negative. Keep it reproducible and label it as a pipeline baseline. Alternatives: invent quality claims or adopt an unreviewed external dataset. Consequences: measured timings/metrics and visible negative-query failure, but no real-world retrieval claim. Advanced OCR/Q&A/training remain deferred until a concrete use case and evaluation data justify them.
