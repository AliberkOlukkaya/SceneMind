# SceneMind — final project assessment

**Audit date:** 2026-10-06 (Europe/Istanbul). **Audited baseline:** `main` at `5c5e909e598e12beebad8f43f6f80f9c40450b9c`. This is an assessment of the existing repository and recorded experiments, not a new retrieval run, feature change, release, or deployment. Percentages below retain each source report's denominator and protocol.

## 1. Executive summary

SceneMind is a local-first video moment finder. A user can upload a video or import a public direct-media/YouTube URL, wait for frame and transcript indexing, search with Visual, Spoken Content, or Smart mode, and click a timestamped thumbnail to seek the local video. Production uses five-second JPEG frames, a pinned CLIP image/text encoder and exact FAISS search; audio uses faster-whisper tiny and BM25; Smart merges speech and visual rankings with RRF60. The browser exposes Smart, Spoken, and Visual. Its default is **Smart**, while the API defaults to **Visual**; an explicit API `auto` route exists but the UI does not offer it. The local default can process inline; the Docker Compose configuration supplies a persistent SQL job worker.

The final frozen deployment acceptance used eight new source-disjoint videos and 124 positive queries. Its **R@5 was 88/124 = 71.0%**, where R@5 means the percentage of frozen queries with at least one timestamp inside a pre-annotated acceptable interval among the first five results. Speech reached 81.8%, Visual 63.6%, and Smart 68.0%. All predeclared aggregate/category quality gates failed; three videos had catastrophic source-level results, including a 40.38-minute English lecture whose audio was misidentified as Welsh. The interval annotations sometimes omit valid repeated moments and had no independent second human reviewer. The score is a reproducible diagnostic, neither generic “accuracy” nor verified click usefulness. Five negative queries still returned possible matches.

The engineering is substantial: bounded upload/URL acquisition, local inference, persistent indexes, durable-job design, real click-to-seek flow, and source-disjoint frozen evaluations. Direct-URL and upload indexing matched on a byte-identical fixture. A 40-minute source processed in 616.7 seconds with 3.34 GiB peak process-tree RSS in one local acceptance run, but its search quality was poor. Ask Video, OCR, object detection, X-CLIP temporal retrieval, and alternative fusion/rerankers remain research or disabled; their tested candidates did not justify promotion. Turkish routing and no-match behavior are not reliable enough for broad claims.

**Public production: no.** The frozen quality gates failed, and container build/runtime, HTTPS deployment, restart recovery, and backup restore are unverified. **Local/controlled portfolio demonstration: yes**, with explicit limitations and a prepared cache; an internet-facing hosted demo still needs operational validation and a clear “possible matches” UX. Engineering outcome: a credible, reproducible multimodal systems project. Product outcome: partial success, useful for some spoken and concrete visual lookups, not dependable general video understanding. Recommendation: **B — freeze ML architecture and prepare a controlled portfolio demo**, without claiming a v1.0 public release or spending another cycle tuning the used holdouts.

Primary evidence: [final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md), [frozen protocol](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_PROTOCOL.md), [deployment audit](FINAL_DEPLOYMENT_AUDIT.md), [project status](../PROJECT_STATUS.md), [tasks](../TASKS.md), and production files cited below.

## 2. Repository and release state

| Check at audit start | Verified state |
| --- | --- |
| Branch / HEAD / remote | `main`; `5c5e909e598e12beebad8f43f6f80f9c40450b9c`; equal to `origin/main`; clean working tree before this report. The documentation-only commit for this assessment necessarily advances HEAD. |
| Tags | `v1.0.0-rc1` only; annotated tag peels to `254a1a05ec8206e5d895e33f98a7b7db073481ce`. No `v1.0.0` tag. |
| GitHub Release | [Repository Releases page](https://github.com/AliberkOlukkaya/SceneMind/releases) displayed no releases on 2026-10-06. This is a point-in-time remote UI check. |
| Tracked files / sensitive artifacts | `git ls-files` audit: 470 tracked files, none above 10 MiB; no tracked media, database, NumPy index, or common weight extensions; no credential-pattern path was found. `.env.local` is untracked and ignored; `data/models` and `data/videos` are ignored. This path/extension/pattern audit is **not** a complete secret-content scan. No ignored secret value was read or printed. |

The release decision in [final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md) was to retain the release candidate and not create `v1.0.0`.

## 3. Production architecture verified against code

```text
Local upload ───────────────┐
Direct public media URL ────┼─> bounded acquisition/validation ─> UUID media storage
Public YouTube URL ─ yt-dlp ┘                                  │
                                                               ├─> 5 s / 480 px JPEGs
                                                               │    └─> CLIP vectors ─> FAISS IP
                                                               └─> 16 kHz mono audio
                                                                    └─> faster-whisper segments ─> BM25
                                                                                          │
                            Visual (CLIP) ──────┐                                         │
                            Spoken (BM25) ───────┼─> Smart: modality RRF60 ─> ranked moments
                            Smart (both) ────────┘                            └─> thumbnail + click-to-seek
```

The arrows correspond to [video extraction](../backend/app/video.py), [visual encoder](../backend/app/encoder.py), [speech pipeline](../backend/app/speech.py), [BM25/fusion](../backend/app/hybrid.py), [URL acquisition](../backend/app/url_ingest.py), [jobs](../backend/app/jobs.py), and [frontend search](../frontend/src/app/search.tsx). “Background” means worker-backed under Compose; the default local setting processes inline.

| Parameter | Current code |
| --- | --- |
| Sampling | First decoded frame, then the first frame at least five seconds later; actual decoded timestamps, 480-pixel width JPEGs. |
| Visual | `openai/clip-vit-base-patch32`, pinned revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; normalized 512-dimensional vectors; exact FAISS `IndexFlatIP` on normalized vectors. Cached under `data/models`, CPU default, batch size 8. |
| Speech | faster-whisper `tiny`, CPU/int8 default, four threads, beam 5, VAD; FFmpeg 16 kHz mono temporary audio; timestamped SQL transcript segments. Search uses BM25 with `k1=1.2`, `b=0.75`. |
| Smart | Up to 50 speech and 50 visual candidates; one vote per modality at a moment, reciprocal-rank fusion with constant 60; not a learned calibrated confidence. |
| Results | API `/search` default Visual with `k=10` and accepted `k=1..50`; frontend submits `k=8` and defaults Smart. Frozen acceptance explicitly used `k=5`. |
| Persistence/jobs | SQLite at `data/scenemind.db` and inline jobs are local defaults. PostgreSQL, durable SQL queue, separate worker, retries and orphan recovery are available via [Compose](../compose.yaml); Alembic migrations exist. Frames, transcript and indexes persist in the media directory/database. |
| Bounds | Upload defaults: 1 GiB, 60 minutes, 4K maximum; disk reserve 512 MiB plus 25% processing headroom. MP4, MOV, WebM, MKV, AVI inspection is local. |
| URL providers | Public HTTP(S) direct media and supported YouTube watch/short/share hosts; no arbitrary webpage crawler. |

**Documentation drift:** [older routing results](../ml/evaluation/QUERY_ROUTING_RESULTS.md) describe AUTO as a UI default. Current [frontend code](../frontend/src/app/search.tsx) defaults to Smart and exposes only Smart/Spoken/Visual; `auto` remains callable through the API. [Older personal acceptance](../ml/evaluation/PERSONAL_ACCEPTANCE_RESULTS.md) cites 250 MiB/30-minute limits that were subsequently raised; current [config](../backend/app/config.py) governs. Historical reports remain unchanged so their original experimental context is preserved.

## 4. What a user can do now

| Capability | Status | Evidence/limit |
| --- | --- | --- |
| Local video upload; public direct-media URL; YouTube import | IMPLEMENTED | Bounded acquisition and local storage; provider availability and rights remain external. |
| Processing status, video library, thumbnail/timestamp results, click-to-seek | IMPLEMENTED | [frontend](../frontend/src/app/page.tsx), API routes and prior desktop/mobile browser flow. |
| Visual Content, Spoken Content, Smart Search | IMPLEMENTED | UI modes map to CLIP, BM25, and RRF respectively. Results are “possible matches,” not guaranteed matches. |
| Up-to-60-minute video ingestion | IMPLEMENTED | Enforced duration bound and measured long-video runs; **retrieval usefulness at that length is not established**. |
| Explicit API `auto` routing | INTERNAL ONLY | Route exists and config flag defaults true; no UI selector and poor Turkish generalization. |
| Ask Video / grounded Q&A | DISABLED | API returns 503 under `qa_enabled=False`; UI displays disabled state. Optional OpenAI key is not needed for Find Moments. |
| OCR, object detection, X-CLIP, semantic/hierarchical QA, alternate rerank/fusion | EXPERIMENTAL | Evaluation code/reports exist; none is a user-facing production capability. |
| Reliable no-match or verified Turkish-language experience | NOT AVAILABLE | Negative queries return ranked moments; Turkish compatibility gate failed. |

## 5. Experimental/disabled feature disposition

| Branch | Purpose and decisive evidence | Code retained? / user-facing? |
| --- | --- | --- |
| Ask Video | Answer from retrieved evidence; final core test answered 30/36 answerable questions (83.33%, below 90% gate) despite 36/36 evidence R@5. Disabled for answer correctness. | Yes, [QA route](../backend/app/qa.py); UI disabled. |
| OCR | Search words in sampled frames; validation OCR R@5 63.64%, audited character accuracy 84.84%, below recognition gate. | Yes, evaluation branch; no production index/UI. |
| AUTO router | Infer route from query; English routing looked strong on an earlier small set, but Turkish held-out route accuracy 80% and personal acceptance exposed mismatches. | Yes, [router](../backend/app/routing.py); explicit API only. |
| X-CLIP | Encode short temporal clips; frozen R@5 25.76% versus CLIP 60.61%, much higher CPU/index cost. | Yes, evaluation code; not production. |
| YOLOX/RT-DETR | Improve small-object detection; conditional detection improved but candidate retrieval, costs, and closed-set coverage did not support integration. | Yes, evaluation code; not production. |
| Alternate fusion/rerankers | Rescue mixed evidence or no-match; holdout/category regressions or false abstention prevented promotion. | Yes, evaluation code; production stays uncapped RRF60. |
| Semantic transcript retrieval | Improve evidence recall; 45-query validation answer success fell from 69.70% to 63.64%. | Yes, QA research; not user-facing. |
| Hierarchical video memory | Better long-video organization; evidence completeness and answer success fell against BM25 (84.85% vs 96.97%; 57.58% vs 78.79%). | Yes, QA research; not user-facing. |

See [final Ask Video](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_RESULTS.md), [OCR](../ml/evaluation/OCR_EVIDENCE_V1_RESULTS.md), [Turkish](../ml/evaluation/TURKISH_COMPATIBILITY_RESULTS.md), [temporal](../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_RESULTS.md), and [fusion](../ml/evaluation/EVIDENCE_PRESERVING_FUSION_V1_RESULTS.md).

## 6. Final production retrieval result

The [single frozen acceptance run](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md) used **8 source-disjoint videos**, **124 positive queries** (44 Speech, 55 Visual, 25 Smart), and **5 separate negative diagnostics**. [Manifest](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_MANIFEST.json) SHA-256: `d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196`. The protocol was frozen at `c85f1ac` before the run. A hit required a returned timestamp inside an annotated interval. **R@5 asks what fraction of queries had at least one such timestamp in the first five results.** MRR@5 is reciprocal rank of the first such hit, averaged over all positives, with misses contributing zero.

| Mode | n | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Speech | 44 | 68.2% | 81.8% | 81.8% | 74.2% |
| Visual | 55 | 30.9% | 56.4% | 63.6% | 44.1% |
| Smart | 25 | 36.0% | 60.0% | 68.0% | 47.1% |
| **Overall** | **124** | **45.2%** | **66.1%** | **71.0% (88/124)** | **55.4%** |

| Video | n | R@5 |
| --- | ---: | ---: |
| Aikido demonstration | 8 | **37.5% — catastrophic** |
| Heritage museum | 8 | 87.5% |
| Manchester street | 8 | 100.0% |
| Museum accessibility | 26 | 96.2% |
| OpenRefine tutorial | 26 | 76.9% |
| Parliament lecture | 32 | **50.0% — catastrophic** |
| PPE tutorial | 8 | 75.0% |
| Social-media studio talk | 8 | **37.5% — catastrophic** |

“Catastrophic” was predeclared as a source with at least eight queries and R@5 below 60%. The predeclared overall R@1/3/5 gates were 75/85/90%, category R@5 gates Speech/Visual/Smart 90/85/90%, long-video R@5 85%, and no catastrophic source. All aggregate/category and long/source gates failed.

## 7. Confidence and limits of the 71.0% figure

The sources were new relative to tracked evaluation media by title/hash checks. Queries came from source-video contact sheets and publisher captions, **not** SceneMind retrieval or ASR output. Agent review established intervals, but there was no independent second human adjudicator and no exhaustive frame-by-frame/repeated-moment labeling. At least five misses were assigned annotation issues, and the “multiple similar moments” group can include further missing acceptable intervals. Frozen labels were not revised after seeing results. Therefore the interval rule may undercount useful repeated scenes, while the lack of independent click judgment means it may also misrepresent actual user satisfaction. The set has only eight sources and unequal category/source mixes, so it is not a population estimate for all videos or languages.

The number establishes that this specific production pipeline put a frozen acceptable interval in the first five for 88 of 124 positive queries under that protocol, and that it missed the declared release gates. It does **not** establish 71% general accuracy, 71% real-user satisfaction, 71% Turkish performance, or a reliable no-match capability. All five absent-concept diagnostics returned five possible results with HTTP 200; there is no validated abstention. A new independent human-reviewed source set would be required for a public usefulness claim. [Acceptance protocol/results](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md).

## 8. Failure audit and top production problems

The final report assigns one **provisional dominant cause** to each of 36 Top-5 misses; these are diagnostic labels, not causal counterfactual experiments.

| Cause | Count | Interpretation |
| --- | ---: | --- |
| `TRANSCRIPTION_ERROR` | 12 | ASR/language detection; English lecture marked `cy`, corrupting Speech and Smart evidence. Model/preprocessing failure, not a mere UI bug; a language-override remedy was not tested on a fresh set. |
| `MULTIPLE_SIMILAR_MOMENTS` | 10 | Ranker may choose a plausible repeated event outside a singly labeled interval; mixes retrieval difficulty with annotation incompleteness. Needs better evaluation annotation before calling all 10 model failures. |
| `ANNOTATION_ISSUE` | 5 | Broad/repeated relevant scenes omitted from accepted intervals. Evaluation defect; do not quietly change the spent holdout. |
| `VISUAL_SEMANTIC_MISS` | 5 | CLIP frame/text mismatch on UI and scenes; representation/ranking limitation. |
| `SMALL_OBJECT` / UI | 3 | Tiny facet/value/spectacles details; five-second/480-pixel sampling and image-text representation limit. |
| `HYBRID_RANKING` | 1 | A plausible OpenRefine matching-status result fell outside fused Top-5; diagnostic, not isolated proof of a universal fusion bug. |
| `SAMPLING` or other separate final category | 0 separately assigned | The 36 labels are exhausted above; other ablations show sampling losses, but adding them to this taxonomy would double-count or invent causes. |

The largest actionable **production** concerns are (1) ASR language robustness on long speech sources, (2) weak frame-level visual semantics/small details, and (3) uncalibrated ranking without no-match behavior. Annotation quality is an additional **evaluation** blocker. Some failures may be fixed by engineering, but the evidence does not identify a tested, generalizable remedy that clears release gates.

## 9. Speech Search

Final frozen Speech R@1/3/5 = **68.2/81.8/81.8%**, MRR@5 **74.2%** across 44 queries. It beats Visual R@5 by 18.2 percentage points on **different query types**, so this is descriptive, not a controlled modality comparison. Speech benefits from literal spoken words and BM25 token overlap; images require CLIP to map semantic descriptions to sparse frames. The 40.38-minute English Parliament lecture was automatically labeled Welsh (`cy`) and its early transcript corrupted, explaining eight Speech misses and contributing to four Smart misses. The remaining Speech miss profile can also involve segmentation, paraphrase/token mismatch, ambiguous repeated utterances and ranking; the final taxonomy does not isolate exact counts for each. [Final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md), [speech code](../backend/app/speech.py), [BM25 code](../backend/app/hybrid.py).

## 10. Visual Search

Final frozen Visual R@1/3/5 = **30.9/56.4/63.6%**, MRR@5 **44.1%** (55 queries). Five-second sparse, 480-pixel static frames lose short or tiny events; CLIP's single-frame embedding also fails to represent interactions, temporal order and exact UI text. The [small-object ablation](../ml/evaluation/SMALL_OBJECT_ABLATION_RESULTS.md) directly verified four event visibilities: visible evidence at 5s **25%**, 2s/1s **100%**, with frame counts 47/116/230; yet CLIP small-object candidate R@5 was only 25/25/50%. Thus sampling can dominate specific events, but denser frames alone do not solve semantic ranking.

On a **different new frozen temporal validation** (5 sources, 66 queries), production CLIP R@5 was **40/66 = 60.61%** versus X-CLIP **17/66 = 25.76%**. A **historical post-decision diagnostic replay**, restricted to 42 queries from six older short visual sources, found paired CLIP **31/42 = 73.81%** versus X-CLIP **15/42 = 35.71%**; it must not be confused with the new validation or the full old 55-query Visual result. In the new baseline's 26 misses, 12 were classified sampling and 14 ranking; X-CLIP introduced 46 ranking/localization misses among its 49. The tested temporal-model replacement hypothesis failed, without proving all video models impossible. [Temporal report](../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_RESULTS.md).

## 11. Smart Search

Smart runs BM25 and CLIP, gathers up to 50 candidates from each, aligns speech hits to nearby thumbnails, and combines modality ranks with `1/(60 + rank)` per vote. It is an uncalibrated rank fusion, not an answer model. Its final frozen R@1/3/5 = **36.0/60.0/68.0%**, MRR@5 **47.1%** for 25 mixed queries. Combining modalities can surface a moment supported by both, but weak additional candidates can displace strong single-modality evidence at Top-5. The [cap fusion holdout](../ml/evaluation/HYBRID_FUSION_HOLDOUT_V1.md) moved Top-5 from 21/28 to 20/28 and Speech from 8/11 to 7/11; [calibrated fusion](../ml/evaluation/CALIBRATED_FUSION_V1_RESULTS.md) did not validate its confidence estimates; [evidence-preserving fusion](../ml/evaluation/EVIDENCE_PRESERVING_FUSION_V1_RESULTS.md) improved overall 21/24 to 22/24 but regressed multimodal MRR, so it was rejected. The English AUTO prototype once had high small-set route accuracy, but [Turkish compatibility](../ml/evaluation/TURKISH_COMPATIBILITY_RESULTS.md) and [personal acceptance](../ml/evaluation/PERSONAL_ACCEPTANCE_RESULTS.md) did not support a default UI router. Current UI defaults Smart and does not expose AUTO; explicit API auto remains.

## 12. Long-video support

| Evidence | Measured processing and resources | What it establishes |
| --- | --- | --- |
| Frozen 40.38-minute Parliament lecture | 2,422.8 s, 485 frames, 442 transcript segments; 35.8 s ingest + 551.5 s Whisper + 29.4 s CLIP = **616.7 s**; **3,338.4 MiB** peak process-tree RSS in local single-process TestClient; **164.0 MiB** persistent UUID media directory. R@1/3/5 **37.5/50.0/50.0%**. | Full pipeline completes but usefulness is poor, chiefly from `cy` ASR. |
| 22.83-minute real silent video | 419 MiB input, 274 frames, 54.174 s ingest + 67.162 s CLIP = 121.335 s, 1.149 GB peak worker RSS, 444.8 MB disk. | Large-file video path works; no speech-quality claim. |
| 45-minute repeated audio stress fixture | 540 frames, 373 segments, 43.792 s ingest + 160.569 s Whisper + 41.033 s CLIP = 245.394 s; 3.018 GB peak worker RSS, 173.378 MB disk. | Durable worker/resource stress, **not** independent 45-minute retrieval accuracy. |

The configured ceiling is 60 minutes; the reports do not certify high-quality retrieval for arbitrary 30–60-minute videos, all codecs, simultaneous jobs, or an 8-GB hosted machine. [Acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md), [long-video stress](../ml/evaluation/LONG_VIDEO_INGEST_RESULTS.md), [support guide](LONG_VIDEO_SUPPORT.md).

## 13. URL ingestion

**Assessment: STRONG as an engineering subsystem, with deployment caveats.** Direct public HTTP(S) media acquisition validates scheme, credential-free URLs, resolved public IPs and local media type; it rechecks each direct redirect (max five), bounds transfer size/time, deletes `.part` files on failure, and reinspects duration/format before ingest. YouTube watch/Shorts/share hosts use pinned `yt-dlp==2026.8.19`, a `<=720p` format selection, no playlist/cookies/DRM handling, bounded transfer and local reinspection. Provider ID/provenance metadata supports duplicate detection. YouTube's internal media subrequests are handled by yt-dlp, so the same per-redirect direct-media validation is not proven for every internal request; public egress policy remains important. [URL code](../backend/app/url_ingest.py), [URL guide](URL_INGESTION.md).

The [local-vs-URL equivalence](../ml/evaluation/URL_INGESTION_V1_EQUIVALENCE.md) used byte-identical 35,611,959-byte, 219.443-second input: 44 frames/timestamps, 60 transcript segments, 44×512 visual index and all nine frozen Top-5 timestamp lists matched. A separate live public YouTube check acquired 634.57 seconds, made 127 frames, indexed/searched, served HTTP 206 video, and the actual browser sought to 490 seconds on desktop/mobile. Equivalence is one fixture, not universal provider reliability or rights assurance.

## 14. Ask Video / Q&A research

| Chronological milestone | Critical validation observation | Decision |
| --- | --- | --- |
| [Grounded Q&A V1](../ml/evaluation/GROUNDED_VIDEO_QA_V1_RESULTS.md) | Evidence R@5 100% on a small set, but answer correctness 88.89%, grounded 80%, unsupported claims 8.70%, correct abstention 66.67%. | Fail safety/quality; remain disabled. |
| [Abstention Safety](../ml/evaluation/QA_ABSTENTION_SAFETY_V1_RESULTS.md) | Validation evidence 94.44%, answer 77.78%; grounded/citation precision 100%, unsupported 0, correct abstain 100%, false abstain 11.11%; list/count 0/2. | Safer, not answer-complete. |
| [Structured Question Evidence](../ml/evaluation/QA_STRUCTURED_QUESTION_EVIDENCE_V1_RESULTS.md) | Evidence R@5 92.31% unchanged; answer 72.22%, unsupported claims 20.75%, correct abstain 70%. | Regression/reject. |
| [Final Ask Video Core](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_RESULTS.md) | Four new videos, 60 questions; evidence R@5 36/36 answerable; answer 30/36 = 83.33% versus 90% gate; grounded/citation precision 100%, unsupported 0, correct abstain 100%, false abstain 4/36. | Core answer quality fails; `qa_enabled=False`. |
| [Semantic transcript evidence](../ml/evaluation/SEMANTIC_HIERARCHICAL_QA_V1_RESULTS.md) | Validation evidence R@5 90.91% vs BM25 81.82%, but answer success 63.64% vs 69.70%. | Better evidence did not improve answers. |
| [Hierarchical memory](../ml/evaluation/HIERARCHICAL_VIDEO_MEMORY_V1_RESULTS.md) | Section R@3 90.91%, yet evidence completeness 84.85% vs BM25 96.97%, answer success 57.58% vs 78.79%. | Reject, including long-video weakness. |

The key distinction is evidence retrieval versus a complete, correctly scoped answer. Abstention improved unsupported-claim safety, but list/count, temporal, and complex answers remained unreliable. The optional [OpenAI-backed QA route](../backend/app/qa.py) is disabled by default and is outside the local-only Find Moments requirement.

## 15. OCR evidence research

The [OCR V1 source-disjoint validation](../ml/evaluation/OCR_EVIDENCE_V1_RESULTS.md) used four development and four validation sources, 22 text events on 89 sampled five-second/480-pixel frames. Sampling covered **21/22 = 95.45%** events; conditional text detection was **19/21 = 90.48%**. Character accuracy was **84.14%** under frozen labels, **84.84%** after audit, below the 85% gate. OCR retrieval R@1/3/5 was **59.09/63.64/63.64%**; false-text rate **8.75%**. CPU OCR median/p95 was **521/2,304 ms per frame**, with **+356.8 MiB** observed RSS; index size **20,671 bytes**, about 2,906 bytes/source-minute. Decision B: detection works better than recognition/retrieval; no OCR evidence index was promoted.

## 16. Temporal video retrieval research

[X-CLIP V1](../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_RESULTS.md) selected one pinned `microsoft/xclip-base-patch32` model with eight RGB frames per 4-second clip and 2-second stride on **five development sources, 48.38 minutes, 66 queries**. A new **source-disjoint five-source frozen validation** had **70.03 minutes, 66 queries**, including a 31.93-minute source. Its baseline CLIP R@5 was **40/66 = 60.61%**; X-CLIP **17/66 = 25.76%**, **−34.85 percentage points**. All five sources failed the 60% source floor. The separate historical 42-query replay was **73.81% CLIP vs 35.71% X-CLIP**, not another frozen validation.

For the baseline's 26 misses, pixel-reviewed diagnosis assigned **12 sampling** and **14 ranking**; X-CLIP had **49 misses**, mostly **46 ranking/localization**, plus two timestamp-midpoint coverage and one ambiguity. Candidate processing repeatedly touched 16,784 frames for 2,098 overlapping clips versus 842 baseline frames; persisted features were **216.44 MB vs 1.725 MB (~125×)**. Completed-stage indexing was **40.38 vs 1.58 seconds/source-minute (~25.6×)**; query p95 **1.493 vs 0.047 seconds**. Whole-run peak RSS was unavailable due sampler failure/interruption, so no memory-feasibility certification is inferred. Decision C rejected this tested model/configuration. Better temporal representation in principle remains open, but a clip model did not fix SceneMind's ranking and localization bottleneck at acceptable evidence/cost.

## 17. Rejected experiments at a glance

| Experiment | Goal | Most informative result | Decision / why not production |
| --- | --- | --- | --- |
| [BLIP verifier](../ml/evaluation/VERIFIER_RESULTS.md) | Verify/rerank visual hits | Unthresholded R@5 stayed 85.7%; threshold R@5 26.2%, false abstention 52.4%; median 3.352 s. | Reject: no recall gain, bad abstention/cost. |
| [UForm pair scorer](../ml/evaluation/LIGHTWEIGHT_PAIR_SCORER_RESULTS.md) | Lightweight visual score | Unthresholded R@5 unchanged; calibrated R@5 50%, false abstention 47.6%; median 202.1 ms. | Reject: no robust quality gain. |
| [YOLOX / RT-DETR](../ml/evaluation/SMALL_OBJECT_ABLATION_RESULTS.md) | Rescue small objects | On 18 verified-visible frames, Nano 416/640/768 recall 66.7/83.3/83.3%, RT-DETR-R18 640 88.9%; stronger model p95 362.8 ms, +552.5 MiB. | Do not promote: narrow closed-set gains, sampling/retrieval remain; earlier detector gate failed. |
| [Secondary sampling](../ml/evaluation/BOUNDED_SECONDARY_RESULTS.md) | Query-gated denser frames | R@5 baseline 85.7%, 2s 76.2%, bounded 81.0%; false abstention 81%, +2.54s p95. | Reject: worse retrieval/cost. |
| [Temporal diversity/coarse candidate](../ml/evaluation/COARSE_CANDIDATE_RESULTS.md) | Spread candidates | R@5 80.95% vs raw 85.71%; 2.40× index, 20pp Speech regression. | Reject: loses useful matches. |
| [Candidate-list ranker](../ml/evaluation/CANDIDATE_LIST_RANKING_RESULTS.md) | Learned Top-K/order/no-match | Selected R@5 76.2% vs raw 85.7%; 90.48% false abstention at FAR 0. | Reject: model fails despite Top-50 oracle opportunity. |
| [AUTO router](../ml/evaluation/TURKISH_COMPATIBILITY_RESULTS.md) | Select mode automatically | Turkish held-out 80% route, under 85% gate; personal test had 12 Turkish mismatches. | Reject as UI default; explicit API path remains. |
| [English no-match](../ml/evaluation/PATH_AWARE_NO_MATCH_RESULTS.md) | Abstain on absent concepts | Path-aware R@5 fell 87.5→41.67%; Visual false-abstention 83.33%. | Reject: harms positives. |
| [Calibrated fusion](../ml/evaluation/CALIBRATED_FUSION_V1_RESULTS.md) | Calibrate modality scores | Validation calibration AUC below rank-only for visual (0.6863 vs 0.7012) and speech (0.6454 vs 0.7644). | Reject before holdout use. |
| [Evidence-preserving fusion](../ml/evaluation/EVIDENCE_PRESERVING_FUSION_V1_RESULTS.md) | Protect strong evidence | R@5 21/24→22/24 but multimodal MRR regressed. | Reject mixed-category tradeoff. |
| [Ask Video](../ml/evaluation/FINAL_ASK_VIDEO_CORE_ACCEPTANCE_V1_RESULTS.md) | Grounded answers | 83.33% answer correctness vs 90% gate. | Disabled. |
| [OCR](../ml/evaluation/OCR_EVIDENCE_V1_RESULTS.md) | Text moments | R@5 63.64%, audited char accuracy 84.84%. | Research only. |
| [X-CLIP](../ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_RESULTS.md) | Temporal backbone | Frozen R@5 25.76% vs CLIP 60.61%, ~125× feature bytes. | Reject. |

These experiments use different data/protocols; their percentages are **not** a unified leaderboard. The small-object visibility-conditioned detector metric is not end-to-end search recall. Check each linked report for denominators and gates.

## 18. What actually worked

- **Local multimodal search pipeline:** real upload, frame/audio inference, indexes, and timestamped results completed in frozen acceptance, with no paid API requirement for Find Moments.
- **Acquisition and provenance:** URL validation, transfer bounds, cleanup and provider metadata are implemented; one byte-identical URL/upload fixture produced the same frame/transcript/index/search outputs, and a live YouTube flow completed.
- **Worker/persistence design:** SQL job states, retries, orphan recovery, Alembic and media/index persistence exist and have local tests; this is an engineering success, while deployed restart/restore still needs proof.
- **Long-video processing:** measured 40-minute real and 45-minute stress paths completed within configured 60-minute bounds. The result establishes processing, not good long-video retrieval.
- **Usable interaction:** browser tests and a real-model desktop/mobile URL flow verified thumbnails, status and seek-to-result behavior. Conservative “possible matches” wording avoids pretending the ranker has no-match certainty.
- **Evaluation discipline:** frozen source-disjoint manifests, explicit gates, negative diagnostics, timestamp-level failure audits and rejection of attractive but regressive models preserved a production baseline rather than silently tuning holdouts.

Evidence: [final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md), [URL equivalence](../ml/evaluation/URL_INGESTION_V1_EQUIVALENCE.md), [deployment audit](FINAL_DEPLOYMENT_AUDIT.md), [long-video results](../ml/evaluation/LONG_VIDEO_INGEST_RESULTS.md).

## 19. What did not work

Universal visual retrieval is unsupported (63.6% final Visual R@5, and X-CLIP regressed on a different frozen set). Reliable no-match is absent (five negative diagnostics still returned five results). AUTO generalization is unproven for bilingual use and is absent from UI. OCR retrieval and Ask Video missed their gates and remain disabled/research. Turkish personal acceptance was weaker than English (historical useful Top-5 **77.78% vs 88.89%**, different from final frozen interval R@5) with route mismatches; the subsequent Turkish compatibility gate also failed. The 40-minute lecture's Welsh ASR error shows that long-video completion is not long-video usefulness. [Personal acceptance](../ml/evaluation/PERSONAL_ACCEPTANCE_RESULTS.md), [Turkish](../ml/evaluation/TURKISH_COMPATIBILITY_RESULTS.md), [final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md).

## 20. ML/AI substance

| Category | Components and role |
| --- | --- |
| Deep learning, **production inference** | Pretrained CLIP for image/text embeddings; pretrained faster-whisper tiny for speech-to-text. |
| Deep learning, **research only** | X-CLIP, BLIP, UForm, YOLOX/RT-DETR and semantic MiniLM-based QA evidence where tested. |
| Classical ML | Former tiny learned AUTO softmax router; candidate-list/calibration experiments. Explicit API auto remains, but not UI default. |
| Information retrieval | BM25 over timestamped transcript segments; RRF60 combining ranks. |
| Vector search | Exact FAISS inner-product index of normalized CLIP vectors. FAISS is an index, not itself a trained understanding model. |
| Deterministic software | FFmpeg decode/resample, five-second sampling, URL checks, job state, persistence, UI seek, interval scoring. |
| LLM, **disabled optional QA** | OpenAI-backed answer generation behind `qa_enabled=False`; no key or paid API is required for production Find Moments. |

SceneMind **does not train a production foundation model from scratch**. Its main technical content is pretrained inference, retrieval, evaluation and systems engineering.

## 21. Deployment status

| Item | Evidence / state |
| --- | --- |
| Dockerfile and Compose | Present. Dockerfile uses Python 3.13 slim plus speech/visual extras; [Compose](../compose.yaml) specifies PostgreSQL 17, API bound to localhost, worker and shared persistent volumes. Compose has no frontend or HTTPS endpoint. |
| Image build / container runtime | **Not verified**: deployment audit lacked a working Docker daemon. Compose configuration syntax passed historically; this is not a build or runtime test. |
| PostgreSQL / worker | SQL support, migration and durable worker implemented/tested locally; full deployed multi-container runtime and restart behavior unverified. |
| Persistent storage | Named DB/media/cache volumes configured; local persisted frame/index behavior tested. Deployed restore and backup unverified. |
| HTTPS / restart / restore | No verified HTTPS deployment, running-container restart recovery, or backup/restore drill. |
| Health/CORS/env | `/health` checks API liveness only, not queue/worker readiness. CORS origins are configurable; frontend API URL is build-time. Compose requires an operator token; local auth defaults off. |
| FFmpeg/model cache | `imageio-ffmpeg` provides executable; models cached on data volume, but first-use network/download and image startup have not been container-validated. |

**Public production ready: NO. Controlled local portfolio demo ready: YES. Hosted portfolio demo operationally ready now: NO**, pending actual image build, HTTPS/auth configuration, restart/restore and capacity validation. [Deployment audit](FINAL_DEPLOYMENT_AUDIT.md), [guide](DEPLOYMENT.md).

## 22. Resources: measured, estimated, recommended

| Class | Values and scope |
| --- | --- |
| **MEASURED** | Final 40.38-minute run: 616.7 s processing, 3,338.4 MiB peak process-tree RSS, 164.0 MiB persistent media; 485 frames/442 segments. Warm TestClient search median/p95 23.9/45.5 ms; not deployed concurrent latency. Production CLIP + Whisper tiny cache about 1.292 GB at audit time. Real silent 22.83-minute video: 1.149 GB peak worker/444.8 MB disk. 45-minute repeated-audio fixture: 3.018 GB peak worker/173.378 MB disk. CPU-only local runs; no GPU requirement or benchmark. |
| **ESTIMATED** | [Deployment guide](DEPLOYMENT.md) suggests **4 vCPU, 8 GB RAM and tens of GB disk** as an operational starting point; this is **not a tested minimum**. Concurrent uploads, provider downloads and first-use model fetching can increase needs. |
| **RECOMMENDED, not measured** | Size host volumes for source copies, temporary downloads/audio, JPEGs, vectors, DB and persistent model cache; monitor worker RSS/queue/disk; validate on the actual deployment hardware before capacity promises. No universal minimum can be derived from one workstation run. |

## 23. Security posture

Upload streams are bounded before local format/duration inspection; source-owned UUID directories and atomic writes reduce cross-job/path hazards. Direct URL handling rejects credentials, non-public resolved addresses, private/loopback/metadata targets and unsafe direct redirects; `.part` downloads and temporary media/index stages are cleaned on failures. API errors are designed not to expose command output/internal paths. Exact CORS origins and optional single-token Basic/Bearer auth exist; `.env.local`/`data/` are ignored, and Find Moments does not need an OpenAI key. The optional QA key must remain server-side. [Security/deployment audit](FINAL_DEPLOYMENT_AUDIT.md), [URL implementation](../backend/app/url_ingest.py), [auth](../backend/app/auth.py).

Known public-service blockers: local default is unauthenticated; Compose token is a single operator credential, not tenant identity; there is no built-in HTTPS, public rate limit/quota, proven egress isolation for every yt-dlp internal subrequest, deployed backup/restore, or full secret-content audit. These are reasons to keep deployment controlled, not evidence of a known exploit.

## 24. Test and quality status

**Fresh on audited HEAD (`5c5e909`), 2026-10-06:** `.venv/Scripts/python.exe -m pytest -q --basetemp=data/test-final-assessment-1` yielded **313 passed, 1 skipped**, with two upstream deprecation warnings; Ruff check of `backend tests scripts ml` passed; frontend `npm run lint`, `npx tsc --noEmit`, and `npm run build` passed. These checks did not rerun frozen ML benchmarks. This report-only commit does not alter code.

**Historical, not freshly rerun:** [final acceptance](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md) documented **18 passed Playwright** desktop/mobile cases, including a real-model click-to-seek; [deployment audit](FINAL_DEPLOYMENT_AUDIT.md) documented passing Compose config and **282 passed, 1 skipped** backend tests then. The newer 313-pass count is current unit/integration status, not a new acceptance or browser run. Frozen regression/evaluation results remain the source reports cited here. Docker image/runtime and deployed checks remain unexecuted.

## 25. Codebase health and debt

| Severity | Evidence-backed concern |
| --- | --- |
| HIGH | Stale statements in earlier routing/personal-acceptance docs conflict with current UI default and upload limits. Readers must use code/current status for current behavior. |
| HIGH | First-use model/provider downloads and Docker image startup are unvalidated in a container; public deployment reliability cannot be assumed. |
| MEDIUM | Many rejected research branches and feature flags remain in the repository, including disabled QA and API-only AUTO. They enlarge maintenance/security review surface even while unavailable in UI. |
| MEDIUM | Large modules such as `backend/app/qa.py` (734 lines), `url_ingest.py` (556) and `video_memory.py` (367), plus parallel evaluation harnesses, are harder to audit/change safely. This assessment does not recommend refactoring before a scoped need. |
| MEDIUM | Pinned model revision and `yt-dlp` improve reproducibility but rely on external cache/provider availability; Windows development and Linux container/FFmpeg behavior need end-to-end parity checks. |
| MEDIUM | CPU cost is material: the 40-minute Whisper stage took 551.5 s; X-CLIP/OCR candidates were costlier. Concurrent worker sizing is unmeasured. |
| LOW | Local `.env.local` and data paths are ignored, but there is no evidence of an exhaustive tracked-content secret scanner in this audit. |

## 26. Portfolio value (judgment, 0–10)

| Dimension | Score | Basis |
| --- | ---: | --- |
| ML engineering | 8 | Pinned pretrained inference, CPU profiles and disciplined alternative-model experiments; no novel trained model. |
| Software engineering | 8 | End-to-end API, persistent media/indexes, guarded acquisition, jobs and usable UI; complex modules need maintenance. |
| Evaluation discipline | 8 | Frozen source-disjoint protocols, held-out gates and failure audits; independent human annotation and repeated-moment completeness missing. |
| Deployment engineering | 5 | Docker/Compose, migrations and worker design exist, but image/runtime, HTTPS and recovery are unverified. |
| Product completeness | 5 | A real moment-search workflow exists; no-match, Turkish, long-video usefulness and Q&A remain weak/unavailable. |
| Technical originality | 5 | Integration and evaluation are thoughtful; core encoders/index/fusion are established methods. |
| Interview value | 9 | Strong evidence of architecture, measurement, failed-hypothesis handling and honest release decisions. |

These are editorial ratings, not benchmark scores.

## 27. Honest public claims

**Safe claims:** “Built a local-first multimodal video retrieval prototype using Whisper, CLIP, FAISS, BM25 and RRF.” “Implemented bounded local/URL/YouTube ingestion, persistent indexes, timestamp search and click-to-seek.” “On a frozen eight-video, 124-positive-query acceptance set, 88/124 (71.0%) had an annotated acceptable timestamp in Top-5; Speech/Visual/Smart R@5 were 81.8/63.6/68.0%, with annotation and source-size limits.” “Rejected X-CLIP and several reranking/OCR/QA candidates after held-out regressions.”

**Do not claim:** “Production-ready universal video understanding,” “90%+ retrieval accuracy,” “71% user success,” “reliable Turkish/negative detection,” “Ask Video or OCR ships,” “X-CLIP improved retrieval,” “proven scalable 4-vCPU/8-GB deployment,” “validated public HTTPS service,” or “fully human-verified benchmark.” State local demo and release-candidate status accurately.

## 28. Final scorecard (judgment, 0–10)

| Component | Score | Short basis |
| --- | ---: | --- |
| Speech Search | 7 | 81.8% frozen R@5; language detection caused major long-source failures. |
| Visual Search | 5 | 63.6% frozen R@5; sparse frames and semantics limit coverage. |
| Smart Search | 5 | 68.0% frozen R@5; uncalibrated fusion can displace evidence. |
| URL ingestion | 8 | Defensive implementation plus byte-identical equivalence and live YouTube flow; provider/deployment caveats. |
| Long-video engineering | 7 | 40–45-minute processing completed; retrieval quality and concurrency unproven. |
| Frontend UX | 7 | Library, modes, status and verified seek; possible-match uncertainty remains. |
| Backend architecture | 8 | Guarded ingestion, persistent index and durable-job option; complexity remains. |
| Evaluation methodology | 8 | Frozen/source-disjoint and failure audits; annotation independence gap. |
| Security | 6 | Good URL boundaries and local secret handling; public auth/TLS/egress gaps. |
| Deployment readiness | 4 | Configured artifacts, no container/HTTPS/recovery proof; quality gate fails. |
| Ask Video | 3 | Good evidence/safety progress, core answer gate failed and feature disabled. |
| OCR | 4 | Detection promising, recognition/retrieval below gate and not integrated. |
| **Overall product** | **5** | Useful bounded demo, not dependable general-purpose/public release. |

## 29. Why the ≥90% R@5 goal was missed

The goal was an **overall frozen interval R@5 gate**, not a single-model metric. The 19-point gap from 90% to 71% is not attributable to one cause. First, the long lecture's English→Welsh ASR error produced 12 of 36 provisionally classified misses and depressed the long-source score to 50%. Second, sparse five-second, 480-pixel stills and CLIP's frame-level semantics miss tiny, transient and compositional evidence; the small-object visibility ablation directly demonstrated sampling loss, while the temporal validation found both sampling (12) and ranking (14) among CLIP's 26 misses. Third, exact nearest-neighbor visual ranking and BM25/uncalibrated RRF cannot reliably distinguish many similar moments or abstain; alternate fusion, rerank and temporal candidates did not establish a broad fix. Fourth, broad queries and incomplete repeated-moment intervals created evaluation ambiguity: at least five explicit annotation misses plus possible overlap with the ten similar-moment cases. Fifth, eight heterogeneous sources and difficult UI/action/negative cases expose task difficulty that the initial architecture does not cover. CPU resource limits made denser/heavier options more expensive, but the tested heavy options also failed quality; resource cost is not a sufficient explanation. Deployment gaps do not cause the measured retrieval misses, though they independently block public release.

**Importance by current evidence:** ASR/language robustness on the long source; visual representation plus sampling/ranking; repeated-moment ambiguity/annotation quality; fusion/no-match; then resource/deployment constraints. This is a qualitative ordering, not a measured causal decomposition. The final failure taxonomy is provisional and cannot be converted into guaranteed gain from any one fix.

## 30. Research decision

**B — Freeze ML architecture and prepare a controlled portfolio demo.** The existing production path is a coherent demo and its engineering is demonstrable. The frozen release benchmark missed every major gate, while targeted BLIP/UForm/detector/sampling/fusion/QA/OCR/X-CLIP work mostly failed held-out quality or resource tests. Another immediate retrieval-model search on consumed holdouts has poor expected value. This is not a claim that future research is impossible or that a hosted deployment is already validated. Work that would be justified later is a *new*, independently human-reviewed source set with complete repeated-moment intervals and a specific user need; do not retune the spent final or temporal holdouts. Near-term action is documentation/portfolio preparation and real deployment rehearsal only if a controlled hosted demo is desired. No tag, release or deployment is performed by this assessment.

## 31. Final recommendation

**SCENEMIND CURRENT STATE:** Functional local-first multimodal moment-search release candidate; final release quality gate failed.

**BEST WORKING FEATURE:** Spoken Content search plus bounded ingestion and click-to-seek workflow.

**WEAKEST PRODUCTION FEATURE:** Visual retrieval for small, transient, compositional or repeated events; no-match is unsupported across modes.

**FINAL FROZEN DEPLOYMENT R@5:** 88/124 = **71.0% annotated-interval Top-5 recall**, not generic accuracy or verified user success.

**PUBLIC PRODUCTION READY:** NO.

**PORTFOLIO DEMO READY:** YES for a controlled local demonstration; hosted deployment checks remain.

**ML RESEARCH SHOULD CONTINUE:** NO, not as the immediate next milestone or on spent holdouts.

**BIGGEST TECHNICAL LESSON:** Retrieving plausible evidence and completing ingestion are distinct from finding the right timestamp or answering correctly; denser/heavier models do not automatically fix ranking.

**BIGGEST ENGINEERING ACHIEVEMENT:** A guarded, persistent, local CPU multimodal pipeline with real browser navigation and unusually explicit frozen evaluation/rejection discipline.

**BIGGEST LIMITATION:** Uneven retrieval quality across sources and modalities, compounded by ASR language failure and incomplete annotation confidence.

**RECOMMENDED NEXT ACTION:** Present a bounded local portfolio demo with the frozen results and failure examples. If hosting it, first verify container build/runtime, HTTPS/token/egress setup, restart and backup restore, resource capacity, and honest uncertainty wording; do not publish `v1.0.0` from this evidence.

## Assessment provenance

This document consolidates existing reports and source inspection. The only new execution for it was low-cost code validation and Git/release inspection. No ML model, frozen benchmark, evaluation label, production code, tag, release or deployment was changed. Historical metrics are linked at the point of use; measured values, operational estimates and editorial judgments are labeled separately.
