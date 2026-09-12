# Decisions

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
