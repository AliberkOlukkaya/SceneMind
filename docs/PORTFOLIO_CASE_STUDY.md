# SceneMind — Portfolio Case Study

## 1. Problem

Finding one moment in a long lecture, demo or ordinary video usually means scrubbing the timeline
or replaying large sections. Filenames and video-level metadata cannot answer questions such as
“where does the speaker explain DNS?” or “show the slide with three network devices.”

## 2. Product

SceneMind is a local-first web application for natural-language timestamp search. A user uploads a
video, builds local speech and visual indexes, chooses Smart Search, Spoken Content or Visual
Content, and receives ranked moments. Clicking a result seeks the same video to that timestamp.

The interface says “Most relevant moments” and “Possible matches” because retrieval produces
candidates, not verified answers. The v1.0 release candidate is English-first and intended for
local engineering review. Final deployment acceptance chose Decision C, so no v1.0.0 release or
hosted service is claimed.

## 3. Architecture

The FastAPI backend streams uploads into UUID-owned directories. FFmpeg extracts timestamped JPEGs
every five seconds; OpenCV validates the video and supplies metadata. CLIP embeds images and visual
queries, and FAISS performs exact cosine-equivalent search over normalized vectors. Whisper emits
timestamped transcript segments, which BM25 searches lexically. Smart Search combines Visual and
Speech ranks with uncapped reciprocal-rank fusion using constant 60.

SQLite is the simple local database. SQLAlchemy and Alembic also support PostgreSQL. The Next.js
workspace shows real pipeline states, plays local media, renders candidate frames and seeks on a
result click. An optional durable worker persists jobs, bounds retries and timeouts, reuses loaded
models, and removes partial artifacts after failure.

## 4. My engineering decisions

- **CLIP:** A pretrained dual encoder provides an understandable zero-shot visual baseline without
  collecting training data. Its shared image/text embedding space makes natural-language frame
  retrieval possible.
- **Whisper tiny:** Local timestamped ASR adds speech evidence with a CPU-friendly starting model.
  The model is configurable because accuracy and cost vary by deployment.
- **FAISS `IndexFlatIP`:** Exact search keeps the baseline deterministic and is fast enough for the
  measured local collections. Approximate indexing would add complexity without a demonstrated
  need.
- **BM25:** Transcript search often benefits from exact terms, quoted phrases and identifiers.
  BM25 is cheap, interpretable and does not require a second text embedding model.
- **RRF60:** CLIP cosine values and BM25 scores are not directly comparable. Rank fusion combines
  them without pretending that they share a calibrated scale.
- **Durable workers:** Long videos outlive browser requests. Persisted jobs, stage deadlines,
  bounded retries, atomic outputs and cleanup make processing recoverable and inspectable.
- **Local-first storage:** Media, frames, indexes, databases and model caches stay outside Git and
  can remain on the operator's machine.

## 5. Evaluation strategy

Evaluation evolved with the product. Synthetic fixtures first verified mechanics. Natural-video
manifests then bound queries to reviewed time intervals. Later studies separated calibration,
development, validation and protected holdout sources by checksum. Queries and gates were frozen
before retrieval where the protocol required it.

Reports record Top-1/3/5, MRR, negative behavior, latency, memory and query-level failures. Some
earlier studies used human-click usefulness; the final deployment diagnostic uses frozen timestamp
intervals, and incomplete annotation of repeated scenes limits its user-success interpretation.
Regression tests preserve
frozen manifests and production output parity. Experimental code stays under `ml/experiments`
until independent evidence supports promotion.

The 30:29 Final English Acceptance V2 video reached useful Top-1/3/5 of
76.9%/76.9%/84.6%. That was useful but failed the predeclared 85% Top-3 and 90% Top-5 gates, so the
result is reported as a failed acceptance rather than rounded into a success claim.

The eight-video, 124-positive final deployment diagnostic was frozen at
`d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196` before one
real-model run. Frozen interval Recall@1/3/5 was **45.2/66.1/71.0%**, MRR@5 **55.4%**.
Speech/Visual/Smart R@5 was **81.8/63.6/68.0%**. A 40.38-minute lecture reached only
50.0% R@5 after Whisper tiny classified English as Welsh. Two more sources were below the
predeclared catastrophic threshold of 60%. The source-video contact sheets and publisher
captions were reviewed independently of SceneMind outputs, but there was no independent human
annotator and some repeated visual moments lacked alternate intervals. No labels were changed
after observation. This is an **interval diagnostic**, not a defensible claim of 71% verified
user success. [Full result](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md).

## 6. What failed

Several plausible improvements did not transfer:

- A BLIP image-text verifier added 3.352 seconds median CPU latency and reduced held-out recall.
- UForm3-small met the latency target but rejected 47.6% of held-out positives.
- Higher-resolution YOLOX improved detector recall when objects were visible, while the production
  five-second sampler missed three of four reviewed events. A stronger RT-DETR added little recall
  for much higher CPU cost.
- Two-second sampling retained all four reviewed events, yet fixed CLIP Top-5 found only one. More
  frames did not automatically produce better final ranking.
- A character n-gram AUTO router looked strong on authored scenario queries, then reached only 60%
  on a human-grounded frozen test. AUTO was removed from the normal interface.
- A capped RRF variant improved Top-1 on development and holdout but regressed aggregate and Speech
  Top-5. It was not promoted.
- Raw-score calibration looked useful inside calibration sources but lost both AUC and Brier to
  rank-only references on new sources. The experiment stopped before constructing fusion.
- Grounded Q&A, semantic/hierarchical retrieval and OCR each failed their separate frozen
  utility or quality gates. They remain disabled and are not included in production Find Moments.

The recurring lesson was that a larger model or denser candidate pool does not repair sampling,
ranking and dataset mismatch automatically.

## 7. Current limitations

SceneMind is English-first, has no reliable no-match detector, and can miss short visual events
between five-second samples. Whisper tiny can mistranscribe and BM25 can miss paraphrases. RRF60
can over-reward incidental cross-modal agreement. General OCR, action understanding, identity
recognition and Video RAG are outside the release. CPU processing takes minutes for long videos;
public multi-user deployment and high availability are not implemented.

The final long lecture took 616.7 seconds for upload/ingest, local ASR and visual indexing,
reached 3,338 MiB peak single-process RSS and used 164 MiB persistent source/index space.
A live public YouTube URL was acquired, indexed and searched again during release checks.
The Docker dependency list was corrected, but the local Docker daemon was unavailable for an
image build. [Deployment layout and measured resource context](DEPLOYMENT.md).

## 8. What I learned

This project made the boundary between model inference and product quality concrete. Retrieval
depends on preprocessing, sampling, labels, candidate recall, ranking and interaction design as
much as it depends on model choice. I learned to protect held-out evidence, diagnose leakage,
measure latency and memory alongside accuracy, and reject improvements that fail transfer. I also
learned that durable ML systems need ordinary software engineering: atomic writes, migrations,
timeouts, retries, path validation, cleanup and honest error states.

## 9. Interview talking points

### Why CLIP?

It offers a readable pretrained image/text baseline for zero-shot frame retrieval. SceneMind uses
the model for inference and does not claim to have trained it.

### Why Whisper?

Visual frames cannot recover spoken explanations. Whisper supplies local, timestamped transcript
evidence; the tiny variant is a practical CPU baseline.

### Why FAISS?

FAISS provides efficient exact inner-product search over normalized embeddings. Exact search is
simple and deterministic at SceneMind's measured scale.

### Why BM25 as well as embeddings?

Transcript queries often contain exact phrases, names or technical terms. BM25 handles that case
cheaply and transparently, while CLIP addresses visual semantics.

### Why RRF?

CLIP and BM25 raw scores have different meanings and scales. RRF combines their ranks without
invalid raw-score addition. A later calibration experiment failed source-disjoint transfer.

### Why not train a model from scratch?

There was no task-specific dataset large enough to justify training. Pretrained models plus frozen
evaluation delivered a stronger, cheaper engineering baseline.

### Why was AUTO removed?

It reached 60% on the human-grounded frozen test and had weak Visual and Speech recall. The UI now
defaults directly to Hybrid and offers explicit Speech and Visual modes.

### What was the hardest failure?

Separating candidate-generation failure from ranking failure. Small objects could be absent from
sampled frames, present but missed by CLIP, or retrieved by one modality and displaced by fusion.

### How did you avoid benchmark overfitting?

I used source-level splits, checksum-bound manifests, frozen gates, one-shot protected holdouts and
query-level regression tests. Failed holdouts were not converted into tuning data.

### What would you improve next?

I would collect new source-disjoint evidence before changing ranking. Candidate work would target
sampling and confidence only after a new validation protocol exists.

### How does long-video processing work?

Uploads stream to disk, durable SQL jobs run each stage, a reusable child caches models, stage
deadlines terminate stuck process trees, and atomic promotion prevents partial output from being
treated as ready.

### Where does deep learning occur?

CLIP encodes frames and visual queries; Whisper transcribes audio. FAISS, BM25, RRF, queues,
storage, validation and the UI are conventional software or retrieval algorithms.
