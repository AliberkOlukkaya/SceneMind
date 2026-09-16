# SceneMind from First Principles

This guide explains the project from the beginning so you can describe it clearly in an interview.

## 1. What problem does SceneMind solve?

A long video is difficult to search because its useful content lives inside images and audio, not
just in its filename. SceneMind turns a natural-language query into ranked timestamps. The user can
ask for “the slide with three routers” or “where the speaker explains DNS” and click a possible
matching moment.

## 2. What happens when a video is uploaded?

The browser streams bytes to FastAPI. The backend checks the extension, byte limit, free disk and
video stream, then stores the source under a generated UUID. FFmpeg extracts frames and OpenCV
supplies metadata. The video becomes ready only after a complete frame directory and atomic JSON
manifest exist. Speech transcription and visual indexing are separate, user-started stages.

## 3. Why sample video into frames?

A 30-minute, 25 fps video contains about 45,000 frames. Encoding all of them on CPU would be slow
and repetitive. SceneMind samples about one frame every five seconds, reducing that example to
roughly 360 frames. The tradeoff is that an event shorter than the interval may occur between
samples and never enter the visual index.

## 4. What does CLIP do?

CLIP is a pretrained neural network with an image encoder and a text encoder. It maps an image and
a visual-language query into the same kind of numeric vector. If their meanings are similar, their
vectors should point in similar directions. SceneMind uses the pinned CLIP ViT-B/32 checkpoint for
inference; it did not train CLIP.

## 5. What is an embedding?

An embedding is a list of numbers representing learned features. Imagine a tiny three-number
embedding where one direction roughly represents indoor/outdoor, another people/no people, and a
third diagram/photo. Real CLIP vectors have many dimensions whose meanings are learned rather than
hand-labeled. Similar content tends to occupy nearby directions.

## 6. Why does cosine similarity work?

Cosine similarity compares vector direction instead of length. SceneMind L2-normalizes vectors, so
their dot product equals cosine similarity. A value closer to 1 means the directions are more
aligned. It is a ranking signal, not a probability that the result is correct.

## 7. What does FAISS do?

FAISS stores frame embeddings and quickly finds vectors with the largest inner product to the
query vector. SceneMind uses exact `IndexFlatIP`, so it checks the complete local vector collection
rather than an approximate graph. At the measured scale this is simple, deterministic and fast.
FAISS is a search library, not a neural network.

## 8. What does Whisper do?

Whisper is a pretrained speech-recognition model. SceneMind extracts mono 16 kHz audio and asks
faster-whisper to emit text segments with start and end times. Those timestamps let a transcript
hit seek the video. The default tiny model favors CPU cost over maximum accuracy.

## 9. What does BM25 do?

BM25 ranks text documents using term matches, term rarity and document-length normalization. Here,
each transcript segment is a document. A rare term such as “PostgreSQL” contributes more than a
common term such as “the.” BM25 is a statistical retrieval formula, not deep learning.

## 10. Why are visual and speech retrieval separate?

What appears and what is said are different evidence channels. A silent product shot is visual;
an off-screen explanation is speech. CLIP cosine values and BM25 values also have unrelated scales,
so keeping the paths separate makes behavior easier to inspect and avoids invalid score addition.

## 11. What does RRF do?

Reciprocal Rank Fusion combines positions rather than raw scores. A result at rank `r` contributes
`1 / (60 + r)` in SceneMind. If the same thumbnail receives Visual and Speech evidence, the
contributions add. This rewards agreement without assuming CLIP and BM25 scores mean the same
thing. It can still over-reward weak incidental agreement, which evaluation found.

## 12. How does Smart Search work?

Smart Search directly runs the Hybrid path. It asks Visual and Speech retrieval for candidates,
groups evidence by the nearest sampled thumbnail, applies uncapped RRF60 and returns the highest
ranked moments. It does not use AUTO routing. Spoken Content and Visual Content bypass fusion and
query their respective paths explicitly.

## 13. How are timestamps produced?

FFmpeg's `showinfo` output provides the actual selected-frame times. Whisper provides segment start
and end times. Visual results seek to a frame timestamp; Speech results seek to the segment start
and show a nearby sampled thumbnail. The backend clips transcript ends to playable video duration.

## 14. Why do long videos need background jobs?

Frame extraction, transcription and embedding can take minutes. Keeping one HTTP request open is
fragile. Durable mode writes a SQL job, lets the upload return, and has a separate worker claim it.
Attempts, backoff and errors survive process restarts, and the UI polls real status.

## 15. Where do PyTorch and deep learning exist?

PyTorch/Transformers run the CLIP image and text encoders. faster-whisper uses CTranslate2 to run
Whisper. These are the two production deep-learning inference paths. Experimental models under
`ml/experiments` are not part of runtime unless explicitly promoted; none replaced the v1.0 core.

## 16. Which components are not ML?

Upload validation, FFmpeg extraction, OpenCV metadata, SQLAlchemy, Alembic, SQLite/PostgreSQL,
FAISS indexing, BM25, RRF, durable queues, atomic files, authentication, Next.js and click-to-seek
logic are conventional media, retrieval or application engineering.

## 17. What does inference mean?

Inference means feeding new input through fixed pretrained weights to obtain output. CLIP converts
a new frame or query into an embedding; Whisper converts new audio into text. SceneMind downloads
and runs existing weights locally.

## 18. Why is there no `backward()` or optimizer in production?

`backward()` computes gradients for training, and an optimizer changes model parameters. SceneMind
does neither in its production path. The goal is predictable local retrieval with pretrained
models. Small experimental logistic models were fitted offline, and rejected models never entered
runtime.

## 19. How does evaluation work?

A query has an expected evidence type and reviewed time interval. The evaluator runs the real
retrieval path, checks whether relevant evidence appears at ranks 1, 3 or 5, computes MRR, and
records negative behavior, latency and failures. Human acceptance asks whether a normal click would
actually be useful; mere interval overlap is not always enough.

## 20. Why does held-out data matter?

If the same examples choose a method and judge it, the result can reward memorization or accidental
fit. SceneMind selects on development data, freezes features and thresholds, then uses different
source videos for holdout. A failed holdout is not reused to tune another version of the candidate.

## 21. What would data leakage mean here?

Leakage includes using acceptance queries to choose weights, allowing frames from one source group
into both calibration and holdout, changing labels after seeing retrieval output, or repeatedly
tuning against the protected set. Checksum-bound manifests, source IDs and regression tests make
these mistakes detectable.

## 22. Major failed experiments and lessons

- **BLIP verifier:** a larger semantic model was slow and reduced recall.
- **UForm scorer:** acceptable latency did not prevent excessive false abstention.
- **Object detectors:** higher resolution helped visible objects, but sampling often removed the
  evidence before detection.
- **Denser sampling:** evidence coverage improved, yet CLIP Top-5 did not improve proportionally.
- **AUTO routers:** authored validation overstated performance; human-grounded transfer was weak.
- **Capped RRF:** Top-1 improved while Top-5 and Speech regressed on frozen holdout.
- **Raw-score calibration:** calibration-source gains failed on new sources.

The lesson is to diagnose the entire retrieval chain and promote only source-disjoint gains.

## 23. Current limitations

The product is English-first. Five-second frames miss brief events. Whisper tiny makes errors and
BM25 is lexical. Hybrid ranking can displace strong one-channel results. No-match detection is
unreliable. OCR, action reasoning, identity recognition and Video RAG are absent. CPU processing
can take minutes, and the deployment is single-operator rather than multi-tenant.

## 24. How would you improve SceneMind next?

First define a concrete user failure and collect new, source-disjoint evidence. For ranking, gather
at least three new calibration sources with complete Visual and Speech candidate traces, then freeze
a separate validation split. For brief visual events, evaluate bounded sampling only with a full
candidate-to-final-ranking metric. Any OCR, action or RAG work should begin with a user scenario,
licensed dataset, baseline, resource budget and promotion gate. The release core should stay fixed
until such evidence exists.
