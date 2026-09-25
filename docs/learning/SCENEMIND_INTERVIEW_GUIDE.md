# SceneMind interview guide

**What problem does SceneMind solve?** It finds possible moments inside a video
from a natural-language query and lets a user jump to the returned timestamp.
The shipped product is an English-first, local-first search tool, not an answer
engine or a universal video-understanding system.

**What happens after upload or URL import?** FastAPI streams an upload or a
validated public URL into a UUID-owned local directory. OpenCV checks the video
stream, size, dimensions and duration. FFmpeg extracts a frame about every five
seconds and, on demand, mono 16 kHz audio. A durable SQL job worker can perform
processing with deadlines, retries and atomic status writes. The API and worker
share SQL and persistent media/model storage. A direct URL and an uploaded file
enter the same downstream pipeline; YouTube uses the pinned `yt-dlp` provider.

**What are Whisper, CLIP and embeddings here?** Whisper is a pretrained speech
recognizer. Its input is extracted audio; its output is timestamped transcript
segments. Production defaults to the CPU-friendly `tiny` model with int8
inference, so transcription errors are expected. CLIP is a pretrained dual
encoder: one side maps JPEG frames, the other query text, into comparable
512-dimensional vectors. SceneMind normalizes the vectors and compares cosine
similarity. It does not train either model from scratch. The five-second frame
interval limits CPU/storage cost, but it can miss short events and tiny objects.

**Why FAISS, BM25 and RRF?** FAISS `IndexFlatIP` performs exact inner-product
search over normalized frame vectors, equivalent to cosine ranking. BM25 ranks
transcript segments by lexical term evidence; exact names and phrases often
work, while paraphrases and ASR errors can fail. Smart Search combines their
ranked candidates by reciprocal-rank fusion with constant 60. RRF combines
ranks instead of incomparable raw CLIP cosine and BM25 scores. The shipped
fusion is uncapped, and cross-modal displacement remains a measured weakness.

**What do the three product modes do?** Smart Search sends `mode=hybrid`, Spoken
Content sends `mode=speech`, and Visual Content sends `mode=visual`. The older
AUTO classifier is API-compatible but removed from the normal UI because
English/Turkish source-disjoint evaluations did not justify it as a safe
default. Users choose explicit evidence when needed.

**How does URL ingestion avoid SSRF?** It accepts supported HTTP(S) destinations
and public YouTube forms, rejects credentials and non-public resolved IP
addresses, manually validates redirect destinations, bounds redirect count,
download bytes and time, and validates the downloaded media again. YouTube
provider availability can still change. This is single-operator protection,
not a substitute for isolation or external network policy in a public service.

**Why were object detection, OCR and Q&A rejected?** Small-object experiments
showed that sparse sampling often missed the evidence before detection, while
stronger CPU detectors increased resource cost. OCR detected text but missed
recognition/retrieval quality gates on frozen video sources. Grounded Q&A had
safe citations in some runs yet answer correctness or long-video evidence
completeness failed its gates. They remain research artifacts and are not in
the Find Moments production path. The README links detailed reports.

**How were results measured?** Freeze source-disjoint videos, source hashes,
queries, accepted timestamp intervals, difficulty labels, production config,
metric definitions and evaluator before search. A hit counts when a returned
timestamp enters an annotated interval. Recall@1/3/5 gives the fraction of
positive queries with a hit in the first 1/3/5 results. MRR@5 averages the
reciprocal of the first relevant rank, zero when absent. Source-disjointness
reduces leakage from tuning. Do not call Recall@5 generic “accuracy,” compare
metrics from different datasets as if they were one run, or tune on the final
holdout. The final acceptance protocol discloses caption/contact-sheet
annotation limits and lacks independent human sign-off.

**What limits deployment?** The worker retains local ML models in memory,
videos and indexes require persistent disk, and API and worker need a shared
database and volume. The current auth gate is for one operator. The app has no
tenant isolation, quotas, high availability or reliable global no-match
decision. Five-second sampling, ASR errors, text-free visual semantics and
multimodal fusion cause misses. Use an HTTPS proxy and exact CORS origins;
budget RAM and disk from measured source profiles, not a generic web-hosting
template. See [deployment](../DEPLOYMENT.md).

**What would you improve later?** Collect genuinely human-reviewed, diverse
feedback; improve sampling only on new development data; compare multilingual
ASR and better visual representations with fresh source-disjoint holdouts; add
operational controls before multi-user use. The v1.0 final acceptance set is
for one release decision, not for further tuning.
