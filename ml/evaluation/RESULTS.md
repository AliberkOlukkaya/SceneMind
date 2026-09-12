# Measured synthetic baseline — 2026-09-12

This is a pipeline regression baseline, not a real-video quality benchmark. Dataset: one generated 10-second 640×360 video, red for 0–5 seconds and blue for 5–10 seconds. Two positive color queries and one deliberately absent snowy-mountain query. Labels are independent of model output.

Command: `python -m ml.evaluation.run --synthetic --k 1 --repeats 5`.

Environment: Windows 11 build 26200, Python 3.13.5, CPU inference with four Torch threads, PyTorch 2.14.0+cpu, Transformers 4.57.6, FAISS 1.15.0. CLIP ViT-B/32 revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. Five-second sampling, two frames. Weights were already downloaded; the model was loaded fresh in this process.

| Measurement | Observed value |
| --- | ---: |
| Positive-query Recall@1 | 1.0 (2 queries) |
| Positive-query Precision@1 | 1.0 (2 queries) |
| Positive-query MRR@1 | 1.0 (2 queries) |
| Negative queries returning a result | 1 of 1 |
| Median of per-query warm medians | 15.154 ms |
| Upload + metadata + frame extraction | 92.798 ms |
| Indexing including model load | 6.719 s |
| Persisted embedding file | 4,224 bytes |
| Video directory total | 11,868 bytes |

Timing uses FastAPI TestClient in-process, not a network round trip. Each query has one initial request followed by five warm requests. Model cache size, Python/native process memory, database bytes and test runner overhead are not included in the storage total. Peak memory was not measured. Hardware-specific latency is not a service-level promise.

The negative query failure is expected because current top-K retrieval has no abstention. The sample is too small and artificial to support claims about objects, actions, speech quality or general video understanding. A separately labeled licensed corpus, held-out queries and broader hardware measurements remain future work.
