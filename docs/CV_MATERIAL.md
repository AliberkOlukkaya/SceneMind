# SceneMind CV Material

## Project

**SceneMind — Multimodal Video Search Engine**

Built a local-first web application that indexes video frames and speech, retrieves timestamped
moments from natural-language queries, and supports click-to-seek exploration through a Next.js
interface and FastAPI backend. Final deployment acceptance chose Decision C;
do not describe this as a deployed v1.0.0 product.

### CV bullets

- Engineered a multimodal retrieval pipeline with pretrained CLIP and Whisper inference, exact
  FAISS vector search, BM25 transcript retrieval and reciprocal-rank fusion across visual and
  speech evidence.
- Built streamed 1 GiB/60-minute ingestion and durable SQL-backed workers with bounded retries,
  stage timeouts, atomic outputs and failure cleanup; a measured 45-minute CPU run completed in
  245 seconds with 540 frames, 373 transcript segments and zero temporary residue.
- Built an eight-video, 124-query checksum-frozen deployment diagnostic and reported
  **71.0% interval Recall@5** (45.2% R@1) with explicit annotation limitations;
  withheld release after a 40-minute source reached only 50.0% R@5 and two other
  sources fell below the predeclared catastrophic threshold.

The last bullet describes a **frozen interval score**, not verified user success
or generic “71% accuracy.” Contact-sheet/caption annotations had no independent
human sign-off and under-covered repeated visual scenes; disclose this in an
interview. The project demonstrates evidence-based release restraint.

## Tech stack line

Python, FastAPI, Next.js, React, TypeScript, PyTorch, Transformers/CLIP, faster-whisper, FAISS,
BM25, SQLAlchemy, Alembic, SQLite/PostgreSQL, FFmpeg, OpenCV, pytest, Playwright, Docker

## One-line GitHub description

Local-first multimodal video search with CLIP, Whisper, FAISS, BM25 and timestamped click-to-seek results.

## One-line portfolio description

An evidence-driven video search system that combines visual and speech retrieval, durable local processing and frozen held-out evaluation.

## LinkedIn description

Built SceneMind, a local-first video search prototype with FastAPI, Next.js,
Whisper, CLIP, BM25, FAISS and RRF60. It indexes speech and sampled frames,
returns timestamped candidates and supports click-to-seek. I froze an
eight-video/124-query deployment diagnostic before one run, reported
45.2/66.1/71.0% interval Recall@1/3/5, and withheld v1.0.0 after long-video
ASR and visual reliability failed release gates. Those numbers are not an
independently human-validated user-success claim.

## Technical interview summary

Explain streamed upload/secure URL acquisition, five-second FFmpeg sampling,
Whisper-tiny transcript segments, CLIP image/text vectors, exact FAISS visual
search, BM25 lexical speech search and uncapped RRF60 Smart Search. Describe
durable SQL jobs, atomic indexes, persistent storage and the 40-minute CPU
run. State why AUTO, OCR and Q&A remain research. The most instructive final
failure was English lecture audio classified as Welsh; the benchmark also
revealed under-annotated repeated visual moments. Neither the retrieval path
nor frozen labels were modified afterward.
