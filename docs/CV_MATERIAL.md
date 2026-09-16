# SceneMind CV Material

## Project

**SceneMind — Multimodal Video Search Engine**

Built a local-first web application that indexes video frames and speech, retrieves timestamped
moments from natural-language queries, and supports click-to-seek exploration through a Next.js
interface and FastAPI backend.

### CV bullets

- Engineered a multimodal retrieval pipeline with pretrained CLIP and Whisper inference, exact
  FAISS vector search, BM25 transcript retrieval and reciprocal-rank fusion across visual and
  speech evidence.
- Built streamed 1 GiB/60-minute ingestion and durable SQL-backed workers with bounded retries,
  stage timeouts, atomic outputs and failure cleanup; a measured 45-minute CPU run completed in
  245 seconds with 540 frames, 373 transcript segments and zero temporary residue.
- Designed source-disjoint, checksum-frozen evaluation and regression tooling; rejected six
  plausible model/ranking candidates when held-out recall, calibration, latency or memory gates
  failed, preserving the safer RRF60 production baseline.

## Tech stack line

Python, FastAPI, Next.js, React, TypeScript, PyTorch, Transformers/CLIP, faster-whisper, FAISS,
BM25, SQLAlchemy, Alembic, SQLite/PostgreSQL, FFmpeg, OpenCV, pytest, Playwright, Docker

## One-line GitHub description

Local-first multimodal video search with CLIP, Whisper, FAISS, BM25 and timestamped click-to-seek results.

## One-line portfolio description

An evidence-driven video search system that combines visual and speech retrieval, durable local processing and frozen held-out evaluation.
