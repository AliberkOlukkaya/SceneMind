# Project status

Current milestone: core V1 (phases 0-6) implemented and validated locally.

Completed: repository foundation; bounded video ingestion and FFmpeg timestamps; local Whisper speech; pinned CLIP embeddings and FAISS search; BM25/RRF hybrid ranking; responsive library/player/transcript/search UI; reproducible interval-based evaluation. Professional setup, architecture, decisions and five learning guides are current.

Executed validation (2026-09-12): 20 pytest tests passed; Ruff passed; frontend ESLint, TypeScript and production build passed; pip check passed. Four desktop/mobile upload/search browser tests passed, including real cached CLIP inference and player.currentTime assertions. Two connection-recovery browser tests passed. Desktop/mobile screenshots visually reviewed. Real speech smoke passed again after the final ML dependency installation. No paid API, GPU instance or training was used.

Measured synthetic baseline: two positive color queries Recall@1/Precision@1/MRR@1=1.0; one negative query returned an irrelevant neighbor. Warm in-process median 15.154 ms; 10-second video ingestion 92.798 ms; indexing including cached-weight model load 6.719 s. See ml/evaluation/RESULTS.md. These are not real-video accuracy or network benchmarks.

Limitations: local single-process service only; no auth, durable queue, model-inference deadline or multi-worker coordination. CLIP cannot establish motion/causality; tiny speech accuracy is limited; no calibrated abstention. VFR metadata is approximate. PostgreSQL and CUDA are configurable but unverified. Two upstream test deprecation warnings remain.

No human blocker for core V1. Advanced phases 7-9 remain deliberately deferred: no demonstrated OCR/action/Q&A need or labeled specialization dataset yet. Production hardening is not complete; do not expose the local service publicly.

Next safe work: curate a licensed held-out real-video benchmark, measure negative-query calibration, add durable isolated workers and authentication before public deployment. Validate PostgreSQL and Linux packaging as deployment requirements become concrete. Git milestones exist locally; final remote synchronization follows repository checks.
