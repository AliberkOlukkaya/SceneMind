# SceneMind v1.0.0-rc1

SceneMind v1.0.0-rc1 is a portfolio release candidate for local multimodal video search. It
packages the validated product baseline and freezes search-core experimentation for v1.0.

## Highlights

- Upload local video and navigate timestamped search results with one click.
- Smart Search combines visual and spoken evidence; explicit Spoken and Visual modes remain available.
- Conservative result language avoids presenting ranked candidates as confirmed answers.
- English-first, paid-API-free inference with locally cached pretrained models.

## Architecture

FastAPI streams media into local UUID-owned storage. FFmpeg/OpenCV extract five-second frames;
CLIP and FAISS provide visual retrieval. Whisper, SQL transcript segments and BM25 provide speech
retrieval. Uncapped RRF60 combines ranks for Smart Search. Next.js supplies the workspace.

## Engineering hardening

- Configurable 1 GiB / 60-minute upload envelope and disk headroom checks.
- Optional durable SQL jobs, bounded retry, stage deadlines and failed-job retry.
- Atomic media/index outputs, transactional transcripts and partial-artifact cleanup.
- SQLite local path, PostgreSQL-compatible migrations, optional operator authentication.
- pytest, Ruff, ESLint, TypeScript, production build, Playwright and frozen regressions.

## Evaluation philosophy

SceneMind records failures rather than hiding them. Development and held-out sources are separated,
manifests and gates are frozen, and a candidate is promoted only when quality and resources pass.
BLIP, UForm, detector, sampling, router and calibrated-fusion branches remain documented rejections.

## Known limitations

- English-first; multilingual reliability is not claimed.
- No reliable global no-match detection.
- Five-second sampling can miss short events and small objects.
- RRF60 has known cross-modal displacement failures.
- No general OCR, action understanding, identity recognition or Video RAG.
- Long-video CPU processing can take minutes and use several gigabytes of RAM.
- This is not a public multi-tenant or highly available deployment.
