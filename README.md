# SceneMind

Search inside video using natural language.

SceneMind is a local-first video intelligence project built with FastAPI, Next.js, TypeScript and Tailwind. No paid AI API is required.

## Implemented
Repository foundation and backend health endpoint. Video ingestion and AI retrieval are planned; see [project status](PROJECT_STATUS.md) for verified progress.

## Development
Requires Python 3.11–3.13 and Node.js 20.9+.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e "backend[dev]"
.venv\Scripts\python -m uvicorn app.main:app --reload
```

In another terminal:
```powershell
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. API documentation: http://localhost:8000/docs. Optional configuration: copy .env.example to .env at the repository root.

## Checks
```powershell
.venv\Scripts\python -m pytest -c backend/pyproject.toml
.venv\Scripts\python -m ruff check backend tests
cd frontend
npm run lint
npx tsc --noEmit
npm run build
```

See [architecture](ARCHITECTURE.md), [roadmap](PROJECT_PLAN.md), and [decisions](DECISIONS.md). Never commit uploaded media, model weights or secrets. This is not yet a production deployment.

## Video workflow
Upload a supported video in the library, wait for processing, select it and click a thumbnail to seek. Limits: 250 MiB, 30 minutes, 4K. MP4/H.264 is recommended for browser playback; other accepted containers depend on browser codec support. Extraction runs locally with a packaged FFmpeg binary. Configure IMAGEIO_FFMPEG_EXE to use a system binary.

Implemented: upload, metadata, frames, thumbnails, processing states and video workspace. Speech and semantic search are not implemented yet. Run one backend worker; this local app has no authentication and should not be exposed publicly.

For reproducible Python dependencies, install `-r backend/requirements.lock` before `-e "backend[dev]"`. The lock records this Windows/Python 3.13 environment. Run `./scripts/check.ps1` for all checks; it uses isolated workspace test directories to avoid Windows temporary-directory permission issues.

## Local speech
Install `.venv\Scripts\python -m pip install -e "backend[speech]"`, restart the backend and select **Transcribe video** in a video workspace. First use downloads Whisper tiny into data/models. Transcripts can be searched by literal text and selected to seek. No API key is needed. See docs/learning/02-speech-transcription.md for model parameters and limitations.

SQLite transcript tables are migrated automatically at backend startup. SCENEMIND_DATABASE_URL changes the database location. Current core metadata remains in per-video manifests. The dependency lock currently includes the optional speech stack installed for verification.

A real model smoke check is available as `python scripts/smoke_speech.py path/to/short-reference.wav`; supply a short recording containing 'learning rate'. This downloads the model if uncached. Unit tests mock inference and do not download weights.
