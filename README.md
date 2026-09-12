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
