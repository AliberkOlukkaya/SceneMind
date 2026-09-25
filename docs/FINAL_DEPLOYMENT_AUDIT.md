# Final deployment and security audit (2026-09-25)

**Result: not ready for deployment (Decision C).** The production Find Moments
path was frozen; this audit changed only the container dependency declaration,
Compose CORS/device settings, evaluation artifacts and documentation. It did
not enable Ask Video, OCR, AUTO routing or alternative retrieval.

| Area | Verified state | Limit / action |
| --- | --- | --- |
| API/frontend origin | Frontend reads build-time `NEXT_PUBLIC_API_URL`; FastAPI has explicit `SCENEMIND_CORS_ORIGINS`; Compose now forwards it. | Set exact HTTPS origins and rebuild frontend for deployed API URL. |
| Host/auth | Compose binds API to `127.0.0.1:8000` and requires an operator token; middleware supports Basic/Bearer and rejects disallowed write origins. | HTTPS proxy, strong token and network restriction are required; no tenants, quotas or public identity management. |
| Database/jobs | PostgreSQL Compose volume, Alembic migrations and durable SQL worker are implemented; failed stages have bounded retry/cleanup. | API and worker need the same database and one shared persistent media/model volume; test restart and restore in a real deployment. |
| FFmpeg/models | `imageio-ffmpeg` bundles the executable; Dockerfile now installs `speech` and `visual` extras. CLIP revision is pinned; model cache is on persistent `/app/data`. | Docker daemon was unavailable: image build/model startup remain unverified. First-use downloads require network or a prepared cache. |
| Upload/storage | Streamed 1 GiB and 60-minute defaults, 4K bound, disk reserve and processing headroom; UUID-owned media and atomic frame/index writes. | Capacity must include source copies, frames, indexes, cache and temporary work. |
| Direct URL/YouTube | Submitted URLs are HTTP(S), credential-free, public-IP validated; direct redirects are rechecked; byte/time bounds and local reinspection apply. The live public YouTube recheck acquired 634.57 s of media, indexed 127 frames, searched and served HTTP 206 media. | Provider behavior may change; egress policy and content permissions are deployment responsibilities. |
| Temporary data/errors | Failure paths remove partial downloads, staged frames/audio and temporary indexes; API messages avoid internal command output/paths. | Operator logs are local and should be access-controlled and rotated. |
| Health/persistence | `/health` responds; media/indexes and SQL jobs survive application restart by design and prior tests. | The endpoint is API liveness only; independently monitor worker/queue and perform a deployed restart/backup restore check. |
| Secrets/Git | `.env.local` and `data/` are ignored. A value-presence check found the local `OPENAI_API_KEY` absent from all 430 tracked files; no tracked file exceeded 10 MiB, and no runtime data file was tracked. | Keep deployment secrets outside Git and never inject an API key into the frontend. Production Find Moments requires no OpenAI key. |

`docker compose config --quiet` passed with placeholder private environment
values. The Docker engine pipe was unavailable, so **no Docker image build or
running-container check passed**. Local checks passed: 282 backend pytest cases
(one skipped), including the new frozen-manifest guard, plus Ruff, ESLint,
TypeScript, Next.js production build and 18 desktop/mobile
Playwright tests. The Playwright URL UI test mocks a remote response; the
separate live YouTube acquisition/search/HTTP Range check was a real API path.
The real YouTube artifact was also opened in the actual Next.js browser UI:
desktop and mobile Visual Search both sought the video to 490 seconds from
the first result, with no horizontal overflow. This is a separate flow check,
not a repeat of final benchmark queries.
See [deployment guide](DEPLOYMENT.md) and
[frozen accuracy report](../ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_RESULTS.md).

Release blockers are the frozen retrieval failures (especially Welsh ASR on an
English 40-minute lecture), incomplete repeated-scene annotations without
independent human sign-off, unavailable Docker image validation and absent
deployed HTTPS/browser/restart/backup proof. Do not create `v1.0.0` from this
state.
