from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import AccessMiddleware
from app.config import settings
from app.database import migrate
from app.hybrid import router as hybrid_router
from app.jobs import router as jobs_router
from app.speech import recover_speech
from app.speech import router as speech_router
from app.video import recover_interrupted, router
from app.visual import recover_visual
from app.visual import router as visual_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate()
    if not settings.durable_jobs:
        recover_interrupted()
        recover_speech()
        recover_visual()
    yield


app = FastAPI(title="SceneMind", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(jobs_router)
app.add_middleware(AccessMiddleware)
app.include_router(speech_router)
app.include_router(visual_router)
app.include_router(hybrid_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "scenemind"}
