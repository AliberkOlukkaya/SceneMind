import json
import logging
from threading import Lock

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.config import settings
from app.encoder import encoder, rank_vectors
from app.video import folder_for, read_manifest

router = APIRouter(prefix="/videos", tags=["visual search"])
visual_lock = Lock()
logger = logging.getLogger(__name__)


def index_status(folder, include_jobs=True):
    from app.jobs import pending

    queued = pending(folder.name, "visual") if include_jobs else None
    if queued:
        return queued
    path = folder / "index.json"
    if not path.exists():
        return {"status": "not_started"}
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(folder, status, error=None):
    temporary = folder / "index.tmp"
    temporary.write_text(
        json.dumps(
            {
                "status": status,
                "error": error,
                "model": settings.visual_model,
                "revision": settings.visual_revision,
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(folder / "index.json")


def recover_visual():
    for path in settings.data_dir.glob("*/index.json"):
        if index_status(path.parent)["status"] == "processing":
            save_status(path.parent, "failed", "Indexing interrupted. Retry indexing.")


def build_index(video_id):
    folder = folder_for(video_id)
    temporary = folder / "embeddings.tmp"
    try:
        record = read_manifest(folder)
        paths = [
            folder / "frames" / frame["thumbnail"].rsplit("/", 1)[1] for frame in record["frames"]
        ]
        vectors = encoder().images(paths)
        if len(vectors) != len(paths):
            raise ValueError("Embedding/frame count mismatch")
        with temporary.open("wb") as output:
            np.save(output, vectors, allow_pickle=False)
        temporary.replace(folder / "embeddings.npy")
        save_status(folder, "ready")
    except Exception:
        logger.exception("Visual indexing failed for %s", video_id)
        save_status(folder, "failed", "Indexing failed. Check model availability and server logs.")
    finally:
        try:
            temporary.unlink(missing_ok=True)
            (folder / "index.tmp").unlink(missing_ok=True)
        finally:
            visual_lock.release()


@router.post("/{video_id}/index", status_code=202)
def start_index(video_id: str, background: BackgroundTasks):
    folder = folder_for(video_id)
    if read_manifest(folder)["status"] != "ready":
        raise HTTPException(409, "Wait for video processing to finish.")
    if settings.durable_jobs:
        from app.jobs import enqueue

        job_id = enqueue(folder.name, "visual")
        return {"status": "queued", "job_id": job_id}
    try:
        import faiss  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise HTTPException(
            503, 'Install the backend visual extra: pip install -e "backend[visual]"'
        )
    if not visual_lock.acquire(blocking=False):
        raise HTTPException(429, "Visual inference is busy. Try again shortly.")
    try:
        save_status(folder, "processing")
    except Exception:
        visual_lock.release()
        raise
    background.add_task(build_index, folder.name)
    return {"status": "processing"}


@router.get("/{video_id}/index")
def get_index(video_id: str):
    folder = folder_for(video_id)
    read_manifest(folder)
    return index_status(folder)


@router.get("/{video_id}/search/visual")
def visual_search(
    video_id: str,
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(default=10, ge=1, le=50),
):
    if not q.strip():
        raise HTTPException(422, "Enter a search query.")
    folder = folder_for(video_id)
    record = read_manifest(folder)
    state = index_status(folder)
    if state["status"] != "ready":
        raise HTTPException(409, "Build the visual index before searching.")
    if (state["model"], state["revision"]) != (settings.visual_model, settings.visual_revision):
        raise HTTPException(409, "Model changed. Rebuild the visual index.")
    if not visual_lock.acquire(blocking=False):
        raise HTTPException(429, "Visual inference is busy. Try again shortly.")
    try:
        vectors = np.load(folder / "embeddings.npy", allow_pickle=False)
        scores, indices = rank_vectors(vectors, encoder().text(q), k)
        from app.calibration import threshold

        cutoff = threshold()
        if cutoff is not None and record["sampling_interval"] != settings.sampling_interval:
            raise HTTPException(409, "Video sampling differs from calibration; reingest it.")
        return {
            "calibration_applied": cutoff is not None,
            "abstained": cutoff is not None and not any(s >= cutoff for s in scores),
            "query": q,
            "score_type": "cosine_similarity",
            "results": [
                {**record["frames"][index], "score": score, "modality": "visual"}
                for score, index in zip(scores, indices)
                if cutoff is None or score >= cutoff
            ],
        }
    finally:
        visual_lock.release()
