"""Durable single-host queue. One supervisor owns the data-directory OS lock."""

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, insert, select, update
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.database import engine

metadata = MetaData()
jobs = Table(
    "jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("video_id", String(36), nullable=False),
    Column("kind", String(20), nullable=False),
    Column("active_key", String(80), unique=True),
    Column("status", String(20), nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("available_at", Float, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("error", String(500)),
)
router = APIRouter(prefix="/jobs", tags=["jobs"])


def enqueue(video_id, kind):
    with engine().begin() as conn:
        if conn.dialect.name == "sqlite":
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        elif conn.dialect.name == "postgresql":
            conn.exec_driver_sql("SELECT pg_advisory_xact_lock(734219)")
        # Active-key uniqueness prevents duplicate jobs across API processes.
        if (
            len(conn.execute(select(jobs.c.id).where(jobs.c.active_key.is_not(None))).all())
            >= settings.max_pending_jobs
        ):
            raise HTTPException(429, "Job queue is full. Try again later.")
        job_id = str(uuid4())
        try:
            conn.execute(
                insert(jobs).values(
                    id=job_id,
                    video_id=video_id,
                    kind=kind,
                    active_key=f"{video_id}:{kind}",
                    status="queued",
                    attempts=0,
                    available_at=time.time(),
                    created_at=time.time(),
                )
            )
        except IntegrityError:
            raise HTTPException(409, "This job is already queued or running.") from None
    return job_id


def claim():
    with engine().begin() as conn:
        row = (
            conn.execute(
                select(jobs)
                .where(jobs.c.status == "queued", jobs.c.available_at <= time.time())
                .order_by(jobs.c.available_at)
                .limit(1)
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        changed = conn.execute(
            update(jobs)
            .where(jobs.c.id == row["id"], jobs.c.status == "queued")
            .values(status="running", attempts=row["attempts"] + 1)
        )
        return dict(row, attempts=row["attempts"] + 1) if changed.rowcount else None


def finish(job, error=None):
    retry = error is not None and job["attempts"] < settings.job_attempts
    with engine().begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.id == job["id"])
            .values(
                status="queued" if retry else "failed" if error else "ready",
                active_key=f"{job['video_id']}:{job['kind']}" if retry else None,
                available_at=time.time() + 2 ** job["attempts"],
                error=error,
            )
        )


@router.get("")
def list_jobs():
    with engine().connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(jobs).order_by(jobs.c.created_at.desc()).limit(100)
            ).mappings()
        ]


def pending(video_id, kind):
    if not settings.durable_jobs:
        return None
    with engine().connect() as conn:
        row = (
            conn.execute(
                select(jobs)
                .where(jobs.c.video_id == video_id, jobs.c.kind == kind)
                .order_by(jobs.c.created_at.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
        if row and row["status"] != "ready":
            return {
                "status": "processing" if row["status"] in {"queued", "running"} else "failed",
                "stage": {
                    "ingest": "preparing_video",
                    "speech": "transcribing",
                    "visual": "indexing",
                }.get(kind, kind),
                "job_status": row["status"],
                "job_id": row["id"],
                "error": row["error"],
            }
    return None


@router.post("/{job_id}/retry", status_code=202)
def retry_job(job_id: str):
    with engine().connect() as conn:
        job = conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "failed":
        raise HTTPException(409, "Only failed jobs can be retried")
    return {"job_id": enqueue(job["video_id"], job["kind"]), "status": "queued"}
