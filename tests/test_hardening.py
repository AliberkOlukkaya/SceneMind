import base64
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from app.config import settings
from app.database import engine, migrate
from app.jobs import claim, enqueue, finish, jobs
from app.main import app
from app.worker import child_loop, worker_lock
from ml.evaluation.calibrate import fit, summarize


@pytest.fixture
def durable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'jobs.db'}")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "durable_jobs", True)
    migrate()
    return TestClient(app)


def test_auth_guards_media_jobs_and_upload(durable, monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "test-only-password")
    assert durable.get("/health").status_code == 200
    for path in ("/videos", "/jobs", "/videos/bad/media", "/openapi.json"):
        assert durable.get(path).status_code == 401
    assert durable.post("/videos?filename=a.mp4", content=b"x").status_code == 401
    assert durable.get("/videos", headers={"Authorization": "Basic !!!"}).status_code == 401
    auth = base64.b64encode(b"scenemind:test-only-password").decode()
    assert durable.get("/videos", headers={"Authorization": f"Basic {auth}"}).status_code == 200
    assert (
        durable.get("/jobs", headers={"Authorization": "Bearer test-only-password"}).status_code
        == 200
    )


def test_atomic_duplicate_and_queue_limit(durable, monkeypatch):
    monkeypatch.setattr(settings, "max_pending_jobs", 1)

    def submit(i):
        try:
            return enqueue(str(i), "ingest")
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, range(4)))
    assert sum(isinstance(result, str) for result in results) == 1
    assert results.count(429) == 3
    job = claim()
    assert claim() is None
    finish(job)
    enqueue("new", "ingest")
    monkeypatch.setattr(settings, "max_pending_jobs", 20)
    with pytest.raises(HTTPException, match="already queued"):
        enqueue("new", "ingest")


def test_retry_persistence_and_explicit_retry(durable):
    job_id = enqueue("00000000-0000-0000-0000-000000000001", "ingest")
    job = claim()
    finish(job, "timeout")
    assert claim() is None  # Backoff is persisted.
    with engine().begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(available_at=0))
    second = claim()
    assert second["attempts"] == 2
    finish(second, "timeout")
    assert durable.get("/jobs").json()[0]["status"] == "failed"
    assert durable.post(f"/jobs/{job_id}/retry").status_code == 202
    assert durable.post(f"/jobs/{job_id}/retry").status_code == 409


def test_durable_upload_survives_api_restart_and_spawned_worker(durable, tmp_path):
    import subprocess

    import imageio_ffmpeg

    source = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:r=10:d=1",
            str(source),
        ],
        check=True,
    )
    response = durable.post("/videos?filename=fixture.mp4", content=source.read_bytes())
    video_id = response.json()["id"]
    with TestClient(app) as restarted:
        assert restarted.get(f"/videos/{video_id}").json()["job_status"] == "queued"
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=child_loop, args=(child, settings.model_dump()))
    process.start()
    child.close()
    try:
        job = claim()
        parent.send(job)
        assert parent.poll(30)
        assert parent.recv() is None
        finish(job)
        assert durable.get(f"/videos/{video_id}").json()["status"] == "ready"
        # Replay uses the same process and replaces frames safely.
        enqueue(video_id, "ingest")
        job = claim()
        parent.send(job)
        assert parent.poll(30)
        assert parent.recv() is None
        finish(job)
    finally:
        process.terminate()
        process.join(10)
        parent.close()
    with engine().connect() as conn:
        assert all(r.status == "ready" for r in conn.execute(select(jobs)))


def test_exclusive_worker_lock(durable):
    with worker_lock():
        with pytest.raises((OSError, RuntimeError)):
            with worker_lock():
                pass
    with worker_lock():
        pass


def test_inference_deadline_terminates_child(durable):
    from app.worker import await_result

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=child_loop, args=(child, settings.model_dump()))
    process.start()
    child.close()
    try:
        error, replacement = await_result(parent, process, 1)
        assert "deadline" in error
        assert replacement is None
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def test_cross_origin_write_rejected(durable):
    assert (
        durable.post(
            "/videos?filename=x.mp4", content=b"x", headers={"Origin": "https://untrusted.example"}
        ).status_code
        == 403
    )


def test_calibration_never_fits_heldout():
    positive = dict(
        split="calibration", mode="visual", intervals=[[0, 1]], scores=[0.5], timestamps=[0]
    )
    negative = dict(split="calibration", mode="visual", intervals=[], scores=[0.3], timestamps=[0])
    threshold = fit([positive, negative])
    assert threshold > 0.3
    assert summarize([positive, negative], threshold, 1)["negative_false_accept_rate"] == 0
    with pytest.raises(ValueError):
        fit([positive, dict(negative, split="heldout")])
    with pytest.raises(ValueError):
        fit([positive, dict(negative, scores=[float("nan")])])


def test_calibration_fails_closed(tmp_path, monkeypatch):
    from app.calibration import threshold

    artifact = tmp_path / "calibration.json"
    monkeypatch.setattr(settings, "calibration_path", artifact)
    with pytest.raises(HTTPException):
        threshold()

    artifact.write_text(
        json.dumps(
            dict(
                visual_model=settings.visual_model,
                revision=settings.visual_revision,
                mode="visual",
                sampling_interval=settings.sampling_interval,
                threshold=0.25,
            )
        )
    )
    assert threshold() == 0.25
    monkeypatch.setattr(settings, "visual_revision", "changed")
    with pytest.raises(HTTPException):
        threshold()


def test_split_leakage_and_regression_gate():
    from copy import deepcopy
    from pathlib import Path

    from ml.evaluation.calibrate import calibrate
    from ml.evaluation.regression import check

    root = Path(__file__).resolve().parents[1] / "ml/evaluation/reports"
    report = json.loads((root / "bunny-v1.json").read_text())
    baseline = calibrate(report)
    changed = deepcopy(baseline)
    changed["metrics"]["heldout"]["1"]["calibrated"]["negative_false_accept_rate"] = 1
    assert check(changed, baseline)
    assert check(baseline, baseline) == []
    leaked = deepcopy(report)
    leaked["videos"][1]["sha256"] = leaked["videos"][0]["sha256"]
    with pytest.raises(ValueError, match="Identical video"):
        calibrate(leaked)
