"""Opt-in real direct-URL versus upload equivalence run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.database import engine_for
from app.jobs import claim, enqueue, finish
from app.main import app
from app.video import read_manifest
from app.worker import execute

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/natural-v2/source/find-link-talk.webm"
SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/0/0e/Find_link_-_lightning_talk.webm"
REPORT = ROOT / "ml/evaluation/reports/url-ingestion-v1.json"
QUERIES = {
    "visual": [
        "a presentation slide with a browser interface",
        "a speaker standing beside the projected screen",
        "a web page shown on the screen",
    ],
    "speech": ["Kansas", "airport", "languages"],
    "hybrid": [
        "the speaker discusses Kansas while a slide is visible",
        "the airport example on the presentation screen",
        "languages mentioned during the browser demonstration",
    ],
}


def _finish_next(expected_kind: str) -> tuple[str, float]:
    job = claim()
    if job is None or job["kind"] != expected_kind:
        raise RuntimeError(f"expected {expected_kind} job, got {job}")
    started = time.perf_counter()
    result = execute(job)
    elapsed = time.perf_counter() - started
    if isinstance(result, dict):
        finish(job, result.get("error"), retryable=result.get("retryable", True))
        error = result.get("error")
        if not error and result.get("next_kind"):
            enqueue(job["video_id"], result["next_kind"])
    else:
        finish(job, result)
        error = result
    if error:
        raise RuntimeError(f"{expected_kind} failed: {error}")
    return job["video_id"], elapsed


def _normalize_transcript(segments: list[dict]) -> str:
    return " ".join(re.findall(r"\w+", " ".join(row["text"] for row in segments).casefold()))


def _search(client: TestClient, video_id: str) -> dict[str, list[dict]]:
    output = {}
    for mode, queries in QUERIES.items():
        for index, query in enumerate(queries, 1):
            response = client.get(
                f"/videos/{video_id}/search",
                params={"q": query, "k": 5, "mode": mode},
            )
            response.raise_for_status()
            output[f"{mode}-{index}"] = response.json()["results"]
    return output


def run() -> dict:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    run_root = ROOT / "data/url-ingestion-v1" / str(uuid4())
    run_root.mkdir(parents=True)
    settings.data_dir = run_root / "videos"
    settings.database_url = f"sqlite:///{run_root / 'scenemind.db'}"
    settings.durable_jobs = True
    settings.min_free_disk_bytes = 0
    engine_for.cache_clear()

    with TestClient(app) as client:
        local_submit = client.post(
            "/videos?filename=find-link-talk.webm",
            content=SOURCE.read_bytes(),
        )
        local_submit.raise_for_status()
        local_id = local_submit.json()["id"]
        imported_submit = client.post("/videos/import-url", json={"url": SOURCE_URL})
        imported_submit.raise_for_status()
        imported_id = imported_submit.json()["id"]

        processed_id, local_ingest = _finish_next("ingest")
        assert processed_id == local_id
        acquired_id, acquisition = _finish_next("acquire")
        assert acquired_id == imported_id
        processed_id, imported_ingest = _finish_next("ingest")
        assert processed_id == imported_id

        stage_times: dict[str, dict[str, float]] = {
            "upload": {"ingest": local_ingest},
            "url": {"acquire": acquisition, "ingest": imported_ingest},
        }
        for label, video_id in (("upload", local_id), ("url", imported_id)):
            client.post(f"/videos/{video_id}/transcript").raise_for_status()
            claimed_id, speech_time = _finish_next("speech")
            assert claimed_id == video_id
            client.post(f"/videos/{video_id}/index").raise_for_status()
            claimed_id, visual_time = _finish_next("visual")
            assert claimed_id == video_id
            stage_times[label].update(speech=speech_time, visual=visual_time)

        local_record = read_manifest(settings.data_dir / local_id)
        imported_record = read_manifest(settings.data_dir / imported_id)
        local_transcript = client.get(f"/videos/{local_id}/transcript").json()["segments"]
        imported_transcript = client.get(f"/videos/{imported_id}/transcript").json()["segments"]
        local_vectors = np.load(settings.data_dir / local_id / "embeddings.npy")
        imported_vectors = np.load(settings.data_dir / imported_id / "embeddings.npy")
        local_results = _search(client, local_id)
        imported_results = _search(client, imported_id)

    transcript_similarity = SequenceMatcher(
        None,
        _normalize_transcript(local_transcript),
        _normalize_transcript(imported_transcript),
    ).ratio()
    frame_deltas = [
        abs(first["timestamp"] - second["timestamp"])
        for first, second in zip(local_record["frames"], imported_record["frames"])
    ]
    comparisons = []
    for query_id in local_results:
        local_timestamps = [row["timestamp"] for row in local_results[query_id]]
        imported_timestamps = [row["timestamp"] for row in imported_results[query_id]]
        paired = [abs(a - b) for a, b in zip(local_timestamps, imported_timestamps)]
        passed = bool(local_timestamps and imported_timestamps) and (max(paired, default=0) <= 5)
        comparisons.append(
            {
                "query_id": query_id,
                "query": QUERIES[query_id.rsplit("-", 1)[0]][int(query_id.rsplit("-", 1)[1]) - 1],
                "local_timestamps": local_timestamps,
                "url_timestamps": imported_timestamps,
                "max_paired_delta_seconds": max(paired, default=None),
                "passed": passed,
            }
        )
    partials = [
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.name.endswith((".part", ".tmp", ".ytdl", "audio.wav"))
    ]
    structural = {
        "duration_delta_seconds": abs(
            local_record["metadata"]["duration"] - imported_record["metadata"]["duration"]
        ),
        "frame_counts": [len(local_record["frames"]), len(imported_record["frames"])],
        "max_frame_timestamp_delta_seconds": max(frame_deltas, default=0),
        "transcript_segment_counts": [len(local_transcript), len(imported_transcript)],
        "transcript_similarity": transcript_similarity,
        "visual_index_shapes": [list(local_vectors.shape), list(imported_vectors.shape)],
        "media_sha256_equal": hashlib.sha256(
            (settings.data_dir / local_id / local_record["source"]).read_bytes()
        ).hexdigest()
        == hashlib.sha256(
            (settings.data_dir / imported_id / imported_record["source"]).read_bytes()
        ).hexdigest(),
        "partial_artifacts": partials,
    }
    structural_passed = (
        structural["duration_delta_seconds"] <= 0.25
        and structural["frame_counts"][0] == structural["frame_counts"][1]
        and structural["max_frame_timestamp_delta_seconds"] <= 0.05
        and abs(
            structural["transcript_segment_counts"][0] - structural["transcript_segment_counts"][1]
        )
        <= 1
        and transcript_similarity >= 0.95
        and structural["visual_index_shapes"][0] == structural["visual_index_shapes"][1]
        and not partials
    )
    report = {
        "schema_version": "1.0.0",
        "experiment": "url-ingestion-v1",
        "source": {
            "provider": "direct",
            "url": SOURCE_URL,
            "license": "CC BY-SA 4.0",
            "local_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        },
        "video_ids": {"upload": local_id, "url": imported_id},
        "stage_seconds": stage_times,
        "url_media_bytes": imported_record["bytes"],
        "duration_seconds": imported_record["metadata"]["duration"],
        "frames": len(imported_record["frames"]),
        "whisper_segments": len(imported_transcript),
        "structural_equivalence": structural,
        "structural_gates_passed": structural_passed,
        "query_comparisons": comparisons,
        "query_comparisons_passed": sum(row["passed"] for row in comparisons),
        "equivalence_passed": structural_passed
        and sum(row["passed"] for row in comparisons) >= 8
        and all(
            any(row["passed"] for row in comparisons if row["query_id"].startswith(mode))
            for mode in QUERIES
        ),
        "retrieval_modified": False,
        "run_root": str(run_root.relative_to(ROOT)).replace("\\", "/"),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real", action="store_true", help="allow network and local model inference"
    )
    args = parser.parse_args()
    if not args.real:
        raise SystemExit("Pass --real to run the opt-in network/model evaluation.")
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
