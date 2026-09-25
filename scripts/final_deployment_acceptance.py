"""Single frozen final acceptance run through the real FastAPI endpoints.

The source media and the detailed run log remain under ignored data/. The
reviewed manifest and this script must be committed before invoking --run.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import psutil
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

MANIFEST = ROOT / "ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_MANIFEST.json"
DATA = ROOT / "data/final-deployment-v1"
OUTPUT = DATA / "run.json"


class MemoryPeak:
    def __init__(self):
        self.peak = 0
        self.live = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        current = psutil.Process()
        while self.live:
            try:
                procs = [current, *current.children(recursive=True)]
                self.peak = max(self.peak, sum(p.memory_info().rss for p in procs if p.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.live = False
        self.thread.join()


def save(run):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    tmp.replace(OUTPUT)


def relevant(result, intervals):
    return any(start <= result["timestamp"] <= end for start, end in intervals)


def aggregate(rows):
    if not rows:
        return None
    ranks = [r["first_relevant_rank"] for r in rows]
    return {
        "n": len(ranks),
        "r1": sum(rank == 1 for rank in ranks) / len(ranks),
        "r3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "r5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr5": sum(1 / rank for rank in ranks if rank is not None and rank <= 5) / len(ranks),
    }


def summarize(run):
    rows = run["queries"]
    groups = {"overall": rows}
    for key in ("mode", "difficulty", "video_id"):
        for value in sorted({row[key] for row in rows}):
            groups[f"{key}:{value}"] = [row for row in rows if row[key] == value]
    return {key: aggregate(value) for key, value in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    data = MANIFEST.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != args.manifest_sha:
        raise SystemExit(f"Manifest checksum mismatch: {actual}")
    manifest = json.loads(data)
    assert len(manifest["sources"]) >= 8 and len(manifest["queries"]) >= 100
    for relative_path, expected in manifest["frozen_code_sha256"].items():
        if hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Frozen code changed: {relative_path}")
    config = manifest["production_config"]
    if (settings.sampling_interval, settings.speech_model, settings.speech_compute_type,
            settings.visual_model, settings.visual_revision, settings.qa_enabled) != (
            config["sampling_interval_seconds"], config["speech_model"],
            config["speech_compute_type"], config["visual_model"],
            config["visual_revision"], config["qa_enabled"]):
        raise SystemExit("Production inference configuration differs from frozen protocol")
    for source in manifest["sources"]:
        path = DATA / "media" / f"{source['source_id']}.webm"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["media_sha256"]:
            raise SystemExit(f"Media checksum mismatch for {source['source_id']}")
    if not args.run:
        print(json.dumps({"manifest_sha256": actual, "videos": len(manifest["sources"]), "queries": len(manifest["queries"])}))
        return
    if OUTPUT.exists():
        raise SystemExit("A final run already exists; no second run allowed.")
    settings.data_dir = DATA / "runtime" / "videos"
    settings.database_url = f"sqlite:///{(DATA / 'runtime' / 'results.db').as_posix()}"
    settings.durable_jobs = False
    settings.calibration_path = None
    run = {"manifest_sha256": actual, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configuration": {"sampling_interval": settings.sampling_interval, "speech_model": settings.speech_model, "speech_compute_type": settings.speech_compute_type, "visual_model": settings.visual_model, "visual_revision": settings.visual_revision, "rrf_constant": 60, "candidate_k": 50, "result_k": 5, "qa_enabled": settings.qa_enabled, "calibration_path": None}, "videos": {}, "queries": [], "negatives": []}
    save(run)
    headers = {"Authorization": f"Bearer {settings.auth_token}"} if settings.auth_token else {}
    with TestClient(app, headers=headers) as client:
        for source in manifest["sources"]:
            sid = source["source_id"]
            path = DATA / "media" / f"{sid}.webm"
            print(f"PROCESS {sid}", flush=True)
            with MemoryPeak() as memory:
                started = time.perf_counter()
                response = client.post("/videos", params={"filename": path.name}, content=path.read_bytes())
                response.raise_for_status()
                video_id = response.json()["id"]
                ingest_seconds = time.perf_counter() - started
                record = client.get(f"/videos/{video_id}").json()
                if record["status"] != "ready":
                    raise RuntimeError(f"Ingest failed for {sid}: {record.get('error')}")
                speech_seconds = None
                segment_count = None
                if any(q["video_id"] == sid and q["mode"] != "visual" for q in manifest["queries"]):
                    started = time.perf_counter()
                    response = client.post(f"/videos/{video_id}/transcript")
                    response.raise_for_status()
                    transcript = client.get(f"/videos/{video_id}/transcript").json()
                    if transcript["status"] != "ready":
                        raise RuntimeError(f"Speech failed for {sid}: {transcript.get('error')}")
                    speech_seconds = time.perf_counter() - started
                    segment_count = len(transcript["segments"])
                started = time.perf_counter()
                response = client.post(f"/videos/{video_id}/index")
                response.raise_for_status()
                index = client.get(f"/videos/{video_id}/index").json()
                if index["status"] != "ready":
                    raise RuntimeError(f"Visual index failed for {sid}: {index.get('error')}")
                visual_seconds = time.perf_counter() - started
            folder = settings.data_dir / video_id
            disk_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            run["videos"][sid] = {"runtime_id": video_id, "duration_seconds": record["metadata"]["duration"], "ingest_seconds": ingest_seconds, "speech_seconds": speech_seconds, "visual_seconds": visual_seconds, "frame_count": len(record["frames"]), "transcript_segment_count": segment_count, "peak_process_tree_rss_bytes": memory.peak, "persistent_disk_bytes": disk_bytes}
            save(run)
            print(f"READY {sid}: {len(record['frames'])} frames, {segment_count} segments", flush=True)

        for query in manifest["queries"]:
            vid = run["videos"][query["video_id"]]["runtime_id"]
            started = time.perf_counter()
            response = client.get(f"/videos/{vid}/search", params={"q": query["query"], "mode": query["mode"], "k": 5})
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            results = payload["results"]
            rank = next((i for i, item in enumerate(results, 1) if relevant(item, query["intervals"])), None)
            run["queries"].append({"id": query["id"], "video_id": query["video_id"], "mode": query["mode"], "difficulty": query["difficulty"], "first_relevant_rank": rank, "latency_ms": latency_ms, "timestamps": [item["timestamp"] for item in results], "selected_route": payload.get("selected_route")})
            save(run)
        for diagnostic in manifest["negative_diagnostics"]:
            vid = run["videos"][diagnostic["video_id"]]["runtime_id"]
            response = client.get(f"/videos/{vid}/search", params={"q": diagnostic["query"], "mode": diagnostic["mode"], "k": 5})
            run["negatives"].append({**diagnostic, "http_status": response.status_code, "timestamps": [item["timestamp"] for item in response.json().get("results", [])]})
            save(run)
    run["summary"] = summarize(run)
    run["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(run)
    print(json.dumps(run["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
