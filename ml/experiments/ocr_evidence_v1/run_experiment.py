"""Run and score local OCR candidates against the frozen OCR Evidence V1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from app.ocr_evidence import OCRFrame, search_evidence, search_text, suppress_duplicates
from ml.experiments.ocr_evidence_v1.metrics import best_match, character_accuracy, timestamp_error

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "ocr-evidence-v1"
MANIFEST = ROOT / "ml" / "evaluation" / "ocr_evidence_v1_manifest.json"
REPORT = ROOT / "ml" / "evaluation" / "reports" / "ocr-evidence-v1.json"


class RapidAdapter:
    name = "rapidocr-3.9.2-onnxruntime-1.30"

    def __init__(self):
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def read(self, path: Path) -> list[dict]:
        result = self.engine(str(path))
        boxes = result.boxes if result.boxes is not None else []
        texts = result.txts if result.txts is not None else []
        scores = result.scores if result.scores is not None else []
        return [
            {"text": str(text), "confidence": float(score), "box": [[float(x), float(y)] for x, y in box]}
            for box, text, score in zip(boxes, texts, scores)
        ]


class EasyAdapter:
    name = "easyocr-1.7.2-english-cpu"

    def __init__(self):
        import easyocr

        self.engine = easyocr.Reader(
            ["en"], gpu=False, model_storage_directory=str(DATA.parent / "models" / "easyocr"), download_enabled=False
        )

    def read(self, path: Path) -> list[dict]:
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"))
        return [
            {"text": str(text), "confidence": float(score), "box": [[float(x), float(y)] for x, y in box]}
            for box, text, score in self.engine.readtext(image)
        ]


def adapter(name: str):
    return RapidAdapter() if name == "rapidocr" else EasyAdapter()


def line_key(line: dict) -> tuple[float, float]:
    return (min(point[1] for point in line["box"]), min(point[0] for point in line["box"]))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def infer(engine_name: str, manifest: dict, split: str) -> dict:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    ocr = adapter(engine_name)
    rss_model = process.memory_info().rss
    sources = []
    for source in [row for row in manifest["sources"] if row["split"] == split]:
        frames, latencies, peak = [], [], process.memory_info().rss
        started = time.perf_counter()
        for frame in source["production_frames"]:
            tick = time.perf_counter()
            lines = sorted(ocr.read(DATA / frame["path"]), key=line_key)
            latencies.append((time.perf_counter() - tick) * 1000)
            peak = max(peak, process.memory_info().rss)
            frames.append({**frame, "lines": lines, "text": "\n".join(line["text"] for line in lines)})
        sources.append(
            {
                "source_id": source["source_id"],
                "frames": frames,
                "latencies_ms": latencies,
                "latency_ms": {"median": statistics.median(latencies), "p95": percentile(latencies, 0.95), "total": (time.perf_counter() - started) * 1000},
                "peak_rss_delta_bytes": max(0, peak - rss_before),
            }
        )
    return {"engine": ocr.name, "split": split, "model_rss_delta_bytes": max(0, rss_model - rss_before), "sources": sources}


def score(manifest: dict, raw: dict) -> dict:
    source_rows = {row["source_id"]: row for row in manifest["sources"]}
    raw_sources = {row["source_id"]: row for row in raw["sources"]}
    target_rows = [row for row in manifest["targets"] if source_rows[row["source_id"]]["split"] == raw["split"]]
    visible_targets = [row for row in target_rows if row["sampled_frame_ids"]]
    detected = 0
    recognition_pairs = []
    retrieval_hits_all = {1: 0, 3: 0, 5: 0}
    retrieval_hits_visible = {1: 0, 3: 0, 5: 0}
    timestamp_errors = []
    sampling_hits = sum(bool(row["sampled_frame_ids"]) for row in target_rows)
    raw_records = post_records = residual_duplicates = 0
    index_bytes = duration = 0.0
    per_source = []
    for source_id, source in source_rows.items():
        if source["split"] != raw["split"]:
            continue
        observed = raw_sources[source_id]
        frames_by_id = {row["frame_id"]: row for row in observed["frames"]}
        ocr_frames = [
            OCRFrame(row["frame_id"], row["timestamp"], row["text"], statistics.mean([line["confidence"] for line in row["lines"]]) if row["lines"] else None)
            for row in observed["frames"]
        ]
        evidence = suppress_duplicates(ocr_frames)
        raw_records += sum(bool(search_text(row.text)) for row in ocr_frames)
        post_records += len(evidence)
        residual_duplicates += sum(search_text(a.text) == search_text(b.text) for a, b in zip(evidence, evidence[1:]))
        serialized = json.dumps([asdict(row) for row in evidence], ensure_ascii=False).encode()
        index_bytes += len(serialized)
        duration += source["duration_seconds"]
        source_targets = [row for row in target_rows if row["source_id"] == source_id]
        source_detected = 0
        for target in source_targets:
            lines = [line["text"] for frame_id in target["sampled_frame_ids"] for line in frames_by_id[frame_id]["lines"]]
            prediction, ratio = best_match(target["reference_text"], lines)
            if ratio >= 0.75:
                detected += 1
                source_detected += 1
            if target["sampled_frame_ids"]:
                recognition_pairs.append((target["reference_text"], prediction))
            results = search_evidence(target["query"], evidence, 5)
            ranks = [
                rank
                for rank, result in enumerate(results, 1)
                if set(result["frame_ids"]).intersection(target["sampled_frame_ids"])
            ]
            for k in retrieval_hits_all:
                hit = bool(ranks and min(ranks) <= k)
                retrieval_hits_all[k] += hit
                if target["sampled_frame_ids"]:
                    retrieval_hits_visible[k] += hit
            if results and ranks and min(ranks) == 1:
                timestamp_errors.append(timestamp_error(results[0]["start_seconds"], target["visible_intervals_seconds"]))
        per_source.append({"source_id": source_id, "targets": len(source_targets), "detected": source_detected, "latency_ms": observed["latency_ms"], "peak_rss_delta_bytes": observed["peak_rss_delta_bytes"], "index_bytes": len(serialized)})
    denominator = max(1, len(target_rows))
    visible_denominator = max(1, len(visible_targets))
    all_latencies = [latency for source in raw["sources"] for latency in source.get("latencies_ms", [])]
    return {
        "engine": raw["engine"], "split": raw["split"], "target_events": len(target_rows),
        "visible_target_events": len(visible_targets),
        "sampling_coverage": sampling_hits / denominator,
        "text_detection_recall_conditional_visible": detected / visible_denominator,
        "text_accuracy_conditional_visible": character_accuracy(recognition_pairs),
        "retrieval_recall_all_targets": {f"at_{k}": retrieval_hits_all[k] / denominator for k in retrieval_hits_all},
        "retrieval_recall_conditional_visible": {f"at_{k}": retrieval_hits_visible[k] / visible_denominator for k in retrieval_hits_visible},
        "false_text_rate": None,
        "false_text_rate_note": "Requires the predeclared post-run visual line audit; target strings are not exhaustive frame transcriptions.",
        "raw_duplicate_rate": (raw_records - post_records) / max(1, raw_records),
        "residual_duplicate_rate": residual_duplicates / max(1, post_records),
        "median_timestamp_error_seconds": statistics.median(timestamp_errors) if timestamp_errors else None,
        "latency_ms": {"median": statistics.median(all_latencies) if all_latencies else 0.0, "p95": percentile(all_latencies, 0.95), "total": sum(s["latency_ms"]["total"] for s in per_source)},
        "peak_rss_delta_bytes": max((s["peak_rss_delta_bytes"] for s in per_source), default=0),
        "index_bytes": int(index_bytes), "index_bytes_per_source_minute": index_bytes / max(duration / 60, 1e-9),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["rapidocr", "easyocr"], required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    args = parser.parse_args()
    payload = MANIFEST.read_bytes()
    manifest = json.loads(payload)
    expected = (MANIFEST.with_suffix(".sha256")).read_text().split()[0] if args.split == "validation" else None
    if expected and hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit("frozen validation manifest checksum mismatch")
    raw = infer(args.engine, manifest, args.split)
    result = {"manifest_sha256": hashlib.sha256(payload).hexdigest(), "metrics": score(manifest, raw)}
    raw_path = DATA / f"raw-{args.split}-{args.engine}.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.split == "validation":
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (DATA / f"development-{args.engine}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
