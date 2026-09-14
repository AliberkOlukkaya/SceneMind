"""Isolated multilingual sentence-retrieval diagnostic after cheap methods fail."""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoTokenizer

from ml.experiments.turkish_compatibility.run_experiment import (
    MANIFEST,
    REPORT,
    build_rows,
    fit_feature_model,
    load_assets,
    metric,
    predict_feature,
)

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e04d38a7d68c60a7a95390045400a555127ab033"


class SentenceEncoder:
    def __init__(self):
        from app.config import settings

        torch.set_num_threads(4)
        started = time.perf_counter()
        before = psutil.Process().memory_info().rss
        options = {"cache_dir": settings.model_cache, "revision": REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        self.model = AutoModel.from_pretrained(MODEL_ID, **options).eval()
        self.load_seconds = time.perf_counter() - started
        self.added_rss_bytes = max(0, psutil.Process().memory_info().rss - before)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 16):
            inputs = self.tokenizer(
                texts[start:start + 16], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, 384), dtype=np.float32)


def semantic_speech(asset: dict, encoder: SentenceEncoder, query: str) -> list[dict]:
    if not asset["segments"]:
        return []
    cache = asset.get("semantic_vectors")
    if cache is None:
        cache = encoder.encode([segment["text"] for segment in asset["segments"]])
        asset["semantic_vectors"] = cache
    query_vector = encoder.encode([query])[0]
    scores = cache @ query_vector
    order = np.argsort(-scores)
    results = []
    for index in order:
        segment = asset["segments"][int(index)]
        nearest = min(
            range(len(asset["timestamps"])),
            key=lambda position: abs(asset["timestamps"][position] - segment["start"]),
        )
        results.append({
            "timestamp": segment["start"],
            "end": segment["end"],
            "thumbnail": f"{nearest:06d}",
            "score": float(scores[index]),
            "text": segment["text"],
            "modality": "speech",
        })
    return results


def main():
    from app.config import settings
    from app.hybrid import fuse

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["promotion_gate_passed"]:
        raise ValueError("semantic diagnostic is only justified after cheap strategies fail")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = load_assets(manifest)
    rows = build_rows(manifest, assets, ["lexical"])
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    encoder = SentenceEncoder()
    for row in rows:
        speech = semantic_speech(
            assets[row["source_id"]], encoder, row["turkish_query"]
        )
        visual = row["strategies"]["lexical"]["VISUAL"]
        row["strategies"]["semantic"] = {
            "VISUAL": visual,
            "SPEECH": speech,
            "HYBRID": fuse(visual, speech, 50),
        }
    router_model = fit_feature_model(calibration)

    def selected_route(row):
        return predict_feature(router_model, row["turkish_query"])

    timings = []
    for row in heldout:
        started = time.perf_counter_ns()
        encoder.encode([row["turkish_query"]])
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    cache_root = (
        Path(settings.model_cache)
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    )
    report["semantic_diagnostic"] = {
        "eligible_for_promotion": False,
        "reason": "Run only after the frozen cheap lexical candidate missed held-out gates.",
        "model": MODEL_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "dimensions": 384,
        "calibration": {
            "speech": metric(calibration, "semantic", "SPEECH"),
            "hybrid": metric(calibration, "semantic", "HYBRID"),
            "auto": metric(calibration, "semantic", selected_route),
        },
        "heldout": {
            "speech": metric(heldout, "semantic", "SPEECH"),
            "hybrid": metric(heldout, "semantic", "HYBRID"),
            "auto": metric(heldout, "semantic", selected_route),
        },
        "resources": {
            "load_seconds": encoder.load_seconds,
            "median_query_latency_ms": statistics.median(timings),
            "p95_query_latency_ms": float(np.percentile(timings, 95)),
            "added_rss_bytes": encoder.added_rss_bytes,
            "cache_bytes": sum(path.stat().st_size for path in cache_root.rglob("*")
                               if path.is_file()) if cache_root.exists() else None,
        },
        "production_dependency_added": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["semantic_diagnostic"], indent=2))


if __name__ == "__main__":
    main()
