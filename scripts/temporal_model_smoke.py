"""Real checkpoint engineering test, with synthetic inputs, never quality data."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.temporal_retrieval import MODEL, REVISION, XClipEncoder  # noqa: E402


def main():
    import torch
    import transformers

    start = time.perf_counter()
    encoder = XClipEncoder(ROOT / "data/models")
    load_seconds = time.perf_counter() - start
    # Fixed synthetic pixels prove implementation equivalence, NOT retrieval quality.
    rng = np.random.default_rng(381)
    frames = list(rng.integers(0, 256, (8, 224, 224, 3), dtype=np.uint8))
    query = "A person opens a door."
    start = time.perf_counter()
    vectors, patches = encoder.video(frames)
    video_seconds = time.perf_counter() - start
    start = time.perf_counter()
    text = encoder.text(query)
    cached = encoder.score(vectors, patches, text)
    query_seconds = time.perf_counter() - start
    inputs = dict(encoder.processor.image_processor(frames, return_tensors="pt"))
    inputs.update(encoder.processor.tokenizer([query], padding=True, return_tensors="pt"))
    with torch.inference_mode():
        native = encoder.model(**inputs).logits_per_video.cpu().numpy().ravel()
        native /= encoder.model.logit_scale.exp().item()
    error = float(np.max(np.abs(native - cached)))
    report = dict(model=MODEL, revision=REVISION, torch=torch.__version__,
                  transformers=transformers.__version__, device="cpu", threads=4,
                  parameters=sum(p.numel() for p in encoder.model.parameters()),
                  load_seconds=load_seconds, synthetic_video_seconds=video_seconds,
                  one_window_query_seconds=query_seconds,
                  rss_bytes=psutil.Process().memory_info().rss,
                  windows_peak_wset_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                  vector_shape=list(vectors.shape), patch_shape=list(patches.shape),
                  native_cosine=native.tolist(), cached_cosine=cached.tolist(),
                  max_absolute_error=error, passed=error <= 1e-5,
                  quality_measurement=False)
    output = ROOT / "data/temporal-v1/model-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Cached scoring diverges from native forward")


if __name__ == "__main__":
    main()
