"""Pinned UForm3-small ONNX image-text scorer for top-K candidates."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


@dataclass(frozen=True)
class UFormSpec:
    model_id: str = "unum-cloud/uform3-image-text-english-small"
    revision: str = "a8d990ed0dc6340b9c25cb47f06f86b9ecb67fa6"
    license: str = "Apache-2.0"
    parameter_count: int = 79_000_000
    image_size: int = 224
    max_text_tokens: int = 64
    embedding_dimensions: int = 256
    checkpoint_bytes: int = 60_584_430
    architecture: str = "ViT-S/16 image encoder plus four-layer BERT text encoder"
    scoring: str = "L2-normalized dual-encoder embedding cosine similarity"

    def as_dict(self) -> dict:
        return asdict(self)


class UFormOnnxScorer:
    def __init__(self, spec: UFormSpec, cache_dir: str | Path, threads: int = 0):
        import onnxruntime as ort
        from uform.numpy_processors import ImageProcessor, TextProcessor

        started = time.perf_counter()
        self.spec = spec
        self.threads = threads
        snapshot = Path(
            snapshot_download(
                spec.model_id,
                revision=spec.revision,
                allow_patterns=[
                    "config.json", "tokenizer.json", "image_encoder.onnx", "text_encoder.onnx"
                ],
                cache_dir=cache_dir,
            )
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        if threads:
            options.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self.image_session = ort.InferenceSession(
            str(snapshot / "image_encoder.onnx"), sess_options=options, providers=providers
        )
        self.text_session = ort.InferenceSession(
            str(snapshot / "text_encoder.onnx"), sess_options=options, providers=providers
        )
        self.image_processor = ImageProcessor(str(snapshot / "config.json"))
        self.text_processor = TextProcessor(
            str(snapshot / "config.json"), str(snapshot / "tokenizer.json")
        )
        self.model_files = {
            name: (snapshot / name).stat().st_size
            for name in ("image_encoder.onnx", "text_encoder.onnx")
        }
        self.providers = self.image_session.get_providers()
        self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)

    def _score(self, query: str, image_paths: list[Path]) -> list[float]:
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        image_inputs = self.image_processor(images)
        text_inputs = self.text_processor(query)
        _, image_embeddings = self.image_session.run(None, {"images": image_inputs})
        _, text_embeddings = self.text_session.run(None, text_inputs)
        image_embeddings = self._normalize(image_embeddings)
        text_embeddings = self._normalize(text_embeddings)
        return (image_embeddings @ text_embeddings[0]).tolist()

    def timed_score(self, query: str, image_paths: list[Path]) -> tuple[list[float], float]:
        started = time.perf_counter()
        scores = self._score(query, image_paths)
        return scores, time.perf_counter() - started

    def warm(self, query: str, image_paths: list[Path], repeats: int = 3) -> None:
        for _ in range(repeats):
            self._score(query, image_paths)
