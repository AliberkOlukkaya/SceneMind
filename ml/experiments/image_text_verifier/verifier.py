"""Pinned BLIP image-text matching scorer; no production imports."""

import statistics
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerifierSpec:
    model_id: str = "Salesforce/blip-itm-base-coco"
    revision: str = "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"
    license: str = "BSD-3-Clause"
    match_label_index: int = 1
    max_text_tokens: int = 35
    image_size: int = 384


class BlipVerifier:
    def __init__(self, spec: VerifierSpec, cache_dir: Path, device: str = "cpu"):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.spec = spec
        self.device = device
        started = time.perf_counter()
        self.processor = BlipProcessor.from_pretrained(
            spec.model_id, revision=spec.revision, cache_dir=cache_dir
        )
        self.model = BlipForImageTextRetrieval.from_pretrained(
            spec.model_id, revision=spec.revision, cache_dir=cache_dir
        ).to(device).eval()
        self.load_seconds = time.perf_counter() - started
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self._torch = torch

    def score(self, query: str, image_paths: list[Path]) -> tuple[list[float], float]:
        from PIL import Image

        images = []
        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            inputs = self.processor(
                images=images,
                text=batched_text(query, len(images)),
                padding=True,
                truncation=True,
                max_length=self.spec.max_text_tokens,
                return_tensors="pt",
            ).to(self.device)
            started = time.perf_counter()
            with self._torch.inference_mode():
                logits = self.model(**inputs, use_itm_head=True).itm_score
                scores = self._torch.softmax(logits, dim=-1)[:, self.spec.match_label_index]
            return scores.detach().cpu().tolist(), time.perf_counter() - started
        finally:
            for image in images:
                image.close()

    def warm(self, image_paths: list[Path]) -> None:
        self.score("a photograph", image_paths[:1])


def median_latency(rows: list[dict]) -> float:
    return statistics.median(row["verifier_seconds"] for row in rows)


def batched_text(query: str, count: int) -> list[str]:
    if not query.strip() or count < 1:
        raise ValueError("nonempty query and positive batch size required")
    return [query] * count
