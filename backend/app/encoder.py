"""Readable CLIP dual encoder. Model and preprocessing are loaded once per process."""

from functools import lru_cache

import numpy as np

from app.config import settings


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero embeddings cannot be normalized")
    return np.ascontiguousarray(vectors / norms)


class CLIPEncoder:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(4)
        options = {"cache_dir": settings.model_cache, "revision": settings.visual_revision}
        self.processor = CLIPProcessor.from_pretrained(settings.visual_model, **options)
        self.model = CLIPModel.from_pretrained(
            settings.visual_model, use_safetensors=False, **options
        )
        self.model.to(settings.model_device).eval()

    def images(self, paths):
        import torch
        from PIL import Image

        batches = []
        for start in range(0, len(paths), settings.embedding_batch_size):
            images = []
            for path in paths[start : start + settings.embedding_batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(settings.model_device)
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            batches.append(features.cpu().numpy())
        return normalize(np.concatenate(batches))

    def text(self, query):
        import torch

        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(settings.model_device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return normalize(features.cpu().numpy())


@lru_cache(maxsize=1)
def encoder():
    return CLIPEncoder()


def rank_vectors(vectors, query, k):
    import faiss

    vectors, query = normalize(vectors), normalize(query)
    if vectors.shape[1] != query.shape[1]:
        raise ValueError("Query and frame embedding dimensions differ")
    # Exact inner product over unit vectors equals cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, indices = index.search(query, min(k, len(vectors)))
    return scores[0].tolist(), indices[0].tolist()
