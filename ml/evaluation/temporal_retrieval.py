"""Isolated temporal experiment. Never imported by production application code."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MODEL = "microsoft/xclip-base-patch32"
REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
SCHEMA = 1


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    samples: tuple[float, ...]

    @property
    def timestamp(self):
        return (self.start + self.end) / 2


def windows(duration, length, stride, frames=8):
    if (not all(math.isfinite(v) and v > 0 for v in (duration, length, stride))
            or stride > length or not isinstance(frames, int) or frames < 2):
        raise ValueError("Positive finite duration/length/stride, stride <= length, frames >=2 required")
    result = []
    for i in range(math.ceil(duration / stride)):
        start, end = i * stride, min(i * stride + length, duration)
        samples = tuple(start + (j + 0.5) * (end - start) / frames for j in range(frames))
        result.append(Window(start, end, samples))
        if end >= duration:
            break
    return result


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 2D vectors")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero vector")
    return np.ascontiguousarray(values / norms)


def decode_window(capture, window):
    """Bounded OpenCV decode; expose actual selected frame positions for audits."""
    import cv2

    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError("Invalid media metadata")
    frames, actual = [], []
    for timestamp in window.samples:
        number = min(count - 1, max(0, math.floor(timestamp * fps + 0.5)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Frame decode failed at {number}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        actual.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    return frames, actual


class XClipEncoder:
    """Cache native vision outputs without dropping video-conditioned prompts."""

    def __init__(self, cache_dir, device="cpu"):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        torch.set_num_threads(4)
        self.device = device
        options = dict(revision=REVISION, cache_dir=cache_dir, local_files_only=True)
        self.processor = XCLIPProcessor.from_pretrained(MODEL, **options)
        self.model = XCLIPModel.from_pretrained(MODEL, use_safetensors=True, **options)
        self.model.to(device).eval()

    def video(self, frames):
        import torch

        if len(frames) != 8:
            raise ValueError("Pinned X-CLIP requires eight frames")
        inputs = self.processor.image_processor(list(frames), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pixels = inputs.pixel_values
            batch, count, channels, height, width = pixels.shape
            vision = self.model.vision_model(
                pixels.reshape(-1, channels, height, width), return_dict=True)
            projected = self.model.visual_projection(vision.pooler_output)
            video = self.model.mit(projected.view(batch, count, -1), return_dict=True).pooler_output
            patches = self.model.prompts_visual_layernorm(vision.last_hidden_state[:, 1:, :])
            patches = patches @ self.model.prompts_visual_projection
            patches = patches.view(batch, count, -1, video.shape[-1]).mean(dim=1)
        return normalize(video.cpu().numpy()), patches.cpu().numpy()

    def text(self, query):
        import torch

        if not query.strip():
            raise ValueError("Empty query")
        inputs = self.processor.tokenizer([query], padding=True, truncation=True,
                                          max_length=77, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.model.get_text_features(**inputs).cpu().numpy()

    def score(self, vectors, patches, text, batch_size=16):
        import torch

        scores = []
        with torch.inference_mode():
            query = torch.as_tensor(text, device=self.device).unsqueeze(0)
            for start in range(0, len(vectors), batch_size):
                visual = torch.as_tensor(patches[start:start + batch_size], device=self.device)
                expanded = query.expand(len(visual), -1, -1)
                conditioned = expanded + self.model.prompts_generator(expanded, visual)
                conditioned = torch.nn.functional.normalize(conditioned, dim=-1)
                video = torch.as_tensor(vectors[start:start + batch_size], device=self.device)
                scores.extend(torch.einsum("bd,bkd->bk", video, conditioned)[:, 0].cpu().tolist())
        return np.asarray(scores, dtype=np.float32)


def save_index(path, clip_windows, vectors, patches, *, identity, actual_samples):
    """One atomic pickle-free archive; identity binds source, config and model."""
    path = Path(path)
    vectors = normalize(vectors)
    patches = np.asarray(patches, dtype=np.float32)
    if (len(clip_windows) != len(vectors) or patches.shape != (len(vectors), 49, 512)
            or vectors.shape[1] != 512 or not np.isfinite(patches).all()
            or len(actual_samples) != len(clip_windows)):
        raise ValueError("Invalid index shape/metadata")
    metadata = dict(schema=SCHEMA, model=MODEL, revision=REVISION, identity=identity,
                    windows=[asdict(w) for w in clip_windows], actual_samples=actual_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, vectors=vectors, patches=patches,
                 metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    temporary.replace(path)


def load_index(path, *, identity):
    with np.load(path, allow_pickle=False) as archive:
        vectors, patches = archive["vectors"], archive["patches"]
        metadata = json.loads(str(archive["metadata"]))
    if (metadata["schema"], metadata["model"], metadata["revision"], metadata["identity"]) != (
            SCHEMA, MODEL, REVISION, identity):
        raise ValueError("Index identity mismatch; cannot reuse different model/source/config")
    clips = [Window(w["start"], w["end"], tuple(w["samples"])) for w in metadata["windows"]]
    if (vectors.shape != (len(clips), 512) or patches.shape != (len(clips), 49, 512)
            or not np.isfinite(patches).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Corrupt index")
    return clips, vectors, patches, metadata


def search(path, query, encoder, *, identity, k=10):
    if not isinstance(k, int) or k < 1:
        raise ValueError("Positive integer k required")
    clips, vectors, patches, _ = load_index(path, identity=identity)
    scores = encoder.score(vectors, patches, encoder.text(query))
    if scores.shape != (len(clips),) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores")
    # Stable chronological ties, no suppression or query-dependent filtering.
    order = np.argsort(-scores, kind="stable")[:k]
    return [dict(start=clips[i].start, end=clips[i].end, timestamp=clips[i].timestamp,
                 score=float(scores[i]), index=int(i)) for i in order]


def first_rank(results, intervals):
    if not intervals or any(not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b)
                            for a, b in intervals):
        raise ValueError("Valid positive intervals required")
    return next((i for i, result in enumerate(results, 1)
                 if any(a <= result["timestamp"] <= b for a, b in intervals)), None)


def aggregate(ranks):
    if not ranks:
        return None
    return {"n": len(ranks), **{f"r{k}": sum(r is not None and r <= k for r in ranks)
                               / len(ranks) for k in (1, 3, 5, 10)},
            "mrr5": sum(1 / r for r in ranks if r is not None and r <= 5) / len(ranks)}


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
