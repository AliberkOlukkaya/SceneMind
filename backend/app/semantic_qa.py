"""Experimental local semantic retrieval and hierarchical Q&A evidence selection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.hybrid import bm25
from app.qa import EvidenceChunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE = "Apache-2.0"
EMBEDDING_DIMENSION = 384
FINE_MAX_SECONDS = 30.0
FINE_MAX_CHARACTERS = 600
CONTEXT_FINE_UNITS = 3
EVIDENCE_MAX_UNITS = 5
EVIDENCE_MAX_CHARACTERS = 3600


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class TranscriptUnit:
    unit_id: str
    level: str
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    segment_ids: tuple[int, ...]
    fine_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticHit:
    unit: TranscriptUnit
    score: float
    rank: int


@dataclass(frozen=True)
class HierarchicalSelection:
    evidence: tuple[EvidenceChunk, ...]
    context_hits: tuple[SemanticHit, ...]
    candidate_fine_ids: tuple[str, ...]
    total_characters: int
    estimated_tokens: int
    temporal_span_seconds: float
    selection_ms: float


class MiniLMEncoder:
    """Pinned local mean-pooled MiniLM encoder using existing Transformers/Torch."""

    dimension = EMBEDDING_DIMENSION

    def __init__(self, cache_dir: Path | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        arguments = {
            "revision": MODEL_REVISION,
            "cache_dir": str(cache_dir) if cache_dir else None,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **arguments)
        self._model = AutoModel.from_pretrained(MODEL_NAME, **arguments)
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        torch = self._torch
        batches = []
        for start in range(0, len(texts), 32):
            encoded = self._tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.inference_mode():
                token_embeddings = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(batches)


def build_representations(video_id: str, segments: list) -> tuple[list[TranscriptUnit], list[TranscriptUnit]]:
    """Build bounded fine units and non-overlapping three-unit context sections."""
    fine: list[TranscriptUnit] = []
    selected = []
    characters = 0

    def flush() -> None:
        nonlocal selected, characters
        if not selected:
            return
        fine.append(
            TranscriptUnit(
                unit_id=f"F{len(fine) + 1:04d}",
                level="fine",
                video_id=video_id,
                start_seconds=round(float(selected[0].start), 3),
                end_seconds=round(float(selected[-1].end), 3),
                text=" ".join(item.text.strip() for item in selected),
                segment_ids=tuple(int(item.id) for item in selected),
                fine_ids=(),
            )
        )
        selected = []
        characters = 0

    for segment in segments:
        proposed = characters + len(segment.text.strip()) + (1 if selected else 0)
        duration = float(segment.end) - float(selected[0].start) if selected else 0.0
        if selected and (proposed > FINE_MAX_CHARACTERS or duration > FINE_MAX_SECONDS):
            flush()
        selected.append(segment)
        characters += len(segment.text.strip()) + (1 if len(selected) > 1 else 0)
    flush()

    context = []
    for start in range(0, len(fine), CONTEXT_FINE_UNITS):
        group = fine[start : start + CONTEXT_FINE_UNITS]
        context.append(
            TranscriptUnit(
                unit_id=f"C{len(context) + 1:04d}",
                level="context",
                video_id=video_id,
                start_seconds=group[0].start_seconds,
                end_seconds=group[-1].end_seconds,
                text=" ".join(item.text for item in group),
                segment_ids=tuple(segment for item in group for segment in item.segment_ids),
                fine_ids=tuple(item.unit_id for item in group),
            )
        )
    return fine, context


class SemanticIndex:
    def __init__(self, units: list[TranscriptUnit], vectors: np.ndarray):
        import faiss

        if vectors.shape != (len(units), EMBEDDING_DIMENSION):
            raise ValueError("semantic vector shape does not match transcript units")
        self.units = list(units)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.index.add(self.vectors)

    @classmethod
    def build(cls, units: list[TranscriptUnit], encoder: TextEncoder):
        return cls(units, encoder.encode([unit.text for unit in units]))

    def search_vector(self, query_vector: np.ndarray, k: int) -> list[SemanticHit]:
        if not self.units or k <= 0:
            return []
        scores, positions = self.index.search(
            np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32),
            min(k, len(self.units)),
        )
        return [
            SemanticHit(self.units[int(position)], float(scores[0][rank]), rank + 1)
            for rank, position in enumerate(positions[0])
            if position >= 0
        ]

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        # FAISS' Windows filename bridge cannot open non-ASCII paths reliably.
        serialized = faiss.serialize_index(self.index)
        (directory / "index.faiss").write_bytes(serialized.tobytes())
        metadata = {
            "schema_version": "1.0.0",
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "dimension": EMBEDDING_DIMENSION,
            "units": [
                {
                    **vars(unit),
                    "segment_ids": list(unit.segment_ids),
                    "fine_ids": list(unit.fine_ids),
                }
                for unit in self.units
            ],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path):
        import faiss

        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["model"] != MODEL_NAME or metadata["revision"] != MODEL_REVISION:
            raise ValueError("semantic index model does not match the pinned model")
        units = [
            TranscriptUnit(
                **{
                    **item,
                    "segment_ids": tuple(item["segment_ids"]),
                    "fine_ids": tuple(item["fine_ids"]),
                }
            )
            for item in metadata["units"]
        ]
        instance = cls.__new__(cls)
        instance.units = units
        serialized = np.frombuffer((directory / "index.faiss").read_bytes(), dtype=np.uint8)
        instance.index = faiss.deserialize_index(serialized)
        if instance.index.d != EMBEDDING_DIMENSION or instance.index.ntotal != len(units):
            raise ValueError("semantic index metadata does not match FAISS data")
        instance.vectors = np.vstack(
            [instance.index.reconstruct(index) for index in range(instance.index.ntotal)]
        ).astype(np.float32)
        return instance


def semantic_retrieve(query: str, index: SemanticIndex, encoder: TextEncoder, k: int):
    return index.search_vector(encoder.encode([query])[0], k)


def hybrid_fine_candidates(
    question: str,
    fine: list[TranscriptUnit],
    fine_hits: list[SemanticHit],
    context_hits: list[SemanticHit],
) -> list[TranscriptUnit]:
    by_id = {unit.unit_id: unit for unit in fine}
    scores: dict[str, float] = {}
    for hit in fine_hits:
        scores[hit.unit.unit_id] = scores.get(hit.unit.unit_id, 0.0) + 1.0 / (20 + hit.rank)
    for hit in context_hits:
        for fine_id in hit.unit.fine_ids:
            scores[fine_id] = scores.get(fine_id, 0.0) + 1.0 / (20 + hit.rank)
    lexical = bm25(question, [unit.text for unit in fine])
    lexical_order = sorted(range(len(fine)), key=lambda i: (-lexical[i], i))
    for rank, position in enumerate(lexical_order[:10], 1):
        if lexical[position] > 0:
            unit_id = fine[position].unit_id
            scores[unit_id] = scores.get(unit_id, 0.0) + 0.5 / (20 + rank)
    return sorted(
        (by_id[unit_id] for unit_id in scores),
        key=lambda unit: (-scores[unit.unit_id], unit.start_seconds, unit.unit_id),
    )


def select_hierarchical_evidence(
    question: str,
    fine: list[TranscriptUnit],
    fine_hits: list[SemanticHit],
    context_hits: list[SemanticHit],
    *,
    max_units: int = EVIDENCE_MAX_UNITS,
    max_characters: int = EVIDENCE_MAX_CHARACTERS,
) -> HierarchicalSelection:
    """Select a bounded fine-grained package discovered through both levels."""
    started = time.perf_counter()
    candidates = hybrid_fine_candidates(question, fine, fine_hits, context_hits)
    selected = []
    characters = 0
    for unit in candidates:
        if len(selected) >= max_units:
            break
        if characters + len(unit.text) > max_characters:
            continue
        if any(set(unit.segment_ids) == set(item.segment_ids) for item in selected):
            continue
        selected.append(unit)
        characters += len(unit.text)
    selected.sort(key=lambda unit: (unit.start_seconds, unit.unit_id))
    evidence = tuple(
        EvidenceChunk(
            evidence_id=unit.unit_id,
            video_id=unit.video_id,
            start_seconds=unit.start_seconds,
            end_seconds=unit.end_seconds,
            text=unit.text,
            segment_ids=unit.segment_ids,
        )
        for unit in selected
    )
    span = (
        max(item.end_seconds for item in evidence) - min(item.start_seconds for item in evidence)
        if evidence
        else 0.0
    )
    return HierarchicalSelection(
        evidence=evidence,
        context_hits=tuple(context_hits),
        candidate_fine_ids=tuple(unit.unit_id for unit in candidates),
        total_characters=characters,
        estimated_tokens=(characters + 3) // 4,
        temporal_span_seconds=span,
        selection_ms=(time.perf_counter() - started) * 1000,
    )
