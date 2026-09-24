"""Experimental hierarchical transcript memory for navigation-only Q&A retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.hybrid import bm25
from app.qa import EvidenceChunk
from app.semantic_qa import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
    SemanticIndex,
    TranscriptUnit,
    build_representations,
)

MEMORY_SCHEMA_VERSION = "1.0.0"
ALGORITHM_VERSION = "hierarchical-video-memory-v1-extractive-1"
SECTION_MIN_SECONDS = 35.0
SECTION_MAX_SECONDS = 180.0
SECTION_PAUSE_SECONDS = 3.0
SECTION_TOPIC_SIMILARITY = 0.40
SECTION_TOP_K = 3
FINE_CANDIDATE_K = 12
EVIDENCE_MAX_UNITS = 5
EVIDENCE_MAX_CHARACTERS = 3600
SUMMARY_MAX_UNITS = 2
SUMMARY_MAX_CHARACTERS = 420

_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_STOP = frozenset(
    "the and for that with this from have are was were will would could should into about "
    "your you they their them but not what when where which who how why can has had its our "
    "there here then than also just some more very does did because been being these those".split()
)


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class SectionMemory:
    section_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    fine_ids: tuple[str, ...]
    segment_ids: tuple[int, ...]
    summary: str
    summary_fine_ids: tuple[str, ...]
    topics: tuple[str, ...]

    @property
    def navigation_text(self) -> str:
        topics = ", ".join(self.topics)
        return f"{self.summary}\nTopics: {topics}" if topics else self.summary


@dataclass(frozen=True)
class MemoryBundle:
    video_id: str
    transcript_fingerprint: str
    fine: tuple[TranscriptUnit, ...]
    sections: tuple[SectionMemory, ...]


@dataclass(frozen=True)
class MemorySelection:
    section_ids: tuple[str, ...]
    section_scores: tuple[float, ...]
    evidence: tuple[EvidenceChunk, ...]
    candidate_fine_ids: tuple[str, ...]
    total_characters: int
    estimated_tokens: int
    section_retrieval_ms: float
    local_selection_ms: float
    retrieval_ms: float


def transcript_fingerprint(video_id: str, segments: list) -> str:
    payload = [
        [int(item.id), round(float(item.start), 3), round(float(item.end), 3), item.text.strip()]
        for item in segments
    ]
    return hashlib.sha256(
        json.dumps([video_id, payload], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def configuration_fingerprint() -> str:
    values = {
        "algorithm": ALGORITHM_VERSION,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "min": SECTION_MIN_SECONDS,
        "max": SECTION_MAX_SECONDS,
        "pause": SECTION_PAUSE_SECONDS,
        "similarity": SECTION_TOPIC_SIMILARITY,
        "summary_units": SUMMARY_MAX_UNITS,
        "summary_chars": SUMMARY_MAX_CHARACTERS,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _topic_terms(text: str, limit: int = 8) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for position, match in enumerate(_WORDS.finditer(text.casefold())):
        word = match.group()
        if word in _STOP:
            continue
        counts[word] = counts.get(word, 0) + 1
        first.setdefault(word, position)
    ordered = sorted(counts, key=lambda item: (-counts[item], first[item], item))
    return tuple(ordered[:limit])


def _summarize(group: list[TranscriptUnit], vectors: np.ndarray) -> tuple[str, tuple[str, ...]]:
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid /= norm
    ranked = sorted(
        range(len(group)),
        key=lambda index: (-float(vectors[index] @ centroid), index),
    )
    chosen: list[int] = []
    characters = 0
    for index in ranked:
        text = group[index].text.strip()
        if len(chosen) >= SUMMARY_MAX_UNITS or characters + len(text) > SUMMARY_MAX_CHARACTERS:
            continue
        chosen.append(index)
        characters += len(text)
    if not chosen:
        chosen = [0]
    chosen.sort()
    return " ".join(group[index].text.strip() for index in chosen), tuple(
        group[index].unit_id for index in chosen
    )


def build_memory(video_id: str, segments: list, encoder: TextEncoder) -> MemoryBundle:
    """Build L1 units, semantic L2 sections, and extractive L3 navigation memory."""
    fine, _ = build_representations(video_id, segments)
    if not fine:
        return MemoryBundle(video_id, transcript_fingerprint(video_id, segments), (), ())
    vectors = encoder.encode([item.text for item in fine])
    if vectors.shape != (len(fine), EMBEDDING_DIMENSION):
        raise ValueError("fine-unit vectors have the wrong shape")

    groups: list[list[int]] = []
    current = [0]
    for index in range(1, len(fine)):
        section_start = fine[current[0]].start_seconds
        duration_if_added = fine[index].end_seconds - section_start
        current_duration = fine[index - 1].end_seconds - section_start
        pause = fine[index].start_seconds - fine[index - 1].end_seconds
        similarity = float(vectors[index - 1] @ vectors[index])
        boundary = duration_if_added > SECTION_MAX_SECONDS or (
            current_duration >= SECTION_MIN_SECONDS
            and (pause >= SECTION_PAUSE_SECONDS or similarity < SECTION_TOPIC_SIMILARITY)
        )
        if boundary:
            groups.append(current)
            current = []
        current.append(index)
    groups.append(current)

    # Avoid an unhelpfully tiny tail while respecting the maximum duration.
    if len(groups) > 1:
        last = groups[-1]
        previous = groups[-2]
        last_duration = fine[last[-1]].end_seconds - fine[last[0]].start_seconds
        merged_duration = fine[last[-1]].end_seconds - fine[previous[0]].start_seconds
        if last_duration < SECTION_MIN_SECONDS and merged_duration <= SECTION_MAX_SECONDS:
            groups[-2] = previous + last
            groups.pop()

    sections = []
    for number, positions in enumerate(groups, 1):
        group = [fine[index] for index in positions]
        summary, summary_ids = _summarize(group, vectors[positions])
        sections.append(
            SectionMemory(
                section_id=f"S{number:04d}",
                video_id=video_id,
                start_seconds=group[0].start_seconds,
                end_seconds=group[-1].end_seconds,
                fine_ids=tuple(item.unit_id for item in group),
                segment_ids=tuple(value for item in group for value in item.segment_ids),
                summary=summary,
                summary_fine_ids=summary_ids,
                topics=_topic_terms(" ".join(item.text for item in group)),
            )
        )
    return MemoryBundle(
        video_id,
        transcript_fingerprint(video_id, segments),
        tuple(fine),
        tuple(sections),
    )


class SectionMemoryIndex:
    def __init__(
        self, bundle: MemoryBundle, section_vectors: np.ndarray, fine_vectors: np.ndarray
    ):
        units = [
            TranscriptUnit(
                unit_id=item.section_id,
                level="memory",
                video_id=item.video_id,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                text=item.navigation_text,
                segment_ids=item.segment_ids,
                fine_ids=item.fine_ids,
            )
            for item in bundle.sections
        ]
        self.bundle = bundle
        self.semantic = SemanticIndex(units, section_vectors)
        self.fine_semantic = SemanticIndex(list(bundle.fine), fine_vectors)

    @classmethod
    def build(cls, bundle: MemoryBundle, encoder: TextEncoder):
        section_vectors = encoder.encode([item.navigation_text for item in bundle.sections])
        fine_vectors = encoder.encode([item.text for item in bundle.fine])
        return cls(bundle, section_vectors, fine_vectors)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.semantic.save(directory / "sections")
        self.fine_semantic.save(directory / "fine")
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "configuration_fingerprint": configuration_fingerprint(),
            "video_id": self.bundle.video_id,
            "transcript_fingerprint": self.bundle.transcript_fingerprint,
            "fine": [asdict(item) for item in self.bundle.fine],
            "sections": [asdict(item) for item in self.bundle.sections],
        }
        (directory / "memory.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, *, expected_transcript_fingerprint: str):
        payload = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise ValueError("memory schema version mismatch")
        if payload.get("configuration_fingerprint") != configuration_fingerprint():
            raise ValueError("memory configuration is stale")
        if payload.get("transcript_fingerprint") != expected_transcript_fingerprint:
            raise ValueError("memory transcript is stale")
        fine = tuple(
            TranscriptUnit(
                **{
                    **item,
                    "segment_ids": tuple(item["segment_ids"]),
                    "fine_ids": tuple(item["fine_ids"]),
                }
            )
            for item in payload["fine"]
        )
        sections = tuple(
            SectionMemory(
                **{
                    **item,
                    "fine_ids": tuple(item["fine_ids"]),
                    "segment_ids": tuple(item["segment_ids"]),
                    "summary_fine_ids": tuple(item["summary_fine_ids"]),
                    "topics": tuple(item["topics"]),
                }
            )
            for item in payload["sections"]
        )
        bundle = MemoryBundle(
            payload["video_id"], payload["transcript_fingerprint"], fine, sections
        )
        semantic = SemanticIndex.load(directory / "sections")
        fine_semantic = SemanticIndex.load(directory / "fine")
        return cls(bundle, semantic.vectors, fine_semantic.vectors)


def retrieve_from_memory(
    question: str,
    index: SectionMemoryIndex,
    encoder: TextEncoder,
    *,
    section_k: int = SECTION_TOP_K,
    fine_k: int = FINE_CANDIDATE_K,
    max_units: int = EVIDENCE_MAX_UNITS,
    max_characters: int = EVIDENCE_MAX_CHARACTERS,
) -> MemorySelection:
    """Navigate with L3 memory, then return only original L1 transcript evidence."""
    started = time.perf_counter()
    query_vector = encoder.encode([question])[0]
    section_hits = index.semantic.search_vector(query_vector, section_k)
    section_finished = time.perf_counter()
    selected_sections = {hit.unit.unit_id for hit in section_hits}
    allowed_ids = {
        fine_id
        for section in index.bundle.sections
        if section.section_id in selected_sections
        for fine_id in section.fine_ids
    }
    candidates = [item for item in index.bundle.fine if item.unit_id in allowed_ids]
    semantic_scores = {
        item.unit_id: float(vector @ query_vector)
        for item, vector in zip(index.bundle.fine, index.fine_semantic.vectors)
        if item.unit_id in allowed_ids
    }
    lexical = bm25(question, [item.text for item in candidates]) if candidates else []
    lexical_order = sorted(range(len(candidates)), key=lambda i: (-lexical[i], i))[:fine_k]
    scores = {item.unit_id: semantic_scores[item.unit_id] for item in candidates}
    for rank, position in enumerate(lexical_order, 1):
        if lexical[position] > 0:
            scores[candidates[position].unit_id] += 0.10 / rank
    ranked = sorted(candidates, key=lambda item: (-scores[item.unit_id], item.start_seconds))[:fine_k]
    selected = []
    characters = 0
    for item in ranked:
        if len(selected) >= max_units:
            break
        if characters + len(item.text) > max_characters:
            continue
        selected.append(item)
        characters += len(item.text)
    selected.sort(key=lambda item: (item.start_seconds, item.unit_id))
    evidence = tuple(
        EvidenceChunk(
            evidence_id=item.unit_id,
            video_id=item.video_id,
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            text=item.text,
            segment_ids=item.segment_ids,
        )
        for item in selected
    )
    finished = time.perf_counter()
    return MemorySelection(
        section_ids=tuple(hit.unit.unit_id for hit in section_hits),
        section_scores=tuple(hit.score for hit in section_hits),
        evidence=evidence,
        candidate_fine_ids=tuple(item.unit_id for item in ranked),
        total_characters=characters,
        estimated_tokens=(characters + 3) // 4,
        section_retrieval_ms=(section_finished - started) * 1000,
        local_selection_ms=(finished - section_finished) * 1000,
        retrieval_ms=(finished - started) * 1000,
    )
