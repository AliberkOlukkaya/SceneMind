"""Experimental timestamped OCR evidence primitives; not wired into production search."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.hybrid import bm25

OCR_SCHEMA_VERSION = "1.0.0"
DEDUP_SIMILARITY = 0.88
DEDUP_MAX_GAP_SECONDS = 5.25


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR output while retaining punctuation needed by code and errors."""
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u00ad", "").replace("\ufeff", "")
    value = re.sub(r"[\t\r\n]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def search_text(text: str) -> str:
    value = normalize_ocr_text(text).casefold()
    return " ".join(re.findall(r"[^\W_]+(?:[-.'/:][^\W_]+)*|[-]{1,2}[a-z0-9-]+", value))


@dataclass(frozen=True)
class OCRFrame:
    frame_id: str
    timestamp: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OCREvidence:
    evidence_id: str
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None
    frame_ids: tuple[str, ...]


def _similar(left: str, right: str) -> float:
    a, b = search_text(left), search_text(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def suppress_duplicates(
    frames: list[OCRFrame],
    *,
    similarity: float = DEDUP_SIMILARITY,
    max_gap_seconds: float = DEDUP_MAX_GAP_SECONDS,
) -> list[OCREvidence]:
    """Merge adjacent repeated screen text while preserving its full time span."""
    evidence: list[OCREvidence] = []
    for frame in sorted(frames, key=lambda item: (item.timestamp, item.frame_id)):
        text = normalize_ocr_text(frame.text)
        if not text:
            continue
        previous = evidence[-1] if evidence else None
        if (
            previous
            and frame.timestamp - previous.end_seconds <= max_gap_seconds
            and _similar(previous.text, text) >= similarity
        ):
            confidence_values = [x for x in (previous.confidence, frame.confidence) if x is not None]
            evidence[-1] = OCREvidence(
                evidence_id=previous.evidence_id,
                start_seconds=previous.start_seconds,
                end_seconds=frame.timestamp,
                text=text if len(search_text(text)) > len(search_text(previous.text)) else previous.text,
                confidence=(sum(confidence_values) / len(confidence_values) if confidence_values else None),
                frame_ids=(*previous.frame_ids, frame.frame_id),
            )
            continue
        evidence.append(
            OCREvidence(
                evidence_id=f"ocr-{len(evidence) + 1:05d}",
                start_seconds=frame.timestamp,
                end_seconds=frame.timestamp,
                text=text,
                confidence=frame.confidence,
                frame_ids=(frame.frame_id,),
            )
        )
    return evidence


def search_evidence(query: str, evidence: list[OCREvidence], k: int = 5) -> list[dict]:
    documents = [search_text(item.text) for item in evidence]
    scores = bm25(search_text(query), documents)
    ranked = sorted(range(len(evidence)), key=lambda index: (-scores[index], evidence[index].start_seconds))
    return [
        {**asdict(evidence[index]), "score": scores[index]}
        for index in ranked
        if scores[index] > 0
    ][:k]


def source_fingerprint(frames: list[OCRFrame], engine: str, configuration: dict) -> str:
    payload = {
        "schema": OCR_SCHEMA_VERSION,
        "engine": engine,
        "configuration": configuration,
        "frames": [asdict(item) for item in frames],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_evidence(
    path: Path,
    evidence: list[OCREvidence],
    *,
    engine: str,
    configuration: dict,
    fingerprint: str,
) -> None:
    payload = {
        "schema_version": OCR_SCHEMA_VERSION,
        "engine": engine,
        "configuration": configuration,
        "source_fingerprint": fingerprint,
        "evidence": [asdict(item) for item in evidence],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_evidence(path: Path, *, expected_fingerprint: str) -> list[OCREvidence]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != OCR_SCHEMA_VERSION:
        raise ValueError("OCR evidence schema changed; rebuild the index.")
    if payload.get("source_fingerprint") != expected_fingerprint:
        raise ValueError("OCR evidence source or configuration changed; rebuild the index.")
    return [
        OCREvidence(**{**row, "frame_ids": tuple(row["frame_ids"])})
        for row in payload["evidence"]
    ]
