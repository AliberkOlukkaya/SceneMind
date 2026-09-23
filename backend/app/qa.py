"""Transcript-grounded question answering with server-resolved citations."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Segment, Transcript, engine
from app.hybrid import bm25
from app.video import folder_for, read_manifest

router = APIRouter(prefix="/videos", tags=["grounded video Q&A"])
ABSTENTION = "I couldn't find enough evidence in this video to answer that reliably."


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    segment_ids: tuple[int, ...]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class GeneratedAnswer(BaseModel):
    answerable: bool
    answer: str = Field(max_length=2000)
    evidence_ids: list[str] = Field(max_length=5)


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(
        self, question: str, evidence: list[EvidenceChunk]
    ) -> tuple[GeneratedAnswer, dict]:
        """Return a structured answer and provider usage metadata."""


def chunk_segments(
    video_id: str,
    segments: list,
    *,
    max_seconds: float | None = None,
    max_characters: int | None = None,
) -> list[EvidenceChunk]:
    """Group adjacent Whisper segments with one-segment overlap."""
    max_seconds = max_seconds or settings.qa_chunk_seconds
    max_characters = max_characters or settings.qa_chunk_characters
    chunks: list[EvidenceChunk] = []
    start = 0
    while start < len(segments):
        selected = []
        characters = 0
        for segment in segments[start:]:
            proposed = characters + len(segment.text) + (1 if selected else 0)
            duration = segment.end - segments[start].start
            if selected and (proposed > max_characters or duration > max_seconds):
                break
            selected.append(segment)
            characters = proposed
        if not selected:
            selected = [segments[start]]
        chunks.append(
            EvidenceChunk(
                evidence_id=f"E{len(chunks) + 1:03d}",
                video_id=video_id,
                start_seconds=round(float(selected[0].start), 3),
                end_seconds=round(float(selected[-1].end), 3),
                text=" ".join(segment.text.strip() for segment in selected),
                segment_ids=tuple(segment.id for segment in selected),
            )
        )
        if start + len(selected) >= len(segments):
            break
        start += max(1, len(selected) - 1)
    return chunks


def retrieve_evidence(question: str, chunks: list[EvidenceChunk], k: int | None = None):
    scores = bm25(question, [chunk.text for chunk in chunks])
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
    return [chunks[index] for index in ranked if scores[index] > 0][: k or settings.qa_top_k]


class OpenAIAnswerGenerator(AnswerGenerator):
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openai_api_key

    def _payload(self, question: str, evidence: list[EvidenceChunk]) -> dict:
        evidence_text = "\n\n".join(f"{item.evidence_id}: {item.text}" for item in evidence)
        instructions = (
            "Answer only from the supplied video transcript evidence. Do not use outside "
            "knowledge or infer unsupported details. If the evidence is insufficient, set "
            "answerable to false and use no evidence IDs. Cite only supplied evidence IDs. "
            "Keep the answer concise and include every evidence ID needed to support its claims."
        )
        schema = {
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "answer": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
            },
            "required": ["answerable", "answer", "evidence_ids"],
            "additionalProperties": False,
        }
        return {
            "model": settings.qa_model,
            "store": False,
            "instructions": instructions,
            "input": f"Question: {question}\n\nVideo evidence:\n{evidence_text}",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "grounded_video_answer",
                    "strict": True,
                    "schema": schema,
                }
            },
        }

    def generate(self, question: str, evidence: list[EvidenceChunk]):
        if not self.api_key:
            raise HTTPException(
                503,
                "Ask Video is not configured. Set OPENAI_API_KEY in the local .env.local file.",
            )
        last_error = None
        for attempt in range(settings.qa_retries + 1):
            try:
                with httpx.Client(timeout=settings.qa_timeout) as client:
                    response = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=self._payload(question, evidence),
                    )
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt < settings.qa_retries
                ):
                    last_error = response
                    continue
                response.raise_for_status()
                body = response.json()
                output_text = body.get("output_text")
                if output_text is None:
                    output_text = next(
                        (
                            part.get("text")
                            for item in body.get("output", [])
                            for part in item.get("content", [])
                            if part.get("type") == "output_text"
                        ),
                        None,
                    )
                generated = GeneratedAnswer.model_validate(json.loads(output_text or ""))
                usage = body.get("usage", {})
                return generated, {
                    "provider": "openai",
                    "model": body.get("model", settings.qa_model),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                }
            except httpx.TimeoutException as error:
                if attempt >= settings.qa_retries:
                    raise HTTPException(504, "The answer provider timed out. Try again.") from error
                last_error = error
            except (httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError) as error:
                raise HTTPException(
                    502, "The answer provider returned an invalid response."
                ) from error
        raise HTTPException(502, "The answer provider is temporarily unavailable.") from last_error


def resolve_answer(generated: GeneratedAnswer, evidence: list[EvidenceChunk]) -> dict:
    lookup = {item.evidence_id: item for item in evidence}
    unknown = [item for item in generated.evidence_ids if item not in lookup]
    if unknown:
        raise HTTPException(502, "The answer provider returned an invalid evidence citation.")
    if not generated.answerable:
        return {"answerable": False, "answer": ABSTENTION, "citations": []}
    citation_ids = list(dict.fromkeys(generated.evidence_ids))
    if not citation_ids or not generated.answer.strip():
        raise HTTPException(502, "The answer provider returned an ungrounded answer.")
    return {
        "answerable": True,
        "answer": generated.answer.strip(),
        "citations": [asdict(lookup[evidence_id]) for evidence_id in citation_ids],
    }


def answer_generator() -> AnswerGenerator:
    return OpenAIAnswerGenerator()


@router.get("/{video_id}/ask/status")
def ask_status(video_id: str):
    read_manifest(folder_for(video_id))
    return {
        "enabled": settings.qa_enabled,
        "configured": settings.qa_enabled and bool(settings.openai_api_key),
        "provider": "openai",
        "model": settings.qa_model,
    }


@router.post("/{video_id}/ask")
def ask_video(video_id: str, request: AskRequest):
    started = time.perf_counter()
    folder = folder_for(video_id)
    record = read_manifest(folder)
    if record.get("status") != "ready":
        raise HTTPException(409, "Wait for video processing to finish.")
    if not settings.qa_enabled:
        raise HTTPException(503, "Ask Video is evaluation-gated and is not enabled.")
    question = request.question.strip()
    if not question:
        raise HTTPException(422, "Question cannot be empty.")
    with Session(engine()) as session:
        transcript = session.get(Transcript, folder.name)
        if transcript is None or transcript.status != "ready":
            raise HTTPException(409, "A ready transcript is required before asking this video.")
        segments = session.scalars(
            select(Segment).where(Segment.video_id == folder.name).order_by(Segment.start)
        ).all()
    if not segments:
        raise HTTPException(409, "The transcript contains no spoken evidence.")
    retrieval_started = time.perf_counter()
    evidence = retrieve_evidence(question, chunk_segments(folder.name, segments))
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    if not evidence:
        return {
            "answerable": False,
            "answer": ABSTENTION,
            "citations": [],
            "evidence": [],
            "usage": {"provider": "none", "model": None, "input_tokens": 0, "output_tokens": 0},
            "latency_ms": {
                "retrieval": retrieval_ms,
                "generation": 0,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
    generation_started = time.perf_counter()
    generated, usage = answer_generator().generate(question, evidence)
    generation_ms = (time.perf_counter() - generation_started) * 1000
    answer = resolve_answer(generated, evidence)
    return {
        **answer,
        "evidence": [asdict(item) for item in evidence],
        "usage": usage,
        "latency_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": (time.perf_counter() - started) * 1000,
        },
    }
