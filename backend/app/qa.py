"""Transcript-grounded question answering with server-resolved citations."""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

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
OUT_OF_SCOPE = (
    "I can't answer that type of question reliably yet. Try asking about a specific "
    "fact or explanation spoken in the video."
)
VISUAL_OUT_OF_SCOPE = (
    "I can't reliably answer visual-only questions yet. Ask about something spoken "
    "in the video."
)
QA_INSTRUCTIONS = (
    "Decide whether the supplied video transcript evidence directly and completely answers "
    "the actual question. Topical similarity is not sufficient. Do not use outside knowledge "
    "or infer missing details. Require the correct entity, requested relation, temporal order, "
    "and complete requested count. Evidence may be combined across supplied chunks. Imperfect "
    "automatic-transcription wording is acceptable when the intended fact is still clear; do "
    "not demand exact phrasing or one self-contained chunk. For before/after questions, evidence "
    "must establish both the anchor and the requested event. Put each independently checkable "
    "factual statement in a separate claim with all supporting evidence IDs. For a requested "
    "list, put each item in a separate claim. Set answerable=false if any required fact is "
    "absent, and name the gap briefly in unsupported_or_missing; then return an empty answer, "
    "evidence_ids, and claims. When answerable=true, unsupported_or_missing must be empty. Cite "
    "only IDs from the supplied evidence and keep the answer concise."
)
QA_STRUCTURED_INSTRUCTIONS = QA_INSTRUCTIONS + (
    " Evidence labels identify bounded structured context. For temporal questions, cite at least "
    "one TEMPORAL_ANCHOR evidence ID and at least one TEMPORAL_TARGET ID, and return those IDs in "
    "their dedicated arrays. The anchor establishes ordering and does not need to be repeated on "
    "an answer claim unless it also supports that claim. The target must be in the requested "
    "direction. The top-level evidence_ids must contain exactly the union of claim evidence_ids. "
    "For explicit lists, "
    "return exactly the requested number of distinct claims; do not split one requirement into "
    "synonyms or omit a required item merely to reach the count. Leave both temporal ID arrays "
    "empty for non-temporal questions."
)
MAX_STRUCTURED_EVIDENCE = 8
MAX_ADDITIONAL_EVIDENCE = 3
MAX_STRUCTURED_CHARACTERS = 6000
MAX_EXPANSION_SECONDS = 90.0


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    segment_ids: tuple[int, ...]
    role: str = "RELEVANT"


@dataclass(frozen=True)
class EvidenceSelection:
    base: tuple[EvidenceChunk, ...]
    expanded: tuple[EvidenceChunk, ...]
    added_count: int
    added_characters: int
    temporal_span_seconds: float
    expansion_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@dataclass(frozen=True)
class ScopeDecision:
    supported: bool
    category: str
    response: str | None = None


class GeneratedAnswer(BaseModel):
    answerable: bool
    answer: str = Field(max_length=2000)
    evidence_ids: list[str] = Field(max_length=5)
    claims: list["GeneratedClaim"] = Field(max_length=8)
    unsupported_or_missing: list[str] = Field(max_length=8)
    # Optional only so frozen historical structured-evidence artifacts remain readable.
    # The production core neither requests nor accepts temporal answers.
    temporal_anchor_ids: list[str] = Field(default_factory=list, max_length=3)
    temporal_target_ids: list[str] = Field(default_factory=list, max_length=3)


class GeneratedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)


@dataclass(frozen=True)
class QuestionConstraints:
    requested_count: int | None = None
    temporal_relation: str | None = None
    temporal_anchor: str | None = None
    relation_types: tuple[str, ...] = ()


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


def classify_question_scope(question: str) -> ScopeDecision:
    """Reject capabilities outside the deliberately narrow spoken-Q&A core."""
    lowered = " ".join(question.casefold().split())
    constraints = analyze_question(question)
    if constraints.requested_count is not None or re.search(
        r"\b(?:list|enumerate)\b", lowered
    ):
        return ScopeDecision(False, "EXPLICIT_LIST_COUNT", OUT_OF_SCOPE)
    if (
        constraints.temporal_relation
        and re.search(r"^(?:what|which|who|where|when)\b", lowered)
    ) or re.search(r"\b(?:first|last)\s+(?:thing|step|event)\b", lowered):
        return ScopeDecision(False, "TEMPORAL_ORDERING", OUT_OF_SCOPE)
    if re.search(
        r"\b(?:entire|whole)\s+video\b|\bthroughout\s+the\s+video\b|"
        r"\b(?:compare|connect)\b.*\b(?:beginning|start)\b.*\b(?:end|ending)\b|"
        r"\b(?:all|every)\s+(?:argument|topic|point|claim)s?\b",
        lowered,
    ):
        return ScopeDecision(False, "LONG_RANGE_COMPOSITIONAL", OUT_OF_SCOPE)
    if re.search(
        r"\b(?:what does|what do)\b.*\b(?:slide|screen|caption|label|button|"
        r"error message|code)\b.*\b(?:say|read|show)\b|"
        r"\btext on (?:the )?screen\b|"
        r"\b(?:read|transcribe)\s+(?:the\s+)?(?:slide|screen|caption)\b",
        lowered,
    ):
        return ScopeDecision(False, "OCR_DEPENDENT", VISUAL_OUT_OF_SCOPE)
    if re.search(
        r"\b(?:what|which)\s+(?:color|colour)\b|\bwhat (?:is|are) (?:visible|shown)\b|"
        r"\b(?:wearing|look like|on screen|in the image|in the frame|in the diagram|"
        r"visual object)\b",
        lowered,
    ):
        return ScopeDecision(False, "VISUAL_ONLY", VISUAL_OUT_OF_SCOPE)
    return ScopeDecision(True, "SUPPORTED_CORE")


def analyze_question(question: str) -> QuestionConstraints:
    """Extract only high-value constraints used by the safety contract."""
    lowered = question.lower()
    count_match = re.search(
        r"\b(?:what|which|name|list|give|identify)\s+(?:(\d+)|("
        + "|".join(_NUMBER_WORDS)
        + r"))\b",
        lowered,
    )
    requested_count = None
    if count_match:
        requested_count = (
            int(count_match.group(1))
            if count_match.group(1)
            else _NUMBER_WORDS[count_match.group(2)]
        )
    temporal_term = next(
        (
            word
            for word in (
                "prior to",
                "before",
                "after",
                "following",
                "then",
                "next",
                "previously",
                "later",
            )
            if re.search(rf"\b{re.escape(word)}\b", lowered)
        ),
        None,
    )
    temporal_relation = None
    temporal_anchor = None
    if temporal_term:
        temporal_relation = (
            "BEFORE" if temporal_term in {"before", "previously", "prior to"} else "AFTER"
        )
        anchor_match = re.search(rf"\b{re.escape(temporal_term)}\b\s+(.+?)(?:\?|$)", lowered)
        if anchor_match:
            temporal_anchor = anchor_match.group(1).strip(" .")
    relations = []
    if re.search(r"\b(?:why|cause|causes|reason|reasons)\b", lowered):
        relations.append("cause_or_reason")
    if re.search(r"\b(?:compare|comparison|difference|differ)\b", lowered):
        relations.append("comparison")
    if re.search(r"\b(?:define|definition|what (?:is|are|does))\b", lowered):
        relations.append("definition_or_fact")
    if requested_count is not None or re.search(r"\b(?:list|approaches|reasons|steps)\b", lowered):
        relations.append("list")
    if temporal_relation:
        relations.append("temporal")
    return QuestionConstraints(
        requested_count, temporal_relation, temporal_anchor, tuple(relations)
    )


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


def _within_budget(selected: list[EvidenceChunk], candidate: EvidenceChunk) -> bool:
    if candidate.evidence_id in {item.evidence_id for item in selected}:
        return False
    if len(selected) >= MAX_STRUCTURED_EVIDENCE:
        return False
    return sum(len(item.text) for item in [*selected, candidate]) <= MAX_STRUCTURED_CHARACTERS


def _with_role(chunk: EvidenceChunk, role: str) -> EvidenceChunk:
    return replace(chunk, role=role)


def _segment_evidence(video_id: str, segment, role: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=f"T{int(segment.id):04d}",
        video_id=video_id,
        start_seconds=round(float(segment.start), 3),
        end_seconds=round(float(segment.end), 3),
        text=segment.text.strip(),
        segment_ids=(int(segment.id),),
        role=role,
    )


_ANCHOR_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "explaining",
    "in",
    "is",
    "it",
    "of",
    "says",
    "saying",
    "speaker",
    "that",
    "the",
    "to",
    "video",
}


def _normalized_terms(text: str) -> set[str]:
    terms = set()
    for term in re.findall(r"[a-z0-9]+", text.casefold()):
        if term in _ANCHOR_STOP_WORDS:
            continue
        if term.endswith("ies") and len(term) > 4:
            term = f"{term[:-3]}y"
        elif term.endswith("ing") and len(term) > 5:
            term = term[:-3]
        elif term.endswith("ed") and len(term) > 4:
            term = term[:-2]
        elif term.endswith("s") and len(term) > 3:
            term = term[:-1]
        terms.add(term)
    return terms


def _localize_temporal_anchor(anchor_text: str, segments: list, video_id: str):
    """Choose the smallest adjacent segment window with strongest anchor coverage."""
    candidates = []
    for index, segment in enumerate(segments):
        candidates.append(_segment_evidence(video_id, segment, "TEMPORAL_ANCHOR"))
        if index + 1 < len(segments):
            following = segments[index + 1]
            candidates.append(
                EvidenceChunk(
                    evidence_id=f"T{int(segment.id):04d}-{int(following.id):04d}",
                    video_id=video_id,
                    start_seconds=round(float(segment.start), 3),
                    end_seconds=round(float(following.end), 3),
                    text=f"{segment.text.strip()} {following.text.strip()}",
                    segment_ids=(int(segment.id), int(following.id)),
                    role="TEMPORAL_ANCHOR",
                )
            )
    query_terms = _normalized_terms(anchor_text)
    lexical_scores = bm25(anchor_text, [item.text for item in candidates])
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (
            -len(query_terms & _normalized_terms(pair[1].text)),
            -lexical_scores[pair[0]],
            len(pair[1].segment_ids),
            pair[1].start_seconds,
        ),
    )
    return ranked[0][1] if ranked and lexical_scores[ranked[0][0]] > 0 else None


def expand_structured_evidence(
    question: str,
    chunks: list[EvidenceChunk],
    base_evidence: list[EvidenceChunk],
    *,
    segments: list | None = None,
    max_list_additions: int = MAX_ADDITIONAL_EVIDENCE,
    temporal_neighbor_count: int = 2,
) -> EvidenceSelection:
    """Conditionally add a small, ordered neighborhood for structured questions."""
    started = time.perf_counter()
    constraints = analyze_question(question)
    selected = list(base_evidence)
    original_ids = {item.evidence_id for item in selected}
    index_by_id = {item.evidence_id: index for index, item in enumerate(chunks)}

    if constraints.temporal_relation and constraints.temporal_anchor:
        temporal_units = chunks
        anchor = None
        if segments and chunks:
            temporal_units = [
                _segment_evidence(chunks[0].video_id, segment, "RELEVANT") for segment in segments
            ]
            anchor = _localize_temporal_anchor(
                constraints.temporal_anchor, segments, chunks[0].video_id
            )
        else:
            anchors = retrieve_evidence(constraints.temporal_anchor, temporal_units, k=3)
            anchor = anchors[0] if anchors else None
        if anchor:
            if segments:
                positions = {int(segment.id): index for index, segment in enumerate(segments)}
                anchor_index = (
                    positions[anchor.segment_ids[-1]]
                    if constraints.temporal_relation == "AFTER"
                    else positions[anchor.segment_ids[0]]
                )
            else:
                temporal_index_by_id = {
                    item.evidence_id: index for index, item in enumerate(temporal_units)
                }
                anchor_index = temporal_index_by_id[anchor.evidence_id]
            selected = [
                _with_role(item, "TEMPORAL_ANCHOR")
                if item.evidence_id == anchor.evidence_id
                else item
                for item in selected
            ]
            if anchor.evidence_id not in {item.evidence_id for item in selected} and _within_budget(
                selected, anchor
            ):
                selected.append(_with_role(anchor, "TEMPORAL_ANCHOR"))
            direction = 1 if constraints.temporal_relation == "AFTER" else -1
            for distance in range(1, temporal_neighbor_count + 1):
                target_index = anchor_index + direction * distance
                if not 0 <= target_index < len(temporal_units):
                    continue
                target = temporal_units[target_index]
                span = (
                    target.end_seconds - anchor.start_seconds
                    if direction == 1
                    else anchor.end_seconds - target.start_seconds
                )
                if span > MAX_EXPANSION_SECONDS:
                    continue
                replacement = _with_role(target, "TEMPORAL_TARGET")
                found = next(
                    (
                        index
                        for index, item in enumerate(selected)
                        if item.evidence_id == target.evidence_id
                    ),
                    None,
                )
                if found is not None:
                    selected[found] = replacement
                elif _within_budget(selected, replacement):
                    selected.append(replacement)
    elif constraints.requested_count is not None or "list" in constraints.relation_types:
        candidates = []
        for item in base_evidence:
            index = index_by_id[item.evidence_id]
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(chunks):
                    neighbor = chunks[neighbor_index]
                    span = max(
                        abs(neighbor.start_seconds - item.start_seconds),
                        abs(neighbor.end_seconds - item.end_seconds),
                    )
                    if span <= MAX_EXPANSION_SECONDS:
                        candidates.append(_with_role(neighbor, "LIST_NEIGHBOR"))
        for candidate in candidates:
            if len({item.evidence_id for item in selected} - original_ids) >= max_list_additions:
                break
            if _within_budget(selected, candidate):
                selected.append(candidate)

    deduplicated = list(dict((item.evidence_id, item) for item in selected).values())
    added = [item for item in deduplicated if item.evidence_id not in original_ids]
    temporal_evidence = [
        item for item in deduplicated if item.role in {"TEMPORAL_ANCHOR", "TEMPORAL_TARGET"}
    ]
    span = (
        max(item.end_seconds for item in temporal_evidence)
        - min(item.start_seconds for item in temporal_evidence)
        if constraints.temporal_relation and temporal_evidence
        else 0.0
    )
    return EvidenceSelection(
        base=tuple(base_evidence),
        expanded=tuple(deduplicated),
        added_count=len(added),
        added_characters=sum(len(item.text) for item in added),
        temporal_span_seconds=span,
        expansion_ms=(time.perf_counter() - started) * 1000,
    )


class OpenAIAnswerGenerator(AnswerGenerator):
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openai_api_key

    def _payload(self, question: str, evidence: list[EvidenceChunk]) -> dict:
        evidence_text = "\n\n".join(f"{item.evidence_id}: {item.text}" for item in evidence)
        constraints = analyze_question(question)
        constraint_text = json.dumps(asdict(constraints), separators=(",", ":"))
        claim_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
            },
            "required": ["text", "evidence_ids"],
            "additionalProperties": False,
        }
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
                "claims": {"type": "array", "items": claim_schema, "maxItems": 8},
                "unsupported_or_missing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
            },
            "required": [
                "answerable",
                "answer",
                "evidence_ids",
                "claims",
                "unsupported_or_missing",
            ],
            "additionalProperties": False,
        }
        return {
            "model": settings.qa_model,
            "store": False,
            "instructions": QA_INSTRUCTIONS,
            "input": (
                f"Question: {question}\n"
                f"Deterministic question constraints: {constraint_text}\n\n"
                f"Video evidence:\n{evidence_text}"
            ),
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


def resolve_answer(
    generated: GeneratedAnswer, evidence: list[EvidenceChunk], question: str = ""
) -> dict:
    lookup = {item.evidence_id: item for item in evidence}
    claim_ids = [item for claim in generated.claims for item in claim.evidence_ids]
    unknown = [item for item in [*generated.evidence_ids, *claim_ids] if item not in lookup]
    if unknown:
        raise HTTPException(502, "The answer provider returned an invalid evidence citation.")
    if not generated.answerable:
        return {"answerable": False, "answer": ABSTENTION, "citations": []}
    citation_ids = list(dict.fromkeys(claim_ids))
    constraints = analyze_question(question)
    contract_invalid = (
        not citation_ids
        or not generated.answer.strip()
        or not generated.claims
        or bool(generated.unsupported_or_missing)
        or set(generated.evidence_ids) != set(citation_ids)
        or (
            constraints.requested_count is not None
            and len(generated.claims) != constraints.requested_count
        )
        or len({claim.text.strip().casefold() for claim in generated.claims})
        != len(generated.claims)
    )
    if constraints.temporal_relation or generated.temporal_anchor_ids or generated.temporal_target_ids:
        contract_invalid = True
    if contract_invalid:
        return {"answerable": False, "answer": ABSTENTION, "citations": []}
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
        "scope": "spoken_facts_explanations_and_localized_summaries",
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
    scope = classify_question_scope(question)
    if not scope.supported:
        return {
            "answerable": False,
            "answer": scope.response,
            "citations": [],
            "evidence": [],
            "scope": asdict(scope),
            "usage": {"provider": "none", "model": None, "input_tokens": 0, "output_tokens": 0},
            "latency_ms": {
                "retrieval": 0,
                "generation": 0,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
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
    chunks = chunk_segments(folder.name, segments)
    base_evidence = retrieve_evidence(question, chunks)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    evidence = base_evidence
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
    answer = resolve_answer(generated, evidence, question)
    return {
        **answer,
        "evidence": [asdict(item) for item in evidence],
        "usage": usage,
        "latency_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": (time.perf_counter() - started) * 1000,
        },
        "scope": asdict(scope),
    }
