import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Segment, Transcript, engine, migrate
from app.main import app
from app.qa import (
    ABSTENTION,
    MAX_ADDITIONAL_EVIDENCE,
    MAX_STRUCTURED_CHARACTERS,
    QA_INSTRUCTIONS,
    QA_STRUCTURED_INSTRUCTIONS,
    EvidenceChunk,
    GeneratedAnswer,
    GeneratedClaim,
    OpenAIAnswerGenerator,
    analyze_question,
    chunk_segments,
    expand_structured_evidence,
    resolve_answer,
    retrieve_evidence,
)
from app.video import save_manifest

VIDEO_ID = "00000000-0000-0000-0000-000000000051"
ROOT = Path(__file__).parents[1]


@pytest.fixture
def qa_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'qa.db'}")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "qa_enabled", True)
    migrate()
    folder = settings.data_dir / VIDEO_ID
    folder.mkdir(parents=True)
    save_manifest(
        folder,
        {
            "id": VIDEO_ID,
            "filename": "lecture.mp4",
            "source": "source.mp4",
            "source_type": "upload",
            "status": "ready",
            "frames": [],
        },
    )
    with Session(engine()) as session, session.begin():
        session.add(Transcript(video_id=VIDEO_ID, status="ready", model="tiny", language="en"))
        session.add_all(
            [
                Segment(
                    video_id=VIDEO_ID, start=10, end=14, text="RAG retrieves relevant documents."
                ),
                Segment(
                    video_id=VIDEO_ID, start=14, end=19, text="This improves factual grounding."
                ),
                Segment(video_id=VIDEO_ID, start=70, end=75, text="Caching reduces repeated work."),
            ]
        )
    return TestClient(app)


def _segments():
    return [
        SimpleNamespace(id=1, start=0.0, end=8.0, text="First explanation."),
        SimpleNamespace(id=2, start=8.0, end=16.0, text="Second explanation."),
        SimpleNamespace(id=3, start=40.0, end=50.0, text="Third explanation."),
    ]


def _evidence():
    return [EvidenceChunk("E001", VIDEO_ID, 10, 19, "RAG grounds answers.", (1, 2))]


def _generated(
    *,
    answerable=True,
    answer="Claim",
    evidence_ids=None,
    claims=None,
    missing=None,
    anchor_ids=None,
    target_ids=None,
):
    evidence_ids = ["E001"] if evidence_ids is None else evidence_ids
    claims = [GeneratedClaim(text="Claim", evidence_ids=evidence_ids)] if claims is None else claims
    return GeneratedAnswer(
        answerable=answerable,
        answer=answer,
        evidence_ids=evidence_ids,
        claims=claims,
        unsupported_or_missing=[] if missing is None else missing,
        temporal_anchor_ids=[] if anchor_ids is None else anchor_ids,
        temporal_target_ids=[] if target_ids is None else target_ids,
    )


def test_chunking_is_deterministic_and_preserves_timestamps_ids_and_overlap():
    first = chunk_segments(VIDEO_ID, _segments(), max_seconds=20, max_characters=100)
    second = chunk_segments(VIDEO_ID, _segments(), max_seconds=20, max_characters=100)
    assert first == second
    assert first[0] == EvidenceChunk(
        "E001", VIDEO_ID, 0.0, 16.0, "First explanation. Second explanation.", (1, 2)
    )
    assert first[1].segment_ids[0] == 2
    assert first[1].start_seconds == 8.0
    assert first[1].end_seconds == 16.0


def test_evidence_retrieval_is_bm25_ranked_and_bounded():
    chunks = [
        EvidenceChunk("E001", VIDEO_ID, 0, 5, "unrelated introduction", (1,)),
        EvidenceChunk("E002", VIDEO_ID, 5, 10, "retrieval grounding documents", (2,)),
    ]
    assert [item.evidence_id for item in retrieve_evidence("retrieval documents", chunks, 1)] == [
        "E002"
    ]
    assert retrieve_evidence("zebra", chunks) == []


def test_frozen_manifest_checksum_split_and_question_counts():
    path = ROOT / "ml/evaluation/grounded_video_qa_v1_manifest.json"
    expected = (
        (ROOT / "ml/evaluation/grounded_video_qa_v1_manifest.sha256")
        .read_text(encoding="ascii")
        .split()[0]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    manifest = json.loads(path.read_text(encoding="utf-8"))
    development = manifest["development"]
    validation = manifest["validation"]
    assert development["source"]["sha256"] != validation["source"]["sha256"]
    assert len(development["questions"]) == len(validation["questions"]) == 12
    assert sum(row["answerable"] for row in validation["questions"]) == 9


def test_frozen_machine_report_records_failed_abstention_gate():
    report = json.loads(
        (ROOT / "ml/evaluation/reports/grounded-video-qa-v1.json").read_text(encoding="utf-8")
    )
    assert (
        report["manifest_sha256"]
        == "ac103e3318f9f46664867c6619daf60d65d99cd5447db236e6efd6b29cadb3a4"
    )
    assert report["validation_run_count"] == 1
    assert report["metrics"]["evidence_recall_at_5"] == 1
    assert report["metrics"]["false_answer_rate"] == pytest.approx(1 / 3)
    assert report["decision"]["code"] == "D"


def test_abstention_safety_sources_and_questions_are_disjoint_and_balanced():
    path = ROOT / "ml/evaluation/qa_abstention_safety_v1_manifest.json"
    expected = (
        (ROOT / "ml/evaluation/qa_abstention_safety_v1_manifest.sha256")
        .read_text(encoding="ascii")
        .split()[0]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(QA_INSTRUCTIONS.encode()).hexdigest()
        == manifest["frozen_configuration"]["prompt_sha256"]
    )
    historical = json.loads(
        (ROOT / "ml/evaluation/grounded_video_qa_v1_manifest.json").read_text(encoding="utf-8")
    )
    development = manifest["development"]
    validation = manifest["validation"]
    development_hashes = {source["sha256"] for source in development["sources"]}
    validation_hashes = {source["sha256"] for source in validation["sources"]}
    historical_hashes = {
        historical["development"]["source"]["sha256"],
        historical["validation"]["source"]["sha256"],
    }
    assert len(development["sources"]) == len(validation["sources"]) == 2
    assert development_hashes.isdisjoint(validation_hashes | historical_hashes)
    assert validation_hashes.isdisjoint(historical_hashes)
    for split in (development, validation):
        assert len(split["questions"]) == 30
        assert sum(row["answerable"] for row in split["questions"]) == 18
        assert sum(row["category"] == "HARD_NEGATIVE" for row in split["questions"]) == 10
        assert all(row["expected_intervals"] for row in split["questions"] if row["answerable"])
        assert all(
            not row["expected_intervals"] for row in split["questions"] if not row["answerable"]
        )


def test_abstention_safety_frozen_report_records_decision_d():
    report = json.loads(
        (ROOT / "ml/evaluation/reports/qa-abstention-safety-v1.json").read_text(encoding="utf-8")
    )
    assert (
        report["manifest_sha256"]
        == "26646d2bf977981950e68888898522e348e6019cc39acac417a8a28acca34255"
    )
    assert report["validation_run_count"] == 1
    assert report["metrics"]["correct_abstention_rate"] == 1
    assert report["metrics"]["false_answer_rate"] == 0
    assert report["metrics"]["answer_correctness"] == pytest.approx(14 / 18)
    assert report["categories"]["LIST_COUNT"]["answer_correctness"] == 0
    assert report["decision"]["code"] == "D"
    assert report["decision"]["ask_video_eligible_for_enablement"] is False
    assert all(row["safe_abstention"] for row in report["historical_replay"]["rows"])


def test_unknown_evidence_id_is_rejected():
    with pytest.raises(HTTPException, match="invalid evidence"):
        resolve_answer(_generated(evidence_ids=["E999"]), _evidence())


def test_answerable_response_without_a_citation_fails_safely():
    result = resolve_answer(_generated(evidence_ids=[], claims=[]), _evidence())
    assert result == {"answerable": False, "answer": ABSTENTION, "citations": []}


def test_abstention_discards_provider_text_and_citations():
    result = resolve_answer(
        _generated(answerable=False, answer="Maybe outside knowledge"),
        _evidence(),
    )
    assert result == {"answerable": False, "answer": ABSTENTION, "citations": []}


def test_empty_and_oversized_questions_are_rejected(qa_client):
    assert qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "   "}).status_code == 422
    assert (
        qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "x" * 501}).status_code == 422
    )


def test_video_must_be_ready(qa_client):
    folder = settings.data_dir / VIDEO_ID
    record = {"id": VIDEO_ID, "status": "processing", "frames": []}
    save_manifest(folder, record)
    response = qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "What is RAG?"})
    assert response.status_code == 409


def test_ready_transcript_is_required(qa_client):
    with Session(engine()) as session, session.begin():
        session.get(Transcript, VIDEO_ID).status = "processing"
    response = qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "What is RAG?"})
    assert response.status_code == 409


def test_nonempty_transcript_is_required(qa_client):
    with Session(engine()) as session, session.begin():
        session.query(Segment).filter(Segment.video_id == VIDEO_ID).delete()
    response = qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "What is RAG?"})
    assert response.status_code == 409


def test_provider_payload_sends_only_question_and_selected_evidence():
    payload = OpenAIAnswerGenerator("secret")._payload("Why retrieval?", _evidence())
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert "E001 [RELEVANT]: RAG grounds answers." in payload["input"]
    assert VIDEO_ID not in payload["input"]
    assert "start_seconds" not in payload["input"]
    assert payload["text"]["format"]["schema"]["required"] == [
        "answerable",
        "answer",
        "evidence_ids",
        "claims",
        "unsupported_or_missing",
        "temporal_anchor_ids",
        "temporal_target_ids",
    ]


def test_question_constraints_extract_count_temporal_and_relation():
    constraints = analyze_question("What three reasons are given after the comparison?")
    assert constraints.requested_count == 3
    assert constraints.temporal_relation == "AFTER"
    assert constraints.temporal_anchor == "the comparison"
    assert set(constraints.relation_types) == {"cause_or_reason", "comparison", "list", "temporal"}


def test_topically_related_but_insufficient_and_outside_knowledge_abstain():
    for missing in (["wrong entity"], ["requested fact is outside the evidence"]):
        result = resolve_answer(
            _generated(missing=missing), _evidence(), "What does RAG guarantee?"
        )
        assert result == {"answerable": False, "answer": ABSTENTION, "citations": []}


def test_wrong_entity_fails_safely():
    result = resolve_answer(
        _generated(missing=["Evidence discusses Wi-Fi, not the requested mobile-data entity."]),
        _evidence(),
        "What does the speaker say about mobile data?",
    )
    assert result["answerable"] is False


def test_before_after_mismatch_fails_safely():
    result = resolve_answer(
        _generated(missing=["The evidence is before the requested temporal anchor."]),
        _evidence(),
        "What happens after DNS resolution?",
    )
    assert result["answerable"] is False


def test_partial_or_wrong_list_count_fails_safely():
    result = resolve_answer(
        _generated(claims=[GeneratedClaim(text="One reason", evidence_ids=["E001"])]),
        _evidence(),
        "What two reasons are given?",
    )
    assert result["answerable"] is False


def _structured_chunks():
    return [
        EvidenceChunk("E001", VIDEO_ID, 0, 10, "Opening context and first point.", (1,)),
        EvidenceChunk("E002", VIDEO_ID, 10, 20, "DNS resolution is introduced here.", (2,)),
        EvidenceChunk("E003", VIDEO_ID, 21, 30, "The browser next checks its cache.", (3,)),
        EvidenceChunk("E004", VIDEO_ID, 31, 40, "Then it opens a network connection.", (4,)),
        EvidenceChunk("E005", VIDEO_ID, 41, 50, "Closing summary.", (5,)),
    ]


def test_list_evidence_expansion_is_bounded_deduplicated_and_local():
    chunks = _structured_chunks()
    base = [chunks[1], chunks[3]]
    selection = expand_structured_evidence("What three points are listed?", chunks, base)
    ids = [item.evidence_id for item in selection.expanded]
    assert len(ids) == len(set(ids))
    assert selection.added_count <= MAX_ADDITIONAL_EVIDENCE
    assert sum(len(item.text) for item in selection.expanded) <= MAX_STRUCTURED_CHARACTERS
    assert ids[:2] == ["E002", "E004"]


def test_ordinary_question_keeps_existing_evidence_path_unchanged():
    chunks = _structured_chunks()
    base = [chunks[2], chunks[0]]
    selection = expand_structured_evidence("How does the browser work?", chunks, base)
    assert selection.expanded == tuple(base)
    assert selection.added_count == 0


def test_after_relation_localizes_anchor_and_expands_forward_only():
    chunks = _structured_chunks()
    base = retrieve_evidence("What happens after DNS resolution?", chunks)
    selection = expand_structured_evidence("What happens after DNS resolution?", chunks, base)
    anchors = [item for item in selection.expanded if item.role == "TEMPORAL_ANCHOR"]
    targets = [item for item in selection.expanded if item.role == "TEMPORAL_TARGET"]
    assert [item.evidence_id for item in anchors] == ["E002"]
    assert targets
    assert all(item.start_seconds > anchors[0].start_seconds for item in targets)


def test_before_relation_localizes_anchor_and_expands_backward_only():
    chunks = _structured_chunks()
    base = retrieve_evidence("What happens before DNS resolution?", chunks)
    selection = expand_structured_evidence("What happens before DNS resolution?", chunks, base)
    anchors = [item for item in selection.expanded if item.role == "TEMPORAL_ANCHOR"]
    targets = [item for item in selection.expanded if item.role == "TEMPORAL_TARGET"]
    assert [item.evidence_id for item in anchors] == ["E002"]
    assert [item.evidence_id for item in targets] == ["E001"]


def test_temporal_expansion_uses_local_segments_for_repeated_anchor_mentions():
    segments = [
        SimpleNamespace(id=1, start=0, end=5, text="DNS resolution is introduced."),
        SimpleNamespace(id=2, start=5, end=10, text="The browser checks its cache."),
        SimpleNamespace(id=3, start=40, end=45, text="DNS resolution is introduced."),
        SimpleNamespace(id=4, start=45, end=50, text="A later recap follows."),
    ]
    chunks = chunk_segments(VIDEO_ID, segments)
    base = retrieve_evidence("What happens after DNS resolution is introduced?", chunks)
    selection = expand_structured_evidence(
        "What happens after DNS resolution is introduced?",
        chunks,
        base,
        segments=segments,
        temporal_neighbor_count=1,
    )
    anchors = [item for item in selection.expanded if item.role == "TEMPORAL_ANCHOR"]
    targets = [item for item in selection.expanded if item.role == "TEMPORAL_TARGET"]
    assert [item.evidence_id for item in anchors] == ["T0001"]
    assert [item.evidence_id for item in targets] == ["T0002"]
    assert selection.temporal_span_seconds == 10


def test_temporal_contract_rejects_wrong_direction_and_orders_citations():
    chunks = _structured_chunks()
    evidence = [
        EvidenceChunk(**{**chunks[1].__dict__, "role": "TEMPORAL_ANCHOR"}),
        EvidenceChunk(**{**chunks[2].__dict__, "role": "TEMPORAL_TARGET"}),
    ]
    claims = [
        GeneratedClaim(text="DNS is introduced.", evidence_ids=["E002"]),
        GeneratedClaim(text="The browser checks its cache next.", evidence_ids=["E003"]),
    ]
    valid = _generated(
        answer="The browser checks its cache.",
        evidence_ids=["E003"],
        claims=claims,
        anchor_ids=["E002"],
        target_ids=["E003"],
    )
    resolved = resolve_answer(valid, evidence, "What happens after DNS resolution?")
    assert [item["evidence_id"] for item in resolved["citations"]][:2] == ["E002", "E003"]
    assert (
        resolve_answer(valid, evidence, "What happens before DNS resolution?")["answerable"]
        is False
    )


def test_temporal_contract_rejects_ids_without_structured_roles():
    chunks = _structured_chunks()
    claims = [GeneratedClaim(text="The browser checks its cache.", evidence_ids=["E003"])]
    generated = _generated(
        answer="The browser checks its cache.",
        evidence_ids=["E003"],
        claims=claims,
        anchor_ids=["E002"],
        target_ids=["E003"],
    )
    assert (
        resolve_answer(generated, [chunks[1], chunks[2]], "What happens after DNS resolution?")[
            "answerable"
        ]
        is False
    )


def test_structured_qa_manifest_is_balanced_annotated_and_source_disjoint():
    manifest_path = ROOT / "ml/evaluation/qa_structured_question_evidence_v1_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_checksum = (
        (ROOT / "ml/evaluation/qa_structured_question_evidence_v1_manifest.sha256")
        .read_text(encoding="ascii")
        .split()[0]
    )
    assert hashlib.sha256(manifest_bytes).hexdigest() == expected_checksum
    assert (
        manifest["frozen_configuration"]["prompt_sha256"]
        == hashlib.sha256(QA_STRUCTURED_INSTRUCTIONS.encode()).hexdigest()
    )
    assert (
        manifest["frozen_configuration"]["implementation_sha256"]
        == hashlib.sha256((ROOT / "backend/app/qa.py").read_bytes()).hexdigest()
    )
    development_sources = {item["source_id"] for item in manifest["development"]["sources"]}
    validation_sources = {item["source_id"] for item in manifest["validation"]["sources"]}
    historical_sources = {
        "machine-learning-models-introduction",
        "how-the-internet-really-works",
        "oceans-explainer",
        "design-free-software-talk",
        "rewiring-video-editor-talk",
        "ui-frameworks-talk",
    }
    assert development_sources.isdisjoint(validation_sources)
    assert (development_sources | validation_sources).isdisjoint(historical_sources)
    for split in ("development", "validation"):
        questions = manifest[split]["questions"]
        counts = {
            category: sum(item["question_type"] == category for item in questions)
            for category in ("ORDINARY", "LIST_COUNT", "TEMPORAL", "HARD_NEGATIVE")
        }
        assert counts == {
            "ORDINARY": 10,
            "LIST_COUNT": 8,
            "TEMPORAL": 8,
            "HARD_NEGATIVE": 10,
        }
        for item in questions:
            assert item["source_id"] in {
                source["source_id"] for source in manifest[split]["sources"]
            }
            if item["question_type"] == "LIST_COUNT":
                assert item["requested_count"] is not None
            if item["question_type"] == "TEMPORAL":
                assert item["anchor_interval"] and item["target_interval"]


def test_claim_citations_must_match_answer_citations():
    generated = _generated(
        evidence_ids=["E001"], claims=[GeneratedClaim(text="Claim", evidence_ids=["E002"])]
    )
    evidence = [*_evidence(), EvidenceChunk("E002", VIDEO_ID, 20, 25, "Other", (3,))]
    assert resolve_answer(generated, evidence)["answerable"] is False


@pytest.mark.parametrize("source_type", ["upload", "url"])
def test_answer_and_citation_resolution_for_upload_and_url_video(
    qa_client, monkeypatch, source_type
):
    folder = settings.data_dir / VIDEO_ID
    record = {"id": VIDEO_ID, "status": "ready", "frames": [], "source_type": source_type}
    save_manifest(folder, record)

    class FakeGenerator:
        def generate(self, question, evidence):
            assert question == "Why use RAG?"
            assert evidence[0].start_seconds == 10
            return (
                GeneratedAnswer(
                    answerable=True,
                    answer="RAG retrieves documents to improve factual grounding [1].",
                    evidence_ids=[evidence[0].evidence_id],
                    claims=[
                        GeneratedClaim(
                            text="RAG retrieves documents to improve factual grounding.",
                            evidence_ids=[evidence[0].evidence_id],
                        )
                    ],
                    unsupported_or_missing=[],
                    temporal_anchor_ids=[],
                    temporal_target_ids=[],
                ),
                {"provider": "fake", "model": "test", "input_tokens": 20, "output_tokens": 10},
            )

    monkeypatch.setattr("app.qa.answer_generator", lambda: FakeGenerator())
    response = qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "Why use RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answerable"] is True
    assert body["citations"][0]["start_seconds"] == 10
    assert body["citations"][0]["end_seconds"] == 19
    assert body["citations"][0]["video_id"] == VIDEO_ID


def test_no_lexical_evidence_abstains_without_provider(qa_client, monkeypatch):
    monkeypatch.setattr(
        "app.qa.answer_generator", lambda: (_ for _ in ()).throw(AssertionError("must not call"))
    )
    response = qa_client.post(
        f"/videos/{VIDEO_ID}/ask", json={"question": "Who won the world cup?"}
    )
    assert response.status_code == 200
    assert response.json()["answerable"] is False
    assert response.json()["usage"]["provider"] == "none"


def test_missing_api_key_has_clear_configuration_state(qa_client, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    status = qa_client.get(f"/videos/{VIDEO_ID}/ask/status").json()
    assert status["configured"] is False
    response = qa_client.post(f"/videos/{VIDEO_ID}/ask", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_provider_timeout_is_bounded(monkeypatch):
    class TimeoutClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr("app.qa.httpx.Client", TimeoutClient)
    monkeypatch.setattr(settings, "qa_retries", 0)
    with pytest.raises(HTTPException) as caught:
        OpenAIAnswerGenerator("secret").generate("question", _evidence())
    assert caught.value.status_code == 504


def test_provider_malformed_output_is_rejected(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": "not-json", "usage": {}}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("app.qa.httpx.Client", Client)
    with pytest.raises(HTTPException) as caught:
        OpenAIAnswerGenerator("secret").generate("question", _evidence())
    assert caught.value.status_code == 502
