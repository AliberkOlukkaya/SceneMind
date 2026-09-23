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
    EvidenceChunk,
    GeneratedAnswer,
    OpenAIAnswerGenerator,
    chunk_segments,
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


def test_unknown_evidence_id_is_rejected():
    with pytest.raises(HTTPException, match="invalid evidence"):
        resolve_answer(
            GeneratedAnswer(answerable=True, answer="Claim", evidence_ids=["E999"]), _evidence()
        )


def test_answerable_response_requires_a_citation():
    with pytest.raises(HTTPException, match="ungrounded"):
        resolve_answer(
            GeneratedAnswer(answerable=True, answer="Claim", evidence_ids=[]), _evidence()
        )


def test_abstention_discards_provider_text_and_citations():
    result = resolve_answer(
        GeneratedAnswer(answerable=False, answer="Maybe outside knowledge", evidence_ids=["E001"]),
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
    assert "E001: RAG grounds answers." in payload["input"]
    assert VIDEO_ID not in payload["input"]
    assert "start_seconds" not in payload["input"]


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
