from fastapi.testclient import TestClient

from app.main import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "scenemind"}


def test_release_candidate_version():
    assert app.version == "1.0.0-rc1"
