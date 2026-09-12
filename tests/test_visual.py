import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.encoder import normalize, rank_vectors
from app.main import app
from app.video import save_manifest

VIDEO_ID = "00000000-0000-0000-0000-000000000002"


def test_normalization_and_exact_ranking():
    pytest.importorskip("faiss")
    scores, indices = rank_vectors(np.array([[0, 4], [3, 0]]), np.array([[1, 0]]), 10)
    assert indices == [1, 0]
    assert scores == pytest.approx([1, 0])
    with pytest.raises(ValueError):
        normalize(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        normalize(np.array([[float("nan"), 1]]))


def test_index_search_and_model_mismatch(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    for module in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    folder = tmp_path / VIDEO_ID
    folder.mkdir()
    save_manifest(
        folder,
        {
            "status": "ready",
            "frames": [
                {"timestamp": 0, "thumbnail": f"/videos/{VIDEO_ID}/frames/000001.jpg"},
                {"timestamp": 5, "thumbnail": f"/videos/{VIDEO_ID}/frames/000002.jpg"},
            ],
        },
    )
    fake = SimpleNamespace(
        images=lambda paths: np.array([[0, 1], [1, 0]], dtype="float32"),
        text=lambda query: np.array([[1, 0]], dtype="float32"),
    )
    monkeypatch.setattr("app.visual.encoder", lambda: fake)
    client = TestClient(app)
    base = f"/videos/{VIDEO_ID}"
    assert client.get(base + "/search?q=car").status_code == 409
    assert client.post(base + "/index").status_code == 202
    assert client.get(base + "/index").json()["status"] == "ready"
    result = client.get(base + "/search?q=car&k=1").json()["results"][0]
    assert result["timestamp"] == 5
    assert result["score"] == pytest.approx(1)
    assert client.get(base + "/search?q=%20").status_code == 422
    monkeypatch.setattr(settings, "visual_revision", "changed")
    assert client.get(base + "/search?q=car").status_code == 409
