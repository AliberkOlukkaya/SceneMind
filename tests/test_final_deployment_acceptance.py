"""Guard the one-shot final acceptance protocol without loading user media/models."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_MANIFEST.json"
FROZEN_SHA256 = "d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196"


def test_frozen_final_acceptance_contract():
    raw = MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256
    data = json.loads(raw)
    sources = {source["source_id"]: source for source in data["sources"]}
    assert len(sources) == 8
    assert len({source["media_sha256"] for source in sources.values()}) == 8
    queries = data["queries"]
    assert len(queries) == 124
    assert len({query["id"] for query in queries}) == len(queries)
    assert Counter(query["mode"] for query in queries) == {
        "speech": 44, "visual": 55, "hybrid": 25
    }
    assert all(count >= 8 for count in Counter(q["video_id"] for q in queries).values())
    assert {query["difficulty"] for query in queries} == {"easy", "medium", "hard"}
    for query in queries:
        duration = sources[query["video_id"]]["duration_seconds"]
        assert query["intervals"]
        assert all(0 <= start <= end <= duration for start, end in query["intervals"])
    for relative_path, checksum in data["frozen_code_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == checksum
