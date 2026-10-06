import json
from copy import deepcopy

import pytest

from scripts import temporal_historical as replay


def test_diagnostic_scope_preserves_frozen_queries_and_includes_required_sources():
    path = replay.ROOT / replay.PROTECTED_MANIFEST
    manifest = json.loads(path.read_text(encoding='utf8'))
    original = deepcopy(manifest)
    selected = replay.select_diagnostic(manifest)
    assert len(selected['sources']) == 6
    assert len(selected['queries']) == 42
    assert {'aikido-demonstration', 'manchester-street'} <= {
        s['source_id'] for s in selected['sources']}
    assert all(q in original['queries'] and q['mode'] == 'visual' for q in selected['queries'])
    assert manifest == original


def test_changed_decision_blocks_replay_before_inference(monkeypatch):
    monkeypatch.setattr(replay, 'sha256', lambda _: 'changed')
    monkeypatch.setattr(replay, 'candidate', lambda *_: pytest.fail('Inference must not start'))
    with pytest.raises(ValueError, match='Prior frozen decision'):
        replay.main()


def test_changed_historical_media_blocks_replay_before_inference(monkeypatch):
    real_sha = replay.sha256

    def changed_media(path):
        return 'changed' if path.suffix == '.webm' else real_sha(path)

    monkeypatch.setattr(replay, 'sha256', changed_media)
    monkeypatch.setattr(replay, 'verify_freeze', lambda *_: {})
    monkeypatch.setattr(replay, 'candidate', lambda *_: pytest.fail('Inference must not start'))
    with pytest.raises(ValueError, match='Historical media bytes'):
        replay.main()
