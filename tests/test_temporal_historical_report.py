from copy import deepcopy

import pytest

from scripts.temporal_historical_report import DECISION_SHA, summarize


def fixture():
    return {
        'status': 'complete',
        'prior_decision_sha256': DECISION_SHA,
        'prior_decision_commit': 'frozen',
        'manifest_sha256': 'original',
        'baseline_report_sha256': 'original_report',
        'selection_rule': 'short visual sources',
        'excluded_long_sources': [],
        'historical_full_visual_r5': 0.5,
        'paired_baseline_metrics': {'n': 1, 'r5': 1.0, 'r10': 1.0},
        'metrics': {'overall': {'n': 1, 'r5': 0.0}},
        'selected_manifest': {'sources': [{'source_id': 'short', 'duration_seconds': 60}]},
        'videos': {'short': {'temporal_clips': 29, 'indexing_seconds': 20}},
        'paired_baseline': [{'id': 'old1', 'video_id': 'short', 'mode': 'visual',
                             'first_relevant_rank': 3}],
        'queries': [{'id': 'old1', 'video_id': 'short', 'query': 'find moment',
                     'intervals': [[10, 12]], 'first_rank': None,
                     'results': [{'timestamp': 20}]}],
    }


def test_historical_report_does_not_infer_v1_recall_at_ten():
    original = fixture()
    copy = deepcopy(original)
    summary = summarize(copy, {})
    assert summary['paired_subset_baseline']['r5'] == 1
    assert summary['paired_subset_baseline']['r10'] is None
    assert summary['paired_r5_delta_percentage_points'] == -100
    assert summary['queries'][0]['review_sheet_sha256'] is None
    assert copy == original


def test_historical_report_rejects_changed_pair_or_early_replay():
    changed = fixture()
    changed['queries'][0]['video_id'] = 'other'
    with pytest.raises(ValueError, match='pair mismatch'):
        summarize(changed, {})
    changed = fixture()
    changed['prior_decision_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='preceded the decision'):
        summarize(changed, {})
