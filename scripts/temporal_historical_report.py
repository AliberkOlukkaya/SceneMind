"""Publish a diagnostic-only paired historical replay after Decision C."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.temporal_historical import DECISION_SHA, sha256  # noqa: E402

OUT = ROOT / 'ml/evaluation/reports/temporal-historical-diagnostic.json'


def summarize(replay, notes):
    if replay['status'] != 'complete' or replay['prior_decision_sha256'] != DECISION_SHA:
        raise ValueError('Historical replay is incomplete or preceded the decision')
    baseline = {q['id']: q for q in replay['paired_baseline']}
    rows = []
    for query in replay['queries']:
        qid = query['id']
        previous = baseline[qid]
        if previous['video_id'] != query['video_id'] or previous['mode'] != 'visual':
            raise ValueError('Historical pair mismatch')
        inspected = notes.get(qid)
        sheet = ROOT / f'data/temporal-v1/review/historical/{qid}-00.jpg'
        rows.append({
            'id': qid,
            'video_id': query['video_id'],
            'query': query['query'],
            'frozen_intervals': query['intervals'],
            'old_v1_first_rank': previous['first_relevant_rank'],
            'candidate_first_rank': query['first_rank'],
            'old_v1_top5_hit': previous['first_relevant_rank'] is not None
                                  and previous['first_relevant_rank'] <= 5,
            'candidate_top5_hit': query['first_rank'] is not None
                                  and query['first_rank'] <= 5,
            'candidate_results': query['results'],
            'review_observation': inspected,
            'review_sheet_sha256': hashlib.sha256(sheet.read_bytes()).hexdigest()
            if inspected else None,
        })
    if len(rows) != len(baseline) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate paired diagnostic')
    sources = {}
    for source in replay['selected_manifest']['sources']:
        sid = source['source_id']
        matched = [row for row in rows if row['video_id'] == sid]
        sources[sid] = {'n': len(matched), 'minutes': source['duration_seconds'] / 60,
                        'old_v1_r5': sum(q['old_v1_top5_hit'] for q in matched) / len(matched),
                        'candidate_r5': sum(q['candidate_top5_hit'] for q in matched) / len(matched),
                        'candidate_clips': replay['videos'][sid]['temporal_clips'],
                        'candidate_index_seconds': replay['videos'][sid]['indexing_seconds']}
    return {'status': 'diagnostic_complete_after_frozen_decision',
            'model_or_config_selected_from_historical': False,
            'decision_sha256': DECISION_SHA,
            'decision_commit': replay['prior_decision_commit'],
            'protected_manifest_sha256': replay['manifest_sha256'],
            'historical_baseline_report_sha256': replay['baseline_report_sha256'],
            'replay_raw_sha256': sha256(ROOT / 'data/temporal-v1/historical/run.json'),
            'selection_rule': replay['selection_rule'],
            'excluded_long_sources': replay['excluded_long_sources'],
            'n_sources': len(sources), 'n_queries': len(rows),
            'old_full_visual_r5_context_only': replay['historical_full_visual_r5'],
            'paired_subset_baseline': {**replay['paired_baseline_metrics'],
                                       'r10': None},
            'paired_subset_candidate': replay['metrics']['overall'],
            'paired_r5_delta_percentage_points': 100 * (
                replay['metrics']['overall']['r5'] - replay['paired_baseline_metrics']['r5']),
            'sources': sources, 'queries': rows,
            'limitations': [
                'Observed V1 labels are frozen and have incomplete alternative intervals for repeated moments.',
                'Six sources at most six minutes; no long-video historical candidate replay.',
                'Pixel review is selective, agent-only, and not continuous playback.',
                'Historical benchmark cannot influence model, configuration, thresholds, or promotion decision.',
                'Historical V1 report stored Top-5 only, so its Recall@10 is unavailable.',
            ]}


def main():
    decision = ROOT / 'ml/evaluation/reports/temporal-validation-decision.json'
    if sha256(decision) != DECISION_SHA:
        raise ValueError('Frozen decision changed')
    replay = json.loads((ROOT / 'data/temporal-v1/historical/run.json').read_text())
    notes = json.loads((ROOT / 'data/temporal-v1/historical_notes.json').read_text())
    diagnostic = summarize(replay, notes)
    OUT.write_text(json.dumps(diagnostic, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: value for key, value in diagnostic.items()
                      if key in {'n_sources', 'n_queries', 'paired_r5_delta_percentage_points',
                                 'paired_subset_baseline', 'paired_subset_candidate', 'sources'}},
                     indent=2))


if __name__ == '__main__':
    main()
