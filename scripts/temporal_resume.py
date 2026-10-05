"""Recover only wholly unqueried sources; preserve an interrupted attempt.

This does not relax the original manifest, model, code or dependency freeze.
No completed query or indexed source is rerun. Unpersisted ingestion work is
lost and explicitly excluded from measured completed-stage resource totals.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.temporal_protocol import verify_freeze  # noqa: E402
from ml.evaluation.temporal_retrieval import sha256  # noqa: E402
from scripts.temporal_benchmark import Peak, candidate, persist, summarize  # noqa: E402


def remaining_manifest(manifest, previous):
    if previous['status'] != 'running' or previous['arm'] != 'candidate':
        raise ValueError('Only interrupted running candidate records are recoverable')
    if previous['temporal_config'] != manifest['selected_config']:
        raise ValueError('Configuration changed')
    completed = set(previous['videos'])
    expected = [q for q in manifest['queries'] if q['video_id'] in completed]
    rows = previous['queries']
    if len(rows) != len(expected):
        raise ValueError('Partially queried source cannot be recovered by this tool')
    for truth, row in zip(expected, rows):
        if any(row.get(key) != value for key, value in truth.items()):
            raise ValueError('Completed query truth changed')
    sources = [s for s in manifest['sources'] if s['source_id'] not in completed]
    if not sources or completed - {s['source_id'] for s in manifest['sources']}:
        raise ValueError('No valid remaining sources')
    return {**manifest, 'sources': sources,
            'queries': [q for q in manifest['queries'] if q['video_id'] not in completed]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    manifest = verify_freeze(ROOT, args.manifest, args.sha256)
    work = ROOT / 'data/temporal-v1/validation/candidate-4'
    output = work / 'run.json'
    previous = json.loads(output.read_text(encoding='utf8'))
    if previous['manifest_sha256'] != args.sha256:
        raise ValueError('Different original manifest')
    remaining = remaining_manifest(manifest, previous)
    for source in remaining['sources']:
        if (work / f"{source['source_id']}.npz").exists():
            raise ValueError('Unqueried existing index requires explicit investigation')
    plan = {'original_run_sha256': sha256(output), 'manifest_sha256': args.sha256,
            'recovery_code_sha256': sha256(__file__),
            'completed_queries_preserved': len(previous['queries']),
            'sources_to_finish': [s['source_id'] for s in remaining['sources']],
            'reason': 'Original process absent after workstation/session interruption; exact exit cause unknown',
            'lost_unpersisted_ingestion_cost_seconds': None}
    with (work / 'resume-started.json').open('x', encoding='utf8') as marker:
        json.dump(plan, marker, indent=2)
    with (work / 'interrupted-run.json').open('xb') as snapshot:
        snapshot.write(output.read_bytes())
    report = deepcopy(previous)
    report['recovery'] = plan
    initial_load = previous['model_load_seconds']

    def save():
        persist(output, report)

    try:
        with Peak() as peak:
            candidate(remaining, work, report, save)
        report['recovery']['additional_model_load_seconds'] = report['model_load_seconds']
        report['model_load_seconds'] = initial_load
        report['recovery']['resumed_process_tree_peak_rss_bytes'] = peak.bytes
        report.update(status='complete', process_tree_peak_rss_bytes=None,
                      metrics=summarize(report['queries']))
        assert report['queries'][:len(previous['queries'])] == previous['queries']
        values = [q['latency_seconds'] for q in report['queries']]
        report['query_latency_seconds'] = {'median': float(np.median(values)),
                                           'p95': float(np.percentile(values, 95))}
        save()
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        save()
        raise
    print(json.dumps(report['metrics']['overall'], indent=2))


if __name__ == '__main__':
    main()
