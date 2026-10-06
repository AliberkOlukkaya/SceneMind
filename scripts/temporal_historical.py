"""Diagnostic replay after the immutable new-validation decision, never tuning."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.temporal_protocol import (  # noqa: E402
    PROTECTED_MANIFEST,
    PROTECTED_SHA,
    verify_freeze,
)
from ml.evaluation.temporal_retrieval import aggregate, sha256  # noqa: E402
from scripts.temporal_benchmark import Peak, candidate, persist, summarize  # noqa: E402

DECISION_SHA = '83616a9b2951f6942d6c5a6811abb383efac525f5c7fc6357bfa93692df07918'
VALIDATION_SHA = '54f006a26f23e50a4e213182ab3b6bb1402baa6da5e1457c187ff74c6b7f462e'


def select_diagnostic(manifest):
    """Bound cost by duration before replay: all visual sources at most six minutes."""
    sources = [s for s in manifest['sources'] if s['duration_seconds'] <= 360]
    identifiers = {s['source_id'] for s in sources}
    queries = [q for q in manifest['queries']
               if q['mode'] == 'visual' and q['video_id'] in identifiers]
    used = {q['video_id'] for q in queries}
    return {'sources': [s for s in sources if s['source_id'] in used], 'queries': queries}


def main():
    decision_path = ROOT / 'ml/evaluation/reports/temporal-validation-decision.json'
    if sha256(decision_path) != DECISION_SHA:
        raise ValueError('Prior frozen decision missing or changed')
    decision = json.loads(decision_path.read_text(encoding='utf8'))
    if decision['decision'] != 'C' or decision['historical_replay_started']:
        raise ValueError('Unexpected decision contract')
    validation = verify_freeze(
        ROOT, ROOT / 'ml/evaluation/temporal_video_retrieval_v1_validation_manifest.json',
        VALIDATION_SHA)
    if sha256(ROOT / PROTECTED_MANIFEST) != PROTECTED_SHA:
        raise ValueError('Historical truth changed')
    historical = json.loads((ROOT / PROTECTED_MANIFEST).read_text(encoding='utf8'))
    manifest = select_diagnostic(historical)
    manifest['sources'] = [dict(s, media_path=f"data/final-deployment-v1/media/{s['source_id']}.webm")
                           for s in manifest['sources']]
    for source in manifest['sources']:
        if sha256(ROOT / source['media_path']) != source['media_sha256']:
            raise ValueError('Historical media bytes changed')
    old_path = ROOT / 'ml/evaluation/reports/final-deployment-acceptance-v1.json'
    old = json.loads(old_path.read_text(encoding='utf8'))
    if old['manifest_sha256'] != PROTECTED_SHA:
        raise ValueError('Historical baseline truth mismatch')
    selected = {q['id'] for q in manifest['queries']}
    baseline = [q for q in old['queries'] if q['id'] in selected]
    if len(baseline) != len(selected):
        raise ValueError('Historical baseline queries missing')
    work = ROOT / 'data/temporal-v1/historical'
    work.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', diagnostic_only=True, arm='candidate',
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  prior_decision_sha256=DECISION_SHA, prior_decision_commit='6796df1',
                  manifest_sha256=PROTECTED_SHA, baseline_report_sha256=sha256(old_path),
                  temporal_config=validation['selected_config'],
                  selection_rule='All visual sources with duration <= 360 seconds; fixed before replay',
                  excluded_long_sources=['social-media-talk', 'parliament-lecture'],
                  selected_manifest=manifest, videos={}, queries=[],
                  paired_baseline=baseline,
                  paired_baseline_metrics=aggregate([q['first_relevant_rank'] for q in baseline]),
                  historical_full_visual_r5=old['summary']['mode:visual']['r5'])
    with (work / 'started.json').open('x', encoding='utf8') as marker:
        json.dump(report, marker, indent=2)

    def save():
        persist(work / 'run.json', report)

    save()
    with Peak() as peak:
        candidate(manifest, work, report, save)
    report.update(status='complete', metrics=summarize(report['queries']),
                  process_tree_peak_rss_bytes=peak.bytes,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(report['metrics']['overall'], indent=2))


if __name__ == '__main__':
    main()
