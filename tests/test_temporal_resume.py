from copy import deepcopy

import pytest

from scripts.temporal_resume import remaining_manifest


def fixture():
    queries = [dict(id='a', video_id='a', intervals=[[1, 2]]),
               dict(id='b', video_id='b', intervals=[[3, 4]])]
    manifest = dict(sources=[{'source_id': 'a'}, {'source_id': 'b'}],
                    queries=queries, selected_config={'length': 4})
    run = dict(status='running', arm='candidate', videos={'a': {}},
               queries=[{**queries[0], 'results': []}], temporal_config={'length': 4})
    return manifest, run


def test_recovery_excludes_completed_queries_without_mutation():
    manifest, run = fixture()
    original = deepcopy(run)
    remaining = remaining_manifest(manifest, run)
    assert remaining['sources'] == [{'source_id': 'b'}]
    assert [q['id'] for q in remaining['queries']] == ['b']
    assert run == original


@pytest.mark.parametrize('problem', ['complete', 'partial', 'truth', 'config'])
def test_recovery_refuses_reexecution_or_changed_contract(problem):
    manifest, run = fixture()
    if problem == 'complete':
        run['status'] = 'complete'
    elif problem == 'partial':
        run['queries'] = []
    elif problem == 'truth':
        run['queries'][0]['intervals'] = [[0, 10]]
    else:
        run['temporal_config']['length'] = 8
    with pytest.raises(ValueError):
        remaining_manifest(manifest, run)
