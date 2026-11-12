import json

from backend.app.artifacts import write_json, write_jsonl
from backend.app.config import Settings
from backend.tests.observability.fake_langfuse import FakeLangfuse
from evaluation import quality


def failed_item(tmp_path, question_id, code):
    saved = tmp_path/f'{question_id}.json'
    write_json(saved, {'trace_id': f'svc-{question_id}', 'error': {'code': code, 'reason': 'duplicate_citation'},
                       'candidates': [{'book_id': 'G1', 'chunk_id': 'c1'}]})
    return {'profile': 'bge', 'mode': 'hybrid', 'question_id': question_id, 'question': '질문',
            'status': 'error', 'result_path': str(saved)}


def test_failed_items_get_status_category_scores_on_the_service_trace(tmp_path, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    folder = tmp_path/'run-1'
    write_jsonl(folder/'runs.jsonl', [failed_item(tmp_path, 'q1', 'output_invalid'),
                                      failed_item(tmp_path, 'q2', 'pipeline_error')])
    monkeypatch.setattr(quality, 'reference_map', lambda s: {q: {'evidence_groups': [[{'book_id': 'G1'}]]} for q in ('q1', 'q2')})
    quality.evaluate_comparison(Settings(_env_file=None), folder)

    rows = {r['question_id']: r for r in map(json.loads, (folder/'scores.jsonl').read_text(encoding='utf-8').splitlines())}
    assert rows['q1']['judge_status'] == 'output_invalid' and rows['q2']['judge_status'] == 'service_error'
    assert rows['q1']['judge']['decision']['score'] == 0 == rows['q2']['judge']['decision']['score']
    assert rows['q1']['trace_id'] == 'svc-q1' and rows['q1']['evaluator_sha256']
    assert rows['q1']['retrieval_failure_cause'] is None and rows['q1']['first_gold_rank'] is None   # no retrieval ran

    roots = [s for s in fake.roots() if s.name == 'quality_evaluation']
    assert len(roots) == 2 and all(r.tags == ['evaluation', 'mode:hybrid', 'profile:bge'] and r.session_id == 'run-1' for r in roots)
    first = {(s['trace_id'], s['name']): s for s in fake.scores}
    assert first[('svc-q1', 'judge_status')]['value'] == 'output_invalid'
    assert first[('svc-q1', 'judge_decision')]['value'] == 0
    assert fake.flushes == 1

    # Rerun: rows are skipped but resent with the same score ids (idempotent recovery).
    ids = sorted(s['score_id'] for s in fake.scores)
    fake.scores.clear()
    quality.evaluate_comparison(Settings(_env_file=None), folder)
    assert sorted(s['score_id'] for s in fake.scores) == ids
    assert len([s for s in fake.roots() if s.name == 'quality_evaluation']) == 2
