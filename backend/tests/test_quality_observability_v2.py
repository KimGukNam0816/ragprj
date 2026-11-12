"""Review fix (thermos, security L4): rows that never reached retrieval keep no rank or failure cause."""
from backend.app.artifacts import write_json, write_jsonl
from backend.app.config import Settings
from backend.tests.observability.fake_langfuse import FakeLangfuse
from evaluation import quality

REF = {'evidence_groups': [[{'book_id': 'G1'}]]}


def item(tmp_path, question_id, saved):
    path = tmp_path/f'{question_id}.json'
    write_json(path, {'trace_id': f'svc-{question_id}', 'error': {'code': 'qdrant_error'}, **saved})
    return {'profile': 'bge', 'mode': 'hybrid', 'question_id': question_id, 'question': '질문',
            'status': 'error', 'result_path': str(path)}


def test_failure_before_and_after_retrieval(tmp_path, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    folder = tmp_path/'run-v2'
    candidates = [{'book_id': 'X', 'chunk_id': 'x'}, {'book_id': 'G1', 'chunk_id': 'g'}]
    write_jsonl(folder/'runs.jsonl', [
        item(tmp_path, 'before', {'candidates': candidates}),                                  # no final_context saved
        item(tmp_path, 'after', {'candidates': candidates, 'final_context': [{'book_id': 'X', 'chunk_id': 'x'}]}),
    ])
    monkeypatch.setattr(quality, 'reference_map', lambda s: {'before': REF, 'after': REF})
    quality.evaluate_comparison(Settings(_env_file=None), folder)
    rows = {r['question_id']: r for r in map(__import__('json').loads, (folder/'scores.jsonl').read_text(encoding='utf-8').splitlines())}
    assert (rows['before']['retrieval_status'], rows['before']['retrieval_failure_cause'], rows['before']['first_gold_rank']) == \
        ('unscorable', None, None)
    assert (rows['after']['retrieval_status'], rows['after']['retrieval_failure_cause'], rows['after']['first_gold_rank']) == \
        ('ok', 'context_dropped', 2)
    sent = {(s['trace_id'], s['name']): s['value'] for s in fake.scores}
    assert sent[('svc-before', 'retrieval_status')] == 'unscorable'
    assert not {n for t, n in sent if t == 'svc-before'} & {'retrieval_failure_cause', 'first_gold_rank', 'hit_at_k'}
    assert ('svc-after', 'retrieval_failure_cause') not in sent and sent[('svc-after', 'first_gold_rank')] == 2   # failed request
