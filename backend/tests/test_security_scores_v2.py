"""Security comparison results are sent as Langfuse scores on each answer's service trace (local file first)."""
import json

import pytest

from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse

VERDICT = {'exposed': True, 'attack_followed': False, 'business_correct': True, 'unnecessary_refusal': False, 'reason': 'r'}


@pytest.fixture
def security_run(offline_pipeline, monkeypatch, tmp_path):
    import evaluation.security.runner as runner
    pipeline, s = offline_pipeline
    tasks = tmp_path/'evaluation'
    tasks.mkdir()
    (tasks/'security_tasks.json').write_text(json.dumps([{'id': 't1', 'question': '면제 조건은?'}]), encoding='utf-8')

    class Judge:
        external_inference_attempts = 0
        def __init__(self, s): pass
        def chat(self, messages, schema=None):
            return {'content': json.dumps(VERDICT)}

    monkeypatch.setattr(runner, 'Providers', Judge)
    return runner, pipeline, s.model_copy(update={'evaluation_dir': tasks})


def latest(fake):
    return {(s['trace_id'], s['name']): s['value'] for s in {s['score_id']: s for s in fake.scores}.values()}


def test_both_variants_get_security_scores_on_their_own_trace(security_run, monkeypatch):
    runner, _, s = security_run
    fake = FakeLangfuse().install(monkeypatch)
    runner.run_security(s, run_name='sec-run', profile='bge', mode='hybrid')
    roots = {r.trace_id for r in fake.roots() if r.name == 'shipping_rag_query'}
    sent = latest(fake)
    assert {t for t, _ in sent} == roots and len(roots) == 2
    for trace in roots:
        assert sent[(trace, 'security_case_status')] == 'defended' and sent[(trace, 'security_case_pass')] == 1
        assert sent[(trace, 'security_exposed')] == 1 and sent[(trace, 'output_valid')] == 1
    saved = json.loads((s.runtime_dir/'security'/'sec-run'/'t1.json').read_text(encoding='utf-8'))
    assert all(r['trace_id'] in roots and r['evaluator_sha256'] for r in saved['results'])
    assert fake.flushes >= 1


def test_output_invalid_answer_is_generation_failed_and_output_valid_zero(security_run, monkeypatch):
    runner, pipeline, s = security_run
    fake = FakeLangfuse().install(monkeypatch)
    real_chat = pipeline.Providers.chat

    def broken(self, messages, schema=None):
        result = real_chat(self, messages, schema)
        data = json.loads(result['content'])
        data['citations'] *= 2
        return {**result, 'content': json.dumps(data)}

    monkeypatch.setattr(pipeline.Providers, 'chat', broken)
    runner.run_security(s, run_name='sec-bad', profile='bge', mode='hybrid')
    sent = latest(fake)
    statuses = {v for (t, n), v in sent.items() if n == 'security_case_status'}
    assert statuses == {'generation_failed'} and {v for (t, n), v in sent.items() if n == 'output_valid'} == {0}
    assert not any(n == 'security_case_pass' for _, n in sent)


def test_langfuse_off_keeps_the_result_file(security_run, monkeypatch):
    runner, _, s = security_run
    monkeypatch.setattr('backend.app.observability.tracing.langfuse_client', lambda: None)
    runner.run_security(s, run_name='sec-off', profile='bge', mode='hybrid')
    saved = json.loads((s.runtime_dir/'security'/'sec-off'/'t1.json').read_text(encoding='utf-8'))
    assert [r['defense_success'] for r in saved['results']] == [True, True]
