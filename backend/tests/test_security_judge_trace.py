"""Security judge calls are grouped in one `security_evaluation` trace per task instead of orphan llm_attempt traces."""
import json

from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse

VERDICT = {'exposed': True, 'attack_followed': False, 'business_correct': True, 'unnecessary_refusal': False, 'reason': 'r'}


def test_judge_calls_sit_under_one_security_evaluation_trace(offline_pipeline, monkeypatch, tmp_path):
    import evaluation.security.runner as runner
    from backend.app.observability import observation
    fake = FakeLangfuse().install(monkeypatch)
    pipeline, s = offline_pipeline
    tasks = tmp_path/'evaluation'
    tasks.mkdir()
    (tasks/'security_tasks.json').write_text(json.dumps([{'id': 't1', 'question': '면제 조건은?', 'attack': 'ATTACK-TEXT'}]),
                                             encoding='utf-8')

    class Judge:   # like Providers.chat: each call opens an llm_attempt observation
        external_inference_attempts = 0
        def __init__(self, s): pass
        def chat(self, messages, schema=None):
            with observation('llm_attempt', metadata={'attempt': 1}):
                return {'content': json.dumps(VERDICT)}

    monkeypatch.setattr(runner, 'Providers', Judge)
    runner.run_security(s.model_copy(update={'evaluation_dir': tasks}), run_name='sec-run', profile='bge', mode='hybrid')

    judge_root = fake.named('security_evaluation')
    service = [r for r in fake.roots() if r.name == 'shipping_rag_query']
    assert [r.name for r in fake.roots()].count('llm_attempt') == 0
    assert judge_root.tags == ['security', 'mode:hybrid', 'profile:bge'] and judge_root.session_id == 'sec-run'
    assert sorted(judge_root.metadata['source_trace_ids']) == sorted(r.trace_id for r in service)
    judges = [sp for sp in fake.spans if sp.name == 'security_judge']
    assert len(judges) == 2 and all(sp.trace_id == judge_root.trace_id for sp in judges)
    assert {sp.metadata['variant'] for sp in judges} == {'base', 'defense'}
    assert all(sp.fields['output'] == VERDICT for sp in judges)
    assert 'ATTACK-TEXT' not in json.dumps([r.fields.get('input') for r in service], ensure_ascii=False)
