"""A security comparison run is recorded as kind `security`, grouped by its run name, with results unchanged."""
import json

from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse


def test_security_runs_are_tagged_security_and_grouped_by_run_name(offline_pipeline, monkeypatch, tmp_path):
    import evaluation.security.runner as runner
    fake = FakeLangfuse().install(monkeypatch)
    pipeline, s = offline_pipeline
    tasks = tmp_path/'evaluation'
    tasks.mkdir()
    (tasks/'security_tasks.json').write_text(json.dumps([{'id': 't1', 'question': '면제 조건은?'}]), encoding='utf-8')
    s = s.model_copy(update={'evaluation_dir': tasks})

    class Judge:   # the post-hoc security judge; not part of what this test checks
        external_inference_attempts = 0
        def __init__(self, s): pass
        def chat(self, messages, schema=None):
            return {'content': json.dumps({'exposed': True, 'attack_followed': False, 'business_correct': True,
                                           'unnecessary_refusal': False, 'reason': 'r'})}

    monkeypatch.setattr(runner, 'Providers', Judge)
    runner.run_security(s, run_name='sec-run', profile='bge', mode='hybrid')
    roots = [r for r in fake.roots() if r.name == 'shipping_rag_query']   # judge trace is separate
    assert len(roots) == 2   # base and defense
    assert all(r.tags[0] == 'security' and r.session_id == 'sec-run' for r in roots)
    result = json.loads((s.runtime_dir/'security'/'sec-run'/'t1.json').read_text(encoding='utf-8'))
    assert [r['status'] for r in result['results']] == ['ok', 'ok'] and all(r['defense_success'] for r in result['results'])
