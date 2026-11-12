import json

from backend.app.artifacts import write_json
from backend.app.config import Settings
from evaluation.retrieval import benchmark


def test_comparison_runs_are_tagged_and_pin_the_prompt_once(tmp_path, monkeypatch):
    monkeypatch.setattr('backend.app.observability.tracing.langfuse_client', lambda: None)
    s = Settings(_env_file=None, runtime_dir=tmp_path)
    for profile in ('bge', 'qwen'):
        write_json(tmp_path/'indexes'/f'active_{profile}.json', {'collection': profile})
    monkeypatch.setattr(benchmark, 'questions', lambda s, selection: [{'id': 'q1', 'question': '질문'}])
    calls = []

    def run_query(s, request, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('offline')

    monkeypatch.setattr(benchmark, 'run_query', run_query)
    folder = benchmark.run_comparison(s, 'core10', run_name='pin-test')
    assert {(c['run_kind'], c['prompt_version']) for c in calls} == {('comparison', None)} and len(calls) == 6
    assert json.loads((folder/'prompt_version.json').read_text(encoding='utf-8'))['source'] == 'local'
