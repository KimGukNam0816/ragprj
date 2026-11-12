"""Review fix (thermos H1/L1): until D5, a comparison run pins the local template even in PROMPT_SOURCE=langfuse mode."""
import json

from backend.app.artifacts import write_json
from backend.app.config import Settings
from backend.app.observability import prompting
from backend.tests.observability.fake_langfuse import FakeLangfuse
from evaluation.retrieval import benchmark


def test_langfuse_mode_neither_fetches_nor_aborts_and_records_local(tmp_path, monkeypatch):
    fake = FakeLangfuse(prompts={'rag_answer': RuntimeError('fetch failed')}).install(monkeypatch)
    monkeypatch.setattr(prompting, '_prompt_source', lambda: 'langfuse')
    s = Settings(_env_file=None, runtime_dir=tmp_path)
    for profile in ('bge', 'qwen'):
        write_json(tmp_path/'indexes'/f'active_{profile}.json', {'collection': profile})
    monkeypatch.setattr(benchmark, 'questions', lambda s, selection: [{'id': 'q1', 'question': '질문'}])
    monkeypatch.setattr(benchmark, 'run_query', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
    folder = benchmark.run_comparison(s, 'core10', run_name='pin-v2')
    pinned = json.loads((folder/'prompt_version.json').read_text(encoding='utf-8'))
    assert (pinned['source'], pinned['version']) == ('local', None) and fake.prompt_calls == []
