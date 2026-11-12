from langfuse.api import Prompt_Chat
from langfuse.model import ChatPromptClient

from backend.app.context_engineering.prompts import RAG_PROMPT
from backend.app.observability import SCORE_NAMES, to_langfuse_messages
from backend.app.observability import setup
from backend.tests.observability.fake_langfuse import FakeLangfuse


def test_preview_writes_nothing(monkeypatch, capsys):
    fake = FakeLangfuse(score_configs=['hit_at_k', 'judge_status']).install(monkeypatch)
    assert setup.main([]) == 0
    out = capsys.readouterr().out
    assert 'preview: create_prompt rag_answer' in out and 'create_score_config hit_at_k' not in out
    assert fake.created_prompts == [] and fake.api.score_configs.created == []


def test_apply_creates_missing_prompt_and_configs_from_scores(monkeypatch):
    fake = FakeLangfuse(score_configs=['hit_at_k']).install(monkeypatch)
    assert setup.main(['--apply']) == 0
    assert fake.created_prompts == [{'name': 'rag_answer', 'prompt': to_langfuse_messages(RAG_PROMPT),
                                     'labels': ['production', 'comparison-v1'], 'type': 'chat'}]
    created = {c['name']: c for c in fake.api.score_configs.created}
    assert set(created) == SCORE_NAMES - {'hit_at_k'}
    assert [c.label for c in created['judge_status']['categories']] == ['unscorable', 'service_error', 'output_invalid', 'ok']
    assert created['recall_at_k'] == {'name': 'recall_at_k', 'data_type': 'NUMERIC', 'min_value': 0, 'max_value': 1}


def test_existing_prompt_is_skipped(monkeypatch):
    stored = ChatPromptClient(Prompt_Chat(name='rag_answer', version=1, prompt=to_langfuse_messages(RAG_PROMPT),
                                          type='chat', config={}, labels=['production'], tags=[]))
    fake = FakeLangfuse(prompts={'rag_answer': stored}, score_configs=sorted(SCORE_NAMES)).install(monkeypatch)
    assert setup.plan(fake) == []
