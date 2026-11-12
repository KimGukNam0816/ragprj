import json

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langfuse.api import Prompt_Chat
from langfuse.model import ChatPromptClient

from backend.app.context_engineering.prompts import PROMPT_VERSION, RAG_PROMPT
from backend.app.observability import ResolvedPrompt, pinned_prompt_version, resolve_prompt, to_langfuse_messages
from backend.app.observability import prompting
from backend.app.observability.prompting import PromptUnavailable
from backend.tests.observability.fake_langfuse import FakeLangfuse

VALUES = {'question': '질문 {x}', 'memory': '없음', 'context': '[{"book_id": "AS001"}]'}


def langfuse_prompt(template, version=3):
    return ChatPromptClient(Prompt_Chat(name='rag_answer', version=version, prompt=to_langfuse_messages(template),
                                        type='chat', config={}, labels=['production'], tags=[]))


@pytest.fixture
def langfuse_mode(monkeypatch):
    monkeypatch.setattr(prompting, '_prompt_source', lambda: 'langfuse')
    monkeypatch.setattr(prompting, '_failed_until', {})


def test_round_trip_through_langfuse_reproduces_formatted_messages():
    """Plan 8 step 1.5: LangChain -> Langfuse -> get_langchain_prompt() must format identically."""
    back = ChatPromptTemplate.from_messages(langfuse_prompt(RAG_PROMPT).get_langchain_prompt())
    assert back.format_messages(**VALUES) == RAG_PROMPT.format_messages(**VALUES)


def test_local_mode_never_calls_the_client(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    prompt = resolve_prompt('rag_answer', RAG_PROMPT, strict=True)
    assert (prompt.source, prompt.version) == ('local', PROMPT_VERSION) and fake.prompt_calls == []
    assert prompt.config_fields == {'prompt_name': 'rag_answer', 'prompt_version': PROMPT_VERSION,
                                    'prompt_source': 'local', 'prompt_sha256': prompt.sha256}


def test_hash_covers_both_messages():
    human_changed = ChatPromptTemplate.from_messages([RAG_PROMPT.messages[0], ('human', '질문: {question}')])
    assert ResolvedPrompt.local('rag_answer', human_changed).sha256 != ResolvedPrompt.local('rag_answer', RAG_PROMPT).sha256


def test_same_template_has_same_hash_in_both_modes(monkeypatch, langfuse_mode):
    FakeLangfuse(prompts={'rag_answer': langfuse_prompt(RAG_PROMPT)}).install(monkeypatch)
    fetched = resolve_prompt('rag_answer', RAG_PROMPT)
    assert (fetched.source, fetched.version) == ('langfuse', 'lf-3')
    assert fetched.sha256 == ResolvedPrompt.local('rag_answer', RAG_PROMPT).sha256
    assert fetched._link is not None


def test_version_wins_over_label(monkeypatch, langfuse_mode):
    fake = FakeLangfuse(prompts={'rag_answer': langfuse_prompt(RAG_PROMPT)}).install(monkeypatch)
    resolve_prompt('rag_answer', RAG_PROMPT, version=3)
    assert fake.prompt_calls[0]['version'] == 3 and 'label' not in fake.prompt_calls[0]


def test_fallback_is_marked_strict_raises_and_negative_cache_skips_the_network(monkeypatch, langfuse_mode):
    fake = FakeLangfuse().install(monkeypatch)   # no prompt stored -> SDK returns its fallback
    prompt = resolve_prompt('rag_answer', RAG_PROMPT)
    assert prompt.source == 'fallback' and prompt.template is RAG_PROMPT
    with pytest.raises(PromptUnavailable) as raised:
        resolve_prompt('rag_answer', RAG_PROMPT, strict=True)
    assert raised.value.code == 'prompt_unavailable'
    assert len(fake.prompt_calls) == 1   # second call within 60 s did not fetch


def test_fetch_exception_is_a_fallback(monkeypatch, langfuse_mode):
    FakeLangfuse(prompts={'rag_answer': RuntimeError('timeout')}).install(monkeypatch)
    assert resolve_prompt('rag_answer', RAG_PROMPT).source == 'fallback'


def test_langfuse_off_is_local_even_when_strict(monkeypatch, langfuse_mode):
    monkeypatch.setattr(prompting.tracing, 'langfuse_client', lambda: None)
    assert resolve_prompt('rag_answer', RAG_PROMPT, strict=True).source == 'local'


def test_pinned_version_is_written_once_and_reused(monkeypatch, langfuse_mode, tmp_path):
    fake = FakeLangfuse(prompts={'rag_answer': langfuse_prompt(RAG_PROMPT, version=7)}).install(monkeypatch)
    assert pinned_prompt_version(tmp_path) == 7
    fake.prompts['rag_answer'] = langfuse_prompt(RAG_PROMPT, version=8)
    assert pinned_prompt_version(tmp_path) == 7
    saved = json.loads((tmp_path / 'prompt_version.json').read_text(encoding='utf-8'))
    assert saved['label'] == 'comparison-v1' and saved['source'] == 'langfuse'


def test_pinned_version_is_null_in_local_mode(monkeypatch, tmp_path):
    FakeLangfuse().install(monkeypatch)
    assert pinned_prompt_version(tmp_path) is None
    assert json.loads((tmp_path / 'prompt_version.json').read_text(encoding='utf-8'))['source'] == 'local'
