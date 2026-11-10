import json
from contextlib import contextmanager

import httpx
import pytest

from backend.app.config import Settings
from backend.app.indexing import _cached_bm25, load_bm25
from backend.app.providers import Providers
from backend.app.schemas import ModelAnswer, ServiceError


@pytest.fixture
def spans(monkeypatch):
    events = []
    @contextmanager
    def observe(*args, **kwargs):
        class Span:
            def update(self, **data):
                events.append(data)
        yield Span()
    monkeypatch.setattr('backend.app.observability.observation', observe)
    return events


def api_settings(**kwargs):
    return Settings(_env_file=None, generation_llm_provider='openai',
                    force_local_only=False, allow_external_llm=True,
                    openai_base_url='https://api.openai.com/v1', openai_api_key='test-secret', **kwargs)


def test_bm25_reuses_version_and_invalidates_changed_file(tmp_path):
    _cached_bm25.cache_clear()
    path = tmp_path/'chunks.jsonl'
    path.write_text(json.dumps({'text': 'shipping invoice', 'chunk_id': 'a'})+'\n')
    first = load_bm25(path, 'v1')
    assert load_bm25(path, 'v1') is first
    assert load_bm25(path, 'v2') is not first
    path.write_text(json.dumps({'text': 'updated shipping invoice longer', 'chunk_id': 'b'})+'\n')
    assert load_bm25(path, 'v1')[0][0]['chunk_id'] == 'b'


def test_openai_usage_ids_and_strict_schema(monkeypatch, spans):
    def post(url, **kwargs):
        assert url == 'https://api.openai.com/v1/chat/completions'
        schema = kwargs['json']['response_format']['json_schema']['schema']
        assert schema['additionalProperties'] is False
        assert set(schema['required']) == set(schema['properties'])
        assert schema['$defs']['Citation']['additionalProperties'] is False
        return httpx.Response(200, request=httpx.Request('POST', url), headers={'x-request-id': 'req-1'},
            json={'id': 'response-1', 'model': 'gpt-6-luna',
                  'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 12, 'completion_tokens': 8,
                            'prompt_tokens_details': {'cached_tokens': 4},
                            'completion_tokens_details': {'reasoning_tokens': 2}}})
    monkeypatch.setattr('backend.app.providers.httpx.post', post)
    provider = Providers(api_settings())
    result = provider.chat([], ModelAnswer.model_json_schema())
    assert result['input_tokens'] == 12 and result['output_tokens'] == 8
    assert result['provider_request_id'] == 'req-1'
    assert result['usage']['prompt_tokens_details']['cached_tokens'] == 4
    assert provider.external_inference_attempts == 1
    assert 'test-secret' not in json.dumps(spans)


def test_retry_records_failed_attempt_and_missing_usage(monkeypatch, spans):
    replies = [503, 200]
    def post(url, **kwargs):
        status = replies.pop(0)
        return httpx.Response(status, request=httpx.Request('POST', url),
            json={'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr('backend.app.providers.httpx.post', post)
    result = Providers(api_settings()).chat([])
    assert [a['status'] for a in result['attempts']] == ['error', 'ok']
    assert result['input_tokens'] is None and result['output_tokens'] is None
    assert spans[0]['level'] == 'ERROR'


def test_deadline_prevents_network_call(monkeypatch, spans):
    monkeypatch.setattr('backend.app.providers.httpx.post', lambda *a, **k: pytest.fail('network'))
    provider = Providers(api_settings())
    provider.deadline = -1
    with pytest.raises(ServiceError) as error:
        provider.chat([])
    assert error.value.code == 'request_timeout'
    assert error.value.attempts[0]['error_code'] == 'request_timeout'


def test_startup_warms_same_cached_reranker(monkeypatch):
    import asyncio
    from backend.app import main
    from backend.app import reranker
    reranker._cached.cache_clear()
    calls = []
    class FakeReranker:
        device = 'test'
        def __init__(self, *args):
            calls.append('load')
        def score(self, query, docs):
            calls.append('score')
            return [1.0]
    monkeypatch.setattr(main, 's', Settings(_env_file=None, reranker_warmup_on_startup=True))
    monkeypatch.setattr(reranker, 'Reranker', FakeReranker)
    async def run():
        async with main.lifespan(main.app):
            assert main.app.state.reranker_warmup['device'] == 'test'
            assert reranker.get_reranker(main.s) is reranker.get_reranker(main.s)
    asyncio.run(run())
    assert calls == ['load', 'score']
    reranker._cached.cache_clear()


def test_warmup_failure_prevents_startup(monkeypatch):
    import asyncio
    from backend.app import main
    monkeypatch.setattr(main, 's', Settings(_env_file=None, reranker_warmup_on_startup=True))
    def fail(s):
        raise RuntimeError('GPU unavailable')
    monkeypatch.setattr('backend.app.reranker.get_reranker', fail)
    async def run():
        async with main.lifespan(main.app):
            pytest.fail('startup must fail')
    with pytest.raises(RuntimeError, match='GPU unavailable'):
        asyncio.run(run())


def test_ollama_duration_conversion(monkeypatch, spans):
    def post(url, **kwargs):
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'message': {'content': '{}'}, 'prompt_eval_count': 10, 'eval_count': 20,
            'load_duration': 1000000000, 'eval_duration': 2000000000})
    monkeypatch.setattr('backend.app.providers.httpx.post', post)
    result = Providers(Settings(_env_file=None)).chat([])
    assert result['timing']['load_seconds'] == 1
    assert result['timing']['output_tokens_per_second'] == 10

