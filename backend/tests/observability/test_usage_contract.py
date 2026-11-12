import httpx
import pytest

from backend.app.config import Settings
from backend.app.observability import generation_metadata, generation_parameters, rank_list, usage_details
from backend.app.providers import Providers


def test_usage_does_not_turn_unknown_into_zero_or_duplicate_content():
    generated = {'content':'답변', 'input_tokens':None, 'output_tokens':0, 'attempts':[]}
    assert usage_details(generated) == {'output':0}
    meta = generation_metadata(generated, provider='openai', request_id='r1')
    assert meta['request_id'] == 'r1' and 'content' not in meta


def sent_payload(monkeypatch, s):
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs['json'])
        request = httpx.Request('POST', url)
        body = {'choices': [{'message': {'content': '{}'}}], 'usage': {}} if 'chat/completions' in url \
            else {'message': {'content': '{}'}}
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx, 'post', post)
    Providers(s)._chat_once([{'role': 'user', 'content': 'q'}], None)
    return captured


@pytest.mark.parametrize('provider', ['ollama', 'openai'])
def test_generation_parameters_match_what_providers_send(monkeypatch, provider):
    s = Settings(_env_file=None, generation_llm_provider=provider, force_local_only=provider == 'ollama',
                 allow_external_llm=provider == 'openai', openai_api_key='test-secret')
    payload = sent_payload(monkeypatch, s)
    sent = payload if provider == 'openai' else payload['options']
    params = generation_parameters(s)
    assert {k: sent[k] for k in params} == params


def test_rank_list_has_no_text_and_prefers_rerank_score():
    rows = [{'chunk_id': 'c1', 'book_id': 'B', 'page': 2, 'text': '본문', 'score': 0.2, 'rerank_score': 0.9},
            {'chunk_id': 'c2', 'book_id': 'B', 'page': 3, 'text': '본문', 'score': 0.1}]
    assert rank_list(rows) == [{'rank': 1, 'chunk_id': 'c1', 'book_id': 'B', 'page': 2, 'score': 0.9},
                               {'rank': 2, 'chunk_id': 'c2', 'book_id': 'B', 'page': 3, 'score': 0.1}]
    assert 'text' in rows[0]
