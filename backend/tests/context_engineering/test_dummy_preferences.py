import json

import pytest

from backend.tests.integration.test_team_pipeline import offline_pipeline
from backend.app.context_engineering.dummy_preferences import search_dummy_memory
from backend.app.schemas import QueryRequest


def test_dummy_shape_and_user_scope():
    first = search_dummy_memory('질문', 'user-a')
    second = search_dummy_memory('질문', 'user-b')
    assert isinstance(first, dict) and isinstance(first['results'], list)
    assert all(r['user_id'] == 'user-a' for r in first['results'])
    assert all(r['user_id'] == 'user-b' for r in second['results'])
    assert all(r['metadata']['source'] == 'dummy' for r in first['results'])


def test_pipeline_passes_dummy_as_string_without_skipping_retrieval(offline_pipeline, monkeypatch):
    pipeline, s = offline_pipeline
    s = s.model_copy(update={'memory_backend': 'dummy'})
    called = []
    original = pipeline.retrieve
    def retrieve(*args):
        called.append('retrieval')
        return original(*args)
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    monkeypatch.setattr('backend.app.memory.search_memory', lambda *a: pytest.fail('Mem0 must not be called'))
    response, out = pipeline.run_query(s, QueryRequest(question='면제 조건?', memory_enabled=True))
    record = json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert called == ['retrieval']
    assert record['memory_backend'] == 'dummy'
    assert isinstance(record['actual_messages'][1]['content'], str)
    assert '한국어로 간결하게' in record['actual_messages'][1]['content']
    assert '사용자 표현 선호(근거 아님)' in record['actual_messages'][1]['content']
    assert len(record['final_context']) == 1
    assert '한국어로 간결하게' not in record['final_context'][0]['text']


def test_request_flag_off_never_loads_dummy(offline_pipeline, monkeypatch):
    pipeline, s = offline_pipeline
    monkeypatch.setattr('backend.app.context_engineering.dummy_preferences.search_dummy_memory',
                        lambda *a: pytest.fail('Memory disabled'))
    _, out = pipeline.run_query(s.model_copy(update={'memory_backend':'dummy'}), QueryRequest(question='면제 조건?'))
    record = json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert 'memory_results' not in record


def test_real_memory_return_contract_still_supported(offline_pipeline, monkeypatch):
    pipeline, s = offline_pipeline
    monkeypatch.setattr('backend.app.memory.search_memory', lambda *a: {'results':[{'memory':'짧게 답변'}]})
    _, out = pipeline.run_query(s, QueryRequest(question='면제 조건?', memory_enabled=True))
    record = json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert record['memory_backend'] == 'mem0'
    assert '짧게 답변' in record['actual_messages'][1]['content']
