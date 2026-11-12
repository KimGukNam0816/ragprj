import pytest

from backend.app.observability import verify_trace
from backend.app.observability.verification import required_spans
from backend.tests.observability.fake_langfuse import FakeLangfuse

BASE = {'shipping_rag_query', 'context_budget', 'context_prompt'}


@pytest.mark.parametrize('source,count,expected', [
    ('retrieval', 3, BASE | {'query_embedding', 'dense_retrieval', 'answer_generation', 'output_validation'}),
    ('fixed', 3, BASE | {'answer_generation', 'output_validation'}),
    ('retrieval', 0, BASE | {'query_embedding', 'dense_retrieval'}),
])
def test_required_spans_by_path(source, count, expected):
    assert required_spans(source, count) == expected


def span(name, **fields):
    return {'traceId': 't1', 'name': name, **fields}


def fixed_trace(root_metadata):
    return [span('shipping_rag_query', metadata=root_metadata, input={'request_id': 'req-1'}),
            span('context_budget'), span('context_prompt'),
            span('answer_generation', input=[{'role': 'user'}], output='{}'), span('output_validation')]


def test_fixed_context_trace_verifies_without_retrieval_spans(monkeypatch):
    FakeLangfuse(observations=fixed_trace({'context_source': 'fixed', 'context_count': 2})).install(monkeypatch)
    result = verify_trace(None, 't1', 'req-1', wait_seconds=0)
    assert result['verified'] is True and result['observation_count'] == 5


def test_string_metadata_from_api_is_coerced(monkeypatch):
    FakeLangfuse(observations=fixed_trace({'context_source': '"fixed"', 'context_count': '2'})).install(monkeypatch)
    assert verify_trace(None, 't1', 'req-1', wait_seconds=0)['verified'] is True


def test_missing_spans_are_reported_after_retries(monkeypatch):
    fake = FakeLangfuse(observations=fixed_trace({})).install(monkeypatch)   # no context_source -> retrieval path
    result = verify_trace(None, 't1', 'req-1', attempts=2, wait_seconds=0)
    assert result['verified'] is False and result['missing'] == ['dense_retrieval', 'query_embedding']
    assert len([c for c in fake.api.observations.calls if c['cursor'] is None]) == 2


def test_no_trace_id():
    assert verify_trace(None, None, 'req-1') == {'verified': False, 'reason': 'No trace ID'}
