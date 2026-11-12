import httpx

from backend.app.observability import query, tracing
from backend.tests.observability.fake_langfuse import FakeLangfuse

ROWS = [{'id': str(i), 'traceId': 't1', 'name': f's{i}', 'startTime': f'2026-11-11T00:00:0{9 - i}'} for i in range(5)]


def no_http(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('query.py must use SDK methods, not raw HTTP (old /api/public paths)')
    for method in ('get', 'post', 'request'):
        monkeypatch.setattr(httpx, method, refuse)


def test_get_trace_collects_all_pages_in_start_order(monkeypatch):
    no_http(monkeypatch)
    fake = FakeLangfuse(observations=ROWS + [{'id': 'x', 'traceId': 't2', 'name': 'other'}]).install(monkeypatch)
    rows = query.get_trace('t1')
    assert [r['id'] for r in rows] == ['4', '3', '2', '1', '0']
    calls = fake.api.observations.calls
    assert [c['cursor'] for c in calls] == [None, '2', '4'] and calls[0]['fields'] == 'core,basic,io,metadata,usage,model,prompt,trace_context'


def test_list_traces_filters_by_session_tags_and_where(monkeypatch):
    fake = FakeLangfuse(observations=[{'id': 'r', 'traceId': 't', 'name': 'shipping_rag_query', 'sessionId': 'run-1'}]).install(monkeypatch)
    where = [{'type': 'boolean', 'column': 'metadata', 'key': 'context_reduced', 'operator': '=', 'value': True}]
    assert len(query.list_traces(run_id='run-1', tags=['comparison'], where=where)) == 1
    call = fake.api.observations.calls[0]
    assert call['session_id'] == 'run-1' and call['is_root_observation'] is True
    assert '"tags"' in call['filter'] and '"context_reduced"' in call['filter']


def test_get_scores_uses_v3(monkeypatch):
    fake = FakeLangfuse(scores=[{'id': 's', 'traceId': 't1', 'name': 'hit_at_k'}]).install(monkeypatch)
    assert query.get_scores(trace_id='t1', name='hit_at_k')[0]['id'] == 's'
    assert fake.api.scores_v3.calls[0]['trace_id'] == 't1'


def test_langfuse_off_or_error_returns_empty_list(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)

    def broken(**kwargs):
        raise RuntimeError('rate limited')

    fake.api.observations.get_many = broken
    assert query.get_trace('t1') == [] and query.list_traces() == []
    monkeypatch.setattr(tracing, 'langfuse_client', lambda: None)
    assert query.get_trace('t1') == [] and query.get_scores() == []
