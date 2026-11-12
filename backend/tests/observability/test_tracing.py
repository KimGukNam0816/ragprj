import asyncio
import logging
from dataclasses import asdict

import pytest

from backend.app.contracts import PreparedContext
from backend.app.observability import evaluation_trace, observation, query_trace, trace_url
from backend.app.observability import tracing
from backend.app.observability.tracing import _scope_fields
from backend.tests.observability.fake_langfuse import FakeLangfuse

MANIFEST = {'run_id': 'run-1', 'request_id': 'req-1', 'question_id': 'q1', 'entrypoint': 'api',
            'config': {'retrieval_mode': 'naive', 'embedding_profile': 'qwen', 'prompt_sha256': 'abcdef0123'}}


def test_scope_fields_drop_none_non_ascii_and_long_values_with_key_only_warning(caplog):
    manifest = {**MANIFEST, 'question_id': '질문-1', 'run_id': 'r' * 201, 'entrypoint': None}
    with caplog.at_level(logging.WARNING, logger='backend.observability'):
        tags, session, metadata = _scope_fields(manifest, 'comparison', variant='x' * 300)
    assert tags == ['comparison', 'mode:naive', 'profile:qwen', 'prompt:abcdef01']
    assert session == 'run-' + __import__('hashlib').sha256(('r' * 201).encode()).hexdigest()[:8] and metadata == {'request_id': 'req-1'}
    messages = [r.getMessage() for r in caplog.records]
    assert {r.fields['key'] for r in caplog.records if r.getMessage() == 'propagated_value_dropped'} >= \
        {'question_id', 'variant'}
    assert not any('질문-1' in m or 'x' * 300 in m for m in messages)


def test_service_traces_have_no_session():
    assert _scope_fields(MANIFEST, 'service')[1] is None
    assert _scope_fields(MANIFEST, 'security', variant='defense')[0][-1] == 'prompt:abcdef01'
    assert 'variant:defense' in _scope_fields(MANIFEST, 'security', variant='defense')[0]


def test_unknown_kind_is_rejected_before_any_work():
    with pytest.raises(ValueError):
        query_trace(MANIFEST, kind='evaluation_run', question='q')


def test_finish_after_exit_only_warns(monkeypatch, caplog):
    fake = FakeLangfuse().install(monkeypatch)
    trace = query_trace(MANIFEST, kind='service', question='q')
    with trace:
        pass

    class Response:
        def model_dump(self):
            return {'status': 'answered', 'citations': [], 'latency_ms': 1}

    with caplog.at_level(logging.WARNING, logger='backend.observability'):
        trace.finish(Response())
    assert 'trace_write_after_close' in caplog.text and fake.dropped == []


def test_record_context_does_not_mutate_prepared(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    prepared = PreparedContext([{'chunk_id': 'a', 'book_id': 'B', 'page': 1, 'text': 't'}], [], 3, 'h', ['b'], [])
    before = asdict(prepared)
    with query_trace(MANIFEST, kind='security', question='q') as trace:
        trace.record_context(prepared, source='fixed')
    assert asdict(prepared) == before
    assert fake.named('shipping_rag_query').metadata.items() >= {'context_source': 'fixed', 'context_count': 1,
                                                                'context_reduced': True, 'dropped_count': 1}.items()


def test_reason_attribute_reaches_root_metadata(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)

    class OutputInvalid(ValueError):
        code, reason = 'output_invalid', 'duplicate_citation'

    with pytest.raises(OutputInvalid):
        with query_trace(MANIFEST, question='q'):
            raise OutputInvalid()
    assert fake.named('shipping_rag_query').metadata['error_reason'] == 'duplicate_citation'


def test_flush_never_raises(monkeypatch):
    FakeLangfuse(flush_error=RuntimeError('network')).install(monkeypatch)
    trace = query_trace(MANIFEST, question='q')
    result = trace.flush(0.0)
    assert result['flush_error'] == 'RuntimeError' and result['service_latency_ms'] > 0


def test_observation_without_prompt_keeps_existing_contract(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    with observation('dense_retrieval', metadata={'k': 1}) as span:
        assert span is fake.named('dense_retrieval')
    assert span.metadata == {'k': 1} and 'model_parameters' not in span.fields
    monkeypatch.setattr(tracing, 'langfuse_client', lambda: None)
    with observation('dense_retrieval') as span:
        assert span is None


def test_evaluation_trace_tags_reach_spans_inside_asyncio_run_and_to_thread(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)

    def judge():
        with observation('llm_judge', kind='evaluator'):
            pass

    async def evaluate():
        with observation('ragas_faithfulness', kind='evaluator'):
            await asyncio.to_thread(judge)

    with evaluation_trace(run_id='run-1', question_id='q1', mode='hybrid', profile='bge',
                          source={'request_id': 'req-1', 'trace_id': 't-svc', 'answer': '본문'}):
        asyncio.run(evaluate())
    root = fake.named('quality_evaluation')
    assert root.metadata['source_trace_id'] == 't-svc' and root.metadata['source_request_id'] == 'req-1'
    assert 'ragas_version' in root.metadata and '본문' not in str(root.metadata)
    for name in ('ragas_faithfulness', 'llm_judge'):
        span = fake.named(name)
        assert span.trace_id == root.trace_id
        assert span.tags == ['evaluation', 'mode:hybrid', 'profile:bge'] and span.session_id == 'run-1'


def test_trace_url_returns_none_on_any_failure(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    assert trace_url('t1').endswith('/traces/t1')
    assert trace_url(None) is None

    def broken(*, trace_id):
        raise RuntimeError('projects api down')

    monkeypatch.setattr(fake, 'get_trace_url', broken)
    assert trace_url('t1') is None
