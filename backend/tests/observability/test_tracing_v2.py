"""Review fixes (thermos): CLI stderr noise, query_done ids, generation_model rule, cloud_check early failure,
and behavior-level replacements for weak assertions in test_tracing.py."""
import logging
import subprocess
import sys
import time

import pytest

from backend.app.observability import query_trace
from backend.app.observability.tracing import _root_config
from backend.app.schemas import QueryResponse, ServiceError
from backend.tests.observability.fake_langfuse import FakeLangfuse

MANIFEST = {'run_id': 'run-1', 'request_id': 'req-1', 'question_id': 'q1', 'entrypoint': 'cli',
            'config': {'retrieval_mode': 'hybrid', 'embedding_profile': 'bge'}}


def response(latency_ms=7):
    return QueryResponse(answer='a', status='answered', reason='r', request_id='req-1', mode='hybrid', model='m',
                         latency_ms=latency_ms, citations=[{'book_id': 'B', 'page': 1, 'chunk_id': 'c'}])


def test_cli_runs_print_nothing_to_stderr_without_setup_logging():
    code = ("from backend.app.observability import log_event\n"
            "try:\n    raise RuntimeError('boom')\nexcept RuntimeError as e:\n"
            "    log_event('error', 'query_failed', exc=e)\nlog_event('warning', 'scores_not_sent', run_id='r')")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stderr == ''


@pytest.mark.parametrize('fail', [False, True])
def test_query_done_keeps_bound_ids_after_the_scope_closed(monkeypatch, caplog, fail):
    FakeLangfuse().install(monkeypatch)
    trace = query_trace(MANIFEST, kind='comparison', question='q')
    with caplog.at_level(logging.INFO, logger='backend.observability'):
        try:
            with trace:
                if fail:
                    raise ServiceError('index_missing', 'x')
                trace.finish(response())
        except ServiceError:
            pass
        trace.flush(time.perf_counter())
    done = next(r for r in caplog.records if r.getMessage() == 'query_done').fields
    assert (done['request_id'], done['run_id'], done['trace_id']) == ('req-1', 'run-1', trace.id)
    assert done.get('error_code') == ('index_missing' if fail else None)
    assert done.get('status') == (None if fail else 'answered')


@pytest.mark.parametrize('provider,model', [('ollama', 'local'), ('openai', 'api')])
def test_generation_model_follows_the_pipeline_rule(provider, model):
    config = {'generation_llm_provider': provider, 'local_llm_model': 'local', 'openai_model': 'api'}
    assert _root_config({'config': config})['generation_model'] == model


def test_finish_after_exit_leaves_the_root_without_finish_fields(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    trace = query_trace(MANIFEST, question='q')
    with trace:
        pass
    trace.finish(response())
    root = fake.named('shipping_rag_query')
    assert 'status' not in root.metadata and 'output' not in root.fields and fake.trace_io == []


def test_service_latency_excludes_flush_time(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    real_flush = fake.flush
    monkeypatch.setattr(fake, 'flush', lambda: (time.sleep(0.2), real_flush()))
    result = query_trace(MANIFEST, question='q').flush(time.perf_counter())
    assert result['flush_ms'] >= 150 and result['service_latency_ms'] < 50


def test_cloud_check_reports_failure_before_any_run_folder(monkeypatch, tmp_path):
    from backend.app import pipeline
    from backend.app.config import Settings
    from backend.app.observability import cloud_check
    FakeLangfuse().install(monkeypatch)
    monkeypatch.setattr('backend.app.config.get_settings', lambda: Settings(_env_file=None, runtime_dir=tmp_path))

    def down(*a, **k):
        raise ServiceError('local_model_missing', 'Install the configured local generation model')

    monkeypatch.setattr(pipeline, 'run_query', down)
    result = cloud_check.run_check('q', run_id='early-fail', timeout=0)
    assert result == {'checks': {'trace_created': False},
                      'facts': {'query': 'failed:local_model_missing', 'error_type': 'ServiceError'}}
