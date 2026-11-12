import json
import logging
import threading

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.app.observability import logs
from backend.app.observability.logs import access_log, log_event, setup_logging, timed_lock


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(logs, '_handlers', {})
    setup_logging('test')
    yield tmp_path / 'test.log'
    for logger in (logs._logger, logging.getLogger('langfuse')):
        for handler in list(logger.handlers):
            if handler in logs._handlers.values():
                logger.removeHandler(handler)
                handler.close()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_only_allowed_fields_are_written_as_one_json_line(log_file):
    log_event('info', 'query_done', request_id='r1', latency_ms=5, question='원문 질문')
    rows = lines(log_file)
    done = next(r for r in rows if r['message'] == 'query_done')
    assert done['request_id'] == 'r1' and done['latency_ms'] == 5 and 'question' not in done
    warning = next(r for r in rows if r['message'] == 'log_fields_dropped')
    assert warning['dropped_keys'] == ['question'] and '원문 질문' not in log_file.read_text(encoding='utf-8')


def test_exc_adds_stack(log_file):
    try:
        raise RuntimeError('boom')
    except RuntimeError as exc:
        log_event('error', 'query_failed', exc=exc, error_code='pipeline_error', error_reason='json')
    row = lines(log_file)[-1]
    assert 'RuntimeError' in row['stack'] and row['error_reason'] == 'json'


def test_setup_is_idempotent_and_attaches_langfuse_logger(log_file):
    setup_logging('test')
    for logger in (logs._logger, logging.getLogger('langfuse')):
        assert sum(h in logs._handlers.values() for h in logger.handlers) == 1


def test_access_log_carries_ids_bound_inside_sync_endpoint_and_logs_422(log_file):
    app = FastAPI()
    app.middleware('http')(access_log)
    lock = threading.Lock()

    class Body(BaseModel):
        text: str

    @app.post('/query')
    def query(body: Body):   # sync endpoint: runs in the threadpool with a copied context
        with timed_lock(lock):
            logs.bind(request_id='req-1', trace_id='t-1')
        raise HTTPException(502, 'output_invalid')

    client = TestClient(app)
    assert client.post('/query?secret=1', json={'text': 'x'}).status_code == 502
    assert client.post('/query', json={}).status_code == 422
    entries = [r for r in lines(log_file) if r['message'] == 'request']
    assert entries[0] | {'latency_ms': 0, 'lock_wait_ms': 0, 'ts': ''} == {
        'ts': '', 'level': 'info', 'logger': 'backend.observability', 'message': 'request', 'method': 'POST',
        'path': '/query', 'http_status': 502, 'latency_ms': 0, 'request_id': 'req-1', 'trace_id': 't-1', 'lock_wait_ms': 0}
    assert entries[1]['http_status'] == 422 and 'request_id' not in entries[1]
    assert 'secret' not in log_file.read_text(encoding='utf-8')
