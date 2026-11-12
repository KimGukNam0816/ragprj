"""Review fix (thermos, quality M2): log messages are fixed names; varying parts go to the key and error_type fields."""
import json
import logging

import pytest

from backend.app.observability import logs, publish_scores, record_score
from backend.app.observability import query
from backend.tests.observability.fake_langfuse import FakeLangfuse


@pytest.fixture
def records(caplog):
    caplog.set_level(logging.WARNING, logger='backend.observability')
    return caplog


def test_score_and_query_warnings_use_fixed_messages(monkeypatch, records):
    fake = FakeLangfuse().install(monkeypatch)
    monkeypatch.setattr(fake, 'create_score', lambda **s: (_ for _ in ()).throw(TimeoutError('slow')))
    publish_scores({'trace_id': 't', 'faithfulness_status': 'ok', 'faithfulness': 0.5}, run_id='r')
    record_score('t', 'made_up', 1)
    record_score('t', 'hit_at_k', 3)
    fake.api.observations.get_many = lambda **k: (_ for _ in ()).throw(ConnectionError('down'))
    query.get_trace('t')
    found = {(r.getMessage(), r.fields.get('key'), r.fields.get('error_type')) for r in records.records}
    assert {('score_send_failed', 'faithfulness', 'TimeoutError'), ('score_name_unknown', 'made_up', None),
            ('score_value_rejected', 'hit_at_k', None), ('langfuse_query_failed', 'observations', 'ConnectionError')} <= found
    assert not any(':' in r.getMessage() for r in records.records)


def test_key_and_error_type_reach_the_json_line(records):
    logs.log_event('warning', 'score_flush_failed', error_type='HTTPError', key='flush')
    line = json.loads(logs._JsonFormatter().format(records.records[-1]))
    assert (line['message'], line['error_type'], line['key']) == ('score_flush_failed', 'HTTPError', 'flush')
