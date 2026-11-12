"""Review fix (thermos M1): validation failures must not leak model output to users or logs."""
import json
import logging

import pytest

from backend.app.context_engineering.validation import USER_MESSAGE, OutputInvalid, validate_answer
from backend.app.observability import logs

SECRET = 'SECRET-모델출력'
LEAKY = [
    json.dumps({'answer': SECRET, 'status': 'answered', 'reason': 'r', 'citations': [], 'extra': SECRET}),
    json.dumps([{'answer': SECRET}]),
    json.dumps({'answer': SECRET, 'status': SECRET, 'reason': 'r', 'citations': []}),
]


@pytest.mark.parametrize('content', LEAKY)
def test_schema_failure_keeps_model_output_out_of_message_detail_and_cause(content):
    with pytest.raises(OutputInvalid) as raised:
        validate_answer(content, [])
    error = raised.value
    assert error.reason == 'schema' and error.message == USER_MESSAGE
    assert SECRET not in str(error) and error.__cause__ is None and error.__suppress_context__


def test_query_failed_log_has_no_model_output(caplog):
    with pytest.raises(OutputInvalid) as raised:
        validate_answer(LEAKY[0], [])
    with caplog.at_level(logging.ERROR, logger='backend.observability'):
        logs.log_event('error', 'query_failed', exc=raised.value)
    stack = logs._JsonFormatter().format(caplog.records[-1])
    assert 'OutputInvalid' in stack and SECRET not in stack


def test_user_sees_the_old_generic_message(monkeypatch):
    with pytest.raises(OutputInvalid) as raised:
        validate_answer('{"answer":"a","status":"answered","reason":"r","citations":[]}', [])
    assert raised.value.message == USER_MESSAGE and 'requires an actual source citation' in str(raised.value)
