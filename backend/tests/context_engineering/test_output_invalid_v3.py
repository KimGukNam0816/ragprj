"""Adversarial review fixes (interrogate): no model-chosen key names in error text, and every rejection is OutputInvalid."""
import json
import logging

import pytest

from backend.app.context_engineering.validation import OutputInvalid, validate_answer
from backend.app.observability import logs

KEY = 'MODEL-KEY-주민번호-900101'
GOOD = {'answer': 'a', 'status': 'abstained', 'reason': 'r', 'citations': []}


@pytest.mark.parametrize('content,reason', [
    ('{"answer":"a","%s":1,"%s":2}' % (KEY, KEY), 'duplicate_field'),
    (json.dumps({**GOOD, KEY: 1}), 'schema'),
    (json.dumps({**GOOD, 'citations': [{'book_id': 'B', 'page': 1, 'chunk_id': 'c', KEY: 1}]}), 'schema'),
])
def test_model_chosen_keys_stay_out_of_error_text_and_log_stack(content, reason, caplog):
    with pytest.raises(OutputInvalid) as raised:
        validate_answer(content, [{'book_id': 'B', 'page': 1, 'chunk_id': 'c'}])
    assert raised.value.reason == reason and KEY not in str(raised.value)
    with caplog.at_level(logging.ERROR, logger='backend.observability'):
        logs.log_event('error', 'query_failed', exc=raised.value)
    assert KEY not in logs._JsonFormatter().format(caplog.records[-1])


def test_known_schema_fields_are_still_named():
    with pytest.raises(OutputInvalid, match='status'):
        validate_answer(json.dumps({**GOOD, 'status': 'maybe'}), [])


@pytest.mark.parametrize('content', ['```' + json.dumps(GOOD) + '```', '```', '{"answer":' + '9' * 5000 + '}',
                                     '[' * 100000])
def test_formerly_untyped_rejections_are_output_invalid_json(content):
    with pytest.raises(OutputInvalid) as raised:
        validate_answer(content, [])
    assert raised.value.reason == 'json' and raised.value.code == 'output_invalid'


def test_a_fenced_answer_with_a_body_line_is_still_accepted():
    assert validate_answer('```json\n' + json.dumps(GOOD) + '\n```', []).status == 'abstained'
