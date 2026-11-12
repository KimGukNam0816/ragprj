import json
from contextlib import contextmanager

import pytest

from backend.app.context_engineering import validation
from backend.app.context_engineering.budget import generation_tokenizer
from backend.app.context_engineering.validation import OutputInvalid, validate_answer
from backend.app.schemas import ServiceError

SOURCE = {'book_id': 'AS001', 'page': 1, 'chunk_id': 'c1', 'text': '근거'}
CITE = {'book_id': 'AS001', 'page': 1, 'chunk_id': 'c1'}


def answer(**fields):
    return json.dumps({'answer': '내용 [AS001 p.1]', 'status': 'answered', 'reason': '근거', 'citations': [CITE], **fields})


CASES = [
    ('json', '{not json', 'Expecting'),
    ('duplicate_field', '{"answer":"a","answer":"b","status":"answered"}', 'Duplicate JSON field'),
    ('schema', answer(extra='x'), 'Extra inputs'),
    ('blank', answer(answer='  '), 'must not be blank'),
    ('citation_out_of_context', answer(citations=[{**CITE, 'chunk_id': 'other'}]), 'actual model context'),
    ('duplicate_citation', answer(citations=[CITE, CITE]), 'Duplicate citation'),
    ('missing_citation', answer(answer='내용', citations=[]), 'requires an actual source citation'),
    ('inline_mismatch', answer(answer='내용 [AS002 p.3]'), 'Inline citation'),
]


@pytest.mark.parametrize('reason,content,message', CASES, ids=[c[0] for c in CASES])
def test_each_check_has_its_reason_and_keeps_message_and_value_error(reason, content, message):
    with pytest.raises(ValueError, match=message) as raised:
        validate_answer(content, [SOURCE])
    error = raised.value
    assert isinstance(error, OutputInvalid) and isinstance(error, ServiceError)
    assert (error.reason, error.code, error.http_status) == (reason, 'output_invalid', 502)


def test_rejection_is_written_to_the_validation_span(monkeypatch):
    events = []

    @contextmanager
    def stage(name, **kwargs):
        class Span:
            def update(self, **data):
                events.append((name, data))
        yield Span()

    monkeypatch.setattr(validation, 'context_stage', stage)
    with pytest.raises(OutputInvalid):
        validate_answer(answer(citations=[CITE, CITE]), [SOURCE])
    assert events == [('context_output_validation', {'output': {'validation': 'rejected', 'reason': 'duplicate_citation'}})]


def test_unmapped_local_model_is_a_service_error_and_still_a_value_error():
    with pytest.raises(ValueError, match='tokenizer mapping') as raised:
        generation_tokenizer('ollama', 'llama3:8b')
    assert raised.value.code == 'tokenizer_unmapped' and raised.value.http_status == 502
