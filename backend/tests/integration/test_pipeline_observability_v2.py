"""Review fix (thermos, quality H2): the checks that only test_pipeline_usage.py made, now against the real run_query."""
import json

import pytest

from backend.app.schemas import QueryRequest
from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse

REQUEST = QueryRequest(question='면제 조건은?', mode='hybrid', embedding_profile='bge')


@pytest.fixture
def fake(monkeypatch):
    return FakeLangfuse().install(monkeypatch)


@pytest.fixture
def duplicated_citations(offline_pipeline, monkeypatch):
    pipeline, _ = offline_pipeline
    real_chat = pipeline.Providers.chat

    def chat(self, messages, schema=None):
        result = real_chat(self, messages, schema)
        data = json.loads(result['content'])
        data['citations'] *= 2
        return {**result, 'content': json.dumps(data)}

    monkeypatch.setattr(pipeline.Providers, 'chat', chat)


def test_root_has_exact_tags_trace_metadata_and_context_children(offline_pipeline, fake):
    pipeline, s = offline_pipeline
    pipeline.run_query(s, REQUEST, run_id='offline-v2', question_id='q1', run_kind='comparison')
    root = fake.named('shipping_rag_query')
    assert fake.roots() == [root]
    assert root.tags == ['comparison', 'mode:hybrid', 'profile:bge', 'entrypoint:cli', f'prompt:{root.metadata["prompt_sha256"][:8]}']
    assert root.trace_metadata == {'question_id': 'q1', 'request_id': root.metadata['request_id']}
    for name in ('context_deduplication', 'context_budget', 'context_prompt', 'context_output_validation'):
        assert fake.named(name).trace_id == root.trace_id
    assert all(span.tags == root.tags for span in fake.spans)
    assert set(fake.named('answer_generation').fields['model_parameters']) == {'max_completion_tokens'}


def test_langfuse_off_still_answers_and_records_flush_fields(offline_pipeline):
    pipeline, s = offline_pipeline
    response, output = pipeline.run_query(s, REQUEST, run_id='offline-v2')
    record = json.loads((output/'result.json').read_text(encoding='utf-8'))
    assert response.status == 'answered' and response.trace_id is None
    assert {'service_latency_ms', 'flush_ms'} <= set(record) and 'flush_error' not in record
    assert 'Langfuse is not configured; tracing is incomplete' in record['warnings']


def test_rejected_output_is_on_the_validation_span_and_user_message_is_generic(offline_pipeline, fake, duplicated_citations):
    from backend.app.context_engineering.validation import USER_MESSAGE, OutputInvalid
    pipeline, s = offline_pipeline
    with pytest.raises(OutputInvalid) as raised:
        pipeline.run_query(s, REQUEST, run_id='offline-v2')
    assert raised.value.message == USER_MESSAGE
    assert fake.named('context_output_validation').fields['output'] == {'validation': 'rejected', 'reason': 'duplicate_citation'}
    saved = json.loads(next((s.runtime_dir/'runs'/'offline-v2').glob('*/result.json')).read_text(encoding='utf-8'))
    assert saved['error']['message'] == USER_MESSAGE
