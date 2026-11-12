"""The architecture.md 1.1 call sequence against the fake client. Blueprint for the pipeline integration (2차)."""
import time

import pytest

from backend.app.artifacts import digest
from backend.app.config import Settings
from backend.app.context_engineering import prepare_context
from backend.app.context_engineering.diagnostics.runner import FixtureTokenizer, load_questions
from backend.app.context_engineering.prompts import PROMPT_VERSION, RAG_PROMPT
from backend.app.observability import generation_parameters, observation, query_trace, resolve_prompt
from backend.app.schemas import QueryResponse, ServiceError
from backend.tests.observability.fake_langfuse import FakeLangfuse

QUESTION = '보관료 무료기간은?'


def manifest_for(prompt, defense_addition=None):
    config = {'generation_llm_provider': 'ollama', 'local_llm_model': 'qwen3:4b', 'openai_model': 'gpt',
              'final_top_k': 4, 'context_max_tokens': 3000, 'embedding_profile': 'bge', 'retrieval_mode': 'hybrid'}
    config.update(prompt.config_fields, defense_sha256=digest(defense_addition) if defense_addition else None)
    return {'run_id': 'run-1', 'request_id': 'req-1', 'question_id': 'q1', 'entrypoint': 'cli',
            'config': config, 'config_hash': digest(config)}


def run_sequence(s, *, kind='comparison', fail_with=None):
    """The section 1.1 order: resolve_prompt, query_trace, with trace, prepare_context, record_context,
    observation(prompt=), finish, flush."""
    started = time.perf_counter()
    prompt = resolve_prompt('rag_answer', RAG_PROMPT, version=None, strict=kind != 'service')
    manifest = manifest_for(prompt)
    trace = query_trace(manifest, kind=kind, question=QUESTION, memory_enabled=False)
    try:
        with trace:
            prepared = prepare_context(s, QUESTION, load_questions()[0]['chunks'], tokenizer=FixtureTokenizer())
            trace.record_context(prepared, source='retrieval')
            with observation('answer_generation', kind='generation', input=prepared.messages, model='qwen3:4b',
                             prompt=prompt, model_parameters=generation_parameters(s),
                             metadata={'provider': s.generation_llm_provider, 'request_id': 'req-1'}) as generation:
                if generation:
                    generation.update(output='{}', usage_details={'input': 10, 'output': 5})
            if fail_with:
                raise fail_with
            response = QueryResponse(answer='답', status='answered', reason='근거', request_id='req-1', trace_id=trace.id,
                                     citations=[{'book_id': 'AS001', 'page': 1, 'chunk_id': 'c1'}],
                                     mode='hybrid', model='qwen3:4b', latency_ms=12)
            trace.finish(response)
        return trace, prepared
    finally:
        trace.flushed = trace.flush(started)


def test_one_trace_with_one_kind_tag_and_context_stage_children(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    trace, prepared = run_sequence(Settings(_env_file=None))

    root = fake.named('shipping_rag_query')
    assert fake.roots() == [root] and trace.id == root.trace_id
    assert root.tags == ['comparison', 'mode:hybrid', 'profile:bge', 'entrypoint:cli', f'prompt:{root.metadata["prompt_sha256"][:8]}']
    assert root.session_id == 'run-1' and root.trace_metadata == {'question_id': 'q1', 'request_id': 'req-1'}
    # Context owner's spans record under the fake (plan 5.1 P5) and belong to the same trace.
    for name in ('context_deduplication', 'context_budget', 'context_prompt'):
        assert fake.named(name).trace_id == root.trace_id
    assert all(s.tags == root.tags for s in fake.spans)
    assert fake.dropped == []


def test_root_metadata_holds_config_context_and_finish_fields(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    trace, prepared = run_sequence(Settings(_env_file=None))
    meta = fake.named('shipping_rag_query').metadata
    assert {k: meta[k] for k in ('top_k', 'generation_provider', 'generation_model', 'prompt_source', 'prompt_version')} == \
        {'top_k': 4, 'generation_provider': 'ollama', 'generation_model': 'qwen3:4b', 'prompt_source': 'local',
         'prompt_version': PROMPT_VERSION}
    assert meta['context_source'] == 'retrieval' and meta['context_count'] == len(prepared.contexts)
    assert meta['context_reduced'] is bool(prepared.dropped_chunk_ids) and meta['dropped_count'] == len(prepared.dropped_chunk_ids)
    assert (meta['status'], meta['citation_count'], meta['latency_ms']) == ('answered', 1, 12)
    assert fake.trace_io[0]['output']['status'] == 'answered'


def test_generation_span_carries_prompt_identity_and_model_parameters(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    run_sequence(Settings(_env_file=None))
    generation = fake.named('answer_generation')
    root = fake.named('shipping_rag_query')
    assert generation.metadata['prompt_source'] == 'local' and generation.metadata['prompt_version'] == PROMPT_VERSION
    assert generation.metadata['prompt_sha256'] == root.metadata['prompt_sha256']
    assert set(generation.fields['model_parameters']) == {'temperature', 'num_ctx', 'num_predict'}
    assert 'prompt' not in generation.fields   # local mode has no Langfuse prompt link
    assert fake.prompt_calls == []             # PROMPT_SOURCE defaults to local: no prompt fetch


def test_flush_reports_service_latency_before_flush(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    trace, _ = run_sequence(Settings(_env_file=None))
    assert set(trace.flushed) == {'service_latency_ms', 'flush_ms'} and fake.flushes == 1


def test_failure_writes_error_code_while_root_is_open_and_reraises(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    error = ServiceError('index_missing', 'no index')
    with pytest.raises(ServiceError):
        run_sequence(Settings(_env_file=None), fail_with=error)
    root = fake.named('shipping_rag_query')
    assert root.metadata['error_code'] == 'index_missing' and root.fields['level'] == 'ERROR'
    assert 'status' not in root.metadata and fake.dropped == []


def test_langfuse_off_runs_the_same_sequence_as_no_ops(monkeypatch):
    from backend.app.observability import tracing
    monkeypatch.setattr(tracing, 'langfuse_client', lambda: None)
    trace, prepared = run_sequence(Settings(_env_file=None), kind='service')
    assert trace.enabled is False and trace.id is None and prepared.contexts
    assert set(trace.flushed) == {'service_latency_ms', 'flush_ms'}
