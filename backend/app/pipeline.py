import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone


from .artifacts import digest, file_hash, write_json
from .config import ROOT, Settings
from .retrieval.service import retrieve
from .observability import (ResolvedPrompt, generation_metadata, generation_parameters, observation, query_trace,
                            submit_live_judge, usage_details, verify_trace)
from .context_engineering.prompts import RAG_PROMPT
from .context_engineering import prepare_context, validate_answer
from .context_engineering.budget import generation_tokenizer
from .memory.preferences import NO_PREFERENCE, render, select_preferences
from .providers import Providers
from .schemas import ModelAnswer, QueryRequest, QueryResponse, ServiceError

# 실행 시작 시 소스 파일을 한 번 보존한다. 실행 중 디스크 변경과 기록을 구분한다.
SOURCE_FILES = [p for p in (ROOT/'backend/app').rglob('*.py') if '__pycache__' not in p.parts]
SOURCE_BYTES = {p.relative_to(ROOT/'backend/app').as_posix(): p.read_bytes() for p in sorted(SOURCE_FILES)}
SOURCE_HASHES={name:__import__('hashlib').sha256(data).hexdigest() for name,data in SOURCE_BYTES.items()}
LOCK_HASH=file_hash(ROOT/'uv.lock')


def gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'


def run_query(s: Settings, request: QueryRequest, *, entrypoint='cli', run_id=None, question_id=None, fixed_context=None, defense_addition=None, deadline=None, run_kind='service', prompt_version=None):
    request_started=time.perf_counter()
    s.check_inference_policy()
    if entrypoint not in {'cli', 'api', 'web'}:
        raise ValueError('Unknown entrypoint')
    if request.mode == 'direct':
        raise ServiceError('mode_not_supported', 'Use naive/hybrid for the PDF pipeline milestone')
    request_id = uuid.uuid4().hex
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + request_id[:8]
    if not run_id.replace('-', '').replace('_', '').isalnum() or len(run_id) > 100:
        raise ValueError('Invalid run ID')
    output = s.runtime_dir / 'runs' / run_id / request_id
    config = {k: getattr(s, k) for k in ['execution_stage', 'generation_llm_provider', 'local_llm_model', 'openai_model', 'embedding_model', 'comparison_embedding_model', 'dense_candidate_k', 'bm25_candidate_k', 'rrf_k', 'rerank_candidate_k', 'final_top_k', 'context_max_tokens', 'local_context_tokens', 'llm_max_output_tokens', 'llm_timeout_seconds', 'llm_max_retries', 'publish_metrics']}
    # TODO: use resolve_prompt('rag_answer', RAG_PROMPT, version=prompt_version, strict=run_kind != 'service')
    #       once prepare_context accepts prompt=, so the recorded prompt is always the template really used.
    prompt = ResolvedPrompt.local('rag_answer', RAG_PROMPT)
    config.update(prompt.config_fields, defense_sha256=digest(defense_addition) if defense_addition else None)
    config['output_schema_sha256'] = digest(ModelAnswer.model_json_schema())
    config['source_sha256'] = SOURCE_HASHES
    config['uv_lock_sha256'] = LOCK_HASH
    snapshot=s.runtime_dir/'code_snapshots'/digest(SOURCE_HASHES)
    snapshot.mkdir(parents=True,exist_ok=True)
    for name,data in SOURCE_BYTES.items():
        target=snapshot/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    config['embedding_profile'] = request.embedding_profile
    config['low_vram_mode']=s.low_vram_mode
    config['embedding_context_tokens']=s.embedding_context_tokens
    config['retrieval_mode'] = request.mode
    config['memory_backend'] = s.memory_backend
    if s.generation_llm_provider == 'ollama':
        import httpx
        tags = httpx.get(s.ollama_base_url.rstrip('/') + '/api/tags', timeout=15).json()['models']
        matching = [m for m in tags if m['name'] in {s.local_llm_model, s.local_llm_model + ':latest'}]
        if not matching:
            raise ServiceError('local_model_missing', 'Install the configured local generation model')
        config['generation_model_digest'] = matching[0]['digest']
    manifest = {'run_id': run_id, 'request_id': request_id, 'question_id': question_id, 'entrypoint': entrypoint, 'config': config, 'config_hash': digest(config), 'device': gpu_info(), 'platform': platform.platform(), 'metric_eligible': bool(s.publish_metrics and not s.smoke_run), 'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'manifest.json', manifest)
    provider = Providers(s)
    trace = query_trace(manifest, kind=run_kind, question=request.question, memory_enabled=request.memory_enabled)
    warning = [] if trace.enabled else ['Langfuse is not configured; tracing is incomplete']
    if s.prompt_source == 'langfuse':
        warning.append('PROMPT_SOURCE=langfuse is not wired into prepare_context yet; local prompt used')
    started = request_started
    provider.deadline=deadline or started+s.request_timeout_seconds
    record = {'request': request.model_dump(), 'warnings': warning}
    trace_id = None
    try:
        with trace:
            trace_id = trace.id
            record['trace_id'] = trace_id
            memory_text=NO_PREFERENCE
            if request.memory_enabled and not s.memory_enabled:
                # MEMORY_ENABLED는 서버 차단 스위치다. 요청 플래그로 다시 켜지 못한다.
                warning.append('Memory is disabled on this server; the request flag cannot enable it')
            if request.memory_enabled and s.memory_enabled:
                if s.memory_backend == 'dummy':
                    if s.publish_metrics:
                        raise ServiceError('dummy_memory_forbidden', 'Dummy preferences are for integration testing only', 422)
                    from .context_engineering.dummy_preferences import search_dummy_memory as search_memory
                    warning.append('Dummy preferences enabled: no Mem0 embedding/search/storage was called')
                else:
                    from .memory import search_memory
                try:
                    memories=search_memory(request.question,request.user_id)
                    record['memory_results']=memories
                    record['memory_backend']=s.memory_backend
                    # 원본 JSON(id·hash·score·created_at)이 아니라 표현 선호 문장만 넣는다.
                    selection=select_preferences(s,memories)
                    record['memory_selection']=selection
                    memory_text=render(selection['preferences'])
                    if selection['dropped']:
                        warning.append('Stored preferences were filtered before the prompt: '
                                       + ', '.join(sorted({d['reason'] for d in selection['dropped']})))
                except Exception as exc:
                    warning.append('Memory search unavailable: '+type(exc).__name__)
            if fixed_context is None:
                contexts, candidates, index = retrieve(s, provider, request.question, request.embedding_profile, request.mode)
            else:
                contexts, index = fixed_context
                contexts = [dict(c) for c in contexts]
                candidates = contexts
            record.update(candidates=candidates, index_fingerprint=index['fingerprint'], index_scope=index['scope'])
            manifest['index_fingerprint'] = index['fingerprint']
            if index['scope'] != 'full':
                if s.publish_metrics:
                    raise ServiceError('formal_index_scope','Formal evaluation requires the full corpus index')
                warning.append('Subset index: development smoke test only, not a full-corpus evaluation')
            prepared = prepare_context(s, request.question, contexts, memory_text=memory_text,
                                       defense_addition=defense_addition)
            trace.record_context(prepared, source='fixed' if fixed_context is not None else 'retrieval')
            contexts, messages = prepared.contexts, prepared.messages
            warning.extend(prepared.warnings)
            record.update(final_context=contexts, context_hash=prepared.context_hash,
                          actual_messages=messages, estimated_input_tokens=prepared.estimated_input_tokens,
                          dropped_chunk_ids=prepared.dropped_chunk_ids)
            if not contexts:
                answer = ModelAnswer(answer='검색된 자료에서 답변에 필요한 근거를 확인하지 못했습니다.', status='abstained', reason='검색 결과 없음', citations=[])
                generated = {'input_tokens': None, 'output_tokens': None, 'attempts': [], 'model': s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model}
            else:
                model = s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model
                with observation('answer_generation', kind='generation', input=messages, model=model, prompt=prompt, model_parameters=generation_parameters(s), metadata={'provider': s.generation_llm_provider, 'request_id': request_id}) as generation:
                    generated = provider.chat(messages, schema=ModelAnswer.model_json_schema())
                    record['generation'] = generated
                    if generation:
                        generation.update(output=generated['content'], usage_details=usage_details(generated),
                                          metadata=generation_metadata(generated, provider=s.generation_llm_provider,
                                                                       request_id=request_id))
                with observation('output_validation', input=generated['content']) as validation:
                    answer = validate_answer(generated['content'], contexts)
                    if validation:
                        validation.update(output=answer.model_dump())
            response = QueryResponse(**answer.model_dump(), request_id=request_id, trace_id=trace_id, mode=request.mode, model=generated['model'], latency_ms=round((time.perf_counter()-started)*1000), input_tokens=generated['input_tokens'], output_tokens=generated['output_tokens'], contexts=contexts, warnings=warning)
            record['response'] = response.model_dump()
            trace.finish(response)
            # 관측 담당(2026-11-12): LIVE_JUDGE_ENABLED일 때만 실서비스 답변을 OpenAI judge로 채점하도록 예약한다.
            # 응답은 기다리지 않고, 점수는 같은 trace에 붙는다. 꺼져 있으면 아무것도 하지 않는다.
            submit_live_judge(s, trace, entrypoint=entrypoint, question=request.question, response=record['response'],
                              contexts=contexts, output=output)
        return response, output
    except Exception as exc:
        record['error'] = {'code': getattr(exc, 'code', 'pipeline_error'), 'type': type(exc).__name__, 'message': getattr(exc, 'message', 'Pipeline failed; see saved steps'), 'reason': getattr(exc, 'reason', None), 'attempts': getattr(exc, 'attempts', [])}
        raise
    finally:
        record['external_inference_attempts'] = provider.external_inference_attempts
        record['latency_ms'] = round((time.perf_counter()-started)*1000)
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['trace_id'] = trace_id
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'result.json', record)
        record.update(trace.flush(started))
        # TODO: latency_ms below is recomputed after the flush and includes it; service_latency_ms is the flush-free value
        total_ms=round((time.perf_counter()-started)*1000)
        record['latency_ms']=total_ms
        if 'response' in record:
            response.latency_ms=total_ms
            record['response']['latency_ms']=total_ms
        write_json(output / 'result.json', record)
