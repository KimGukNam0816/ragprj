import json
import platform
import subprocess
import time
import uuid
from functools import lru_cache
from datetime import datetime, timedelta, timezone

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .artifacts import digest, file_hash, write_json
from .config import ROOT, Settings
from .indexing import retrieve
from .observability import langfuse_client, observation
from .prompts import RAG_PROMPT
from .providers import Providers
from .schemas import ModelAnswer, QueryRequest, QueryResponse, ServiceError

# Capture the loaded implementation once. Later edits on disk must not masquerade
# as code that an already-running worker actually executed.
SOURCE_BYTES={name:(ROOT/'backend/app'/name).read_bytes() for name in ['config.py','prompts.py','providers.py','pipeline.py','ingestion.py','indexing.py','observability.py','schemas.py','memory.py','reranker.py','main.py']}
SOURCE_HASHES={name:__import__('hashlib').sha256(data).hexdigest() for name,data in SOURCE_BYTES.items()}
LOCK_HASH=file_hash(ROOT/'uv.lock')


def validate_answer(content: str, contexts):
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    answer = ModelAnswer.model_validate_json(text)
    valid = {(c['book_id'], c['page'], c['chunk_id']) for c in contexts}
    if any((c.book_id, c.page, c.chunk_id) not in valid for c in answer.citations):
        raise ValueError('Citation does not exist in actual model context')
    if answer.status == 'answered' and not answer.citations:
        raise ValueError('Answered result requires an actual source citation')
    return answer


def gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'


@lru_cache(maxsize=1)
def generation_tokenizer(provider='ollama', model=''):
    if provider == 'openai':
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # 미지원 모델은 근사값을 사용하고 실제 사용량은 API usage로 기록한다.
            return tiktoken.get_encoding('o200k_base')
    tokenizer = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3-4B', 'tokenizer.json'))
    tokenizer.no_truncation()
    return tokenizer


def run_query(s: Settings, request: QueryRequest, *, entrypoint='cli', run_id=None, question_id=None, fixed_context=None, defense_addition=None, deadline=None):
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
    config['prompt_sha256'] = digest(RAG_PROMPT.messages[0].prompt.template)
    config['output_schema_sha256'] = digest(ModelAnswer.model_json_schema())
    config['source_sha256'] = SOURCE_HASHES
    config['uv_lock_sha256'] = LOCK_HASH
    snapshot=s.runtime_dir/'code_snapshots'/digest(SOURCE_HASHES)
    snapshot.mkdir(parents=True,exist_ok=True)
    for name,data in SOURCE_BYTES.items():
        target=snapshot/name
        if not target.exists():target.write_bytes(data)
    config['embedding_profile'] = request.embedding_profile
    config['low_vram_mode']=s.low_vram_mode
    config['embedding_context_tokens']=s.embedding_context_tokens
    config['retrieval_mode'] = request.mode
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
    client = langfuse_client()
    warning = [] if client else ['Langfuse is not configured; tracing is incomplete']
    started = request_started
    provider.deadline=deadline or started+s.request_timeout_seconds
    record = {'request': request.model_dump(), 'warnings': warning}
    trace_id = None
    try:
        with observation('shipping_rag_query', input={'question': request.question}, metadata={**manifest, 'memory_enabled': request.memory_enabled}) as root:
            trace_id = client.get_current_trace_id() if client else None
            record['trace_id'] = trace_id
            memory_text='사용하지 않음'
            if request.memory_enabled:
                from .memory import search_memory
                try:
                    memories=search_memory(request.question,request.user_id)
                    record['memory_results']=memories
                    memory_text=json.dumps(memories,ensure_ascii=False)[:1500]
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
            # Qwen3-4B uses its actual tokenizer to bound the final prompt in local mode.
            tokenizer = generation_tokenizer(s.generation_llm_provider, s.openai_model)
            while True:
                # Only source identifiers and body belong in the model context, not index internals.
                context = json.dumps([{k: c[k] for k in ['book_id', 'page', 'chunk_id', 'text']} for c in contexts], ensure_ascii=False)
                messages = [{'role': 'system' if m.type == 'system' else 'user', 'content': m.content} for m in RAG_PROMPT.format_messages(question=request.question, memory=memory_text, context=context)]
                if defense_addition:
                    messages[0]['content'] += '\n' + defense_addition
                tokenized = tokenizer.encode('\n'.join(m['content'] for m in messages))
                approximate_tokens = len(tokenized.ids if s.generation_llm_provider == 'ollama' else tokenized) + 256
                budget = min(s.context_max_tokens, s.local_context_tokens-s.llm_max_output_tokens) if s.generation_llm_provider == 'ollama' else s.context_max_tokens
                if approximate_tokens <= budget:
                    break
                if not contexts:
                    raise ServiceError('context_budget', 'Question/prompt exceeds token budget', 422)
                contexts.pop()
                warning.append('A lower-ranked chunk was dropped to satisfy the input budget')
            record.update(final_context=contexts, context_hash=digest(contexts), actual_messages=messages, estimated_input_tokens=approximate_tokens)
            if not contexts:
                answer = ModelAnswer(answer='검색된 자료에서 답변에 필요한 근거를 확인하지 못했습니다.', status='abstained', reason='검색 결과 없음', citations=[])
                generated = {'input_tokens': None, 'output_tokens': None, 'attempts': [], 'model': s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model}
            else:
                model = s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model
                with observation('answer_generation', kind='generation', input=messages, model=model, metadata={'provider': s.generation_llm_provider, 'request_id': request_id}) as generation:
                    generated = provider.chat(messages, schema=ModelAnswer.model_json_schema())
                    record['generation'] = generated
                    if generation:
                        generation.update(output=generated['content'], usage_details={k: v for k, v in {'input': generated['input_tokens'], 'output': generated['output_tokens']}.items() if v is not None},
                                          metadata={'provider': s.generation_llm_provider, 'request_id': request_id,
                                                    **{k:v for k,v in generated.items() if k != 'content'}})
                with observation('output_validation', input=generated['content']) as validation:
                    answer = validate_answer(generated['content'], contexts)
                    if validation:
                        validation.update(output=answer.model_dump())
            response = QueryResponse(**answer.model_dump(), request_id=request_id, trace_id=trace_id, mode=request.mode, model=generated['model'], latency_ms=round((time.perf_counter()-started)*1000), input_tokens=generated['input_tokens'], output_tokens=generated['output_tokens'], contexts=contexts, warnings=warning)
            record['response'] = response.model_dump()
            if root:
                root.update(output=response.model_dump())
            if client:
                client.set_current_trace_io(input={'question': request.question}, output=response.model_dump())
        return response, output
    except Exception as exc:
        record['error'] = {'code': getattr(exc, 'code', 'pipeline_error'), 'type': type(exc).__name__, 'message': getattr(exc, 'message', 'Pipeline failed; see saved steps'), 'attempts': getattr(exc, 'attempts', [])}
        raise
    finally:
        record['external_inference_attempts'] = provider.external_inference_attempts
        record['latency_ms'] = round((time.perf_counter()-started)*1000)
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['trace_id'] = trace_id
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'result.json', record)
        if client:
            try:client.flush()
            except Exception:
                record.setdefault('warnings',[]).append('Langfuse flush failed; Cloud verification required')
        total_ms=round((time.perf_counter()-started)*1000)
        record['latency_ms']=total_ms
        if 'response' in record:
            response.latency_ms=total_ms
            record['response']['latency_ms']=total_ms
        write_json(output / 'result.json', record)


def verify_trace(s, trace_id, request_id, started_at=None):
    """Read back exported trace, do not equate flush success with ingestion."""
    if not trace_id:
        return {'verified': False, 'reason': 'No trace ID'}
    import httpx
    end = datetime.now(timezone.utc)
    start=datetime.fromisoformat(started_at) if started_at else end-timedelta(hours=1)
    query_end=min(end,start+timedelta(hours=1))
    params = {'traceId': trace_id, 'fields': 'core,basic,usage,io,metadata', 'fromStartTime': (start-timedelta(minutes=1)).isoformat(), 'toStartTime': query_end.isoformat(), 'limit': 100}
    last_status = None
    for attempt in range(5):
        try:
            r = httpx.get(s.langfuse_base_url.rstrip('/') + '/api/public/v2/observations', params=params, auth=(s.langfuse_public_key, s.langfuse_secret_key.get_secret_value()), timeout=15)
            last_status = r.status_code
            if r.is_success:
                data = r.json()
                rows = [x for x in data.get('data', []) if x.get('traceId') == trace_id]
                names = {x.get('name') for x in rows}
                generation = next((x for x in rows if x.get('name') == 'answer_generation'), {})
                required = {'shipping_rag_query', 'query_embedding', 'dense_retrieval', 'answer_generation', 'output_validation'}
                io_available = bool(generation.get('input') and generation.get('output'))
                verified = request_id in json.dumps(rows) and required <= names and io_available
                if verified:
                    return {'verified': True, 'api': 'v2/observations', 'trace_id': trace_id, 'observation_count': len(rows), 'names': sorted(names), 'generation_io_available': io_available}
            if r.status_code == 429:
                delay = min(10, max(2, int(r.headers.get('retry-after', '2'))))
                time.sleep(delay)
                continue
        except httpx.HTTPError:
            pass
        time.sleep(4)
    return {'verified': False, 'trace_id': trace_id, 'http_status': last_status, 'reason': 'Required observations not readable yet; verify in Langfuse'}

