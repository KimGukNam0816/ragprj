import time

import httpx

from .config import Settings
from .schemas import ServiceError


def ollama_models(s: Settings) -> list[dict]:
    """설치된 Ollama 모델 목록. 서버가 꺼져 있으면 원인을 바로 알 수 있는 메시지로 바꾼다."""
    try:
        r = httpx.get(s.ollama_base_url.rstrip('/') + '/api/tags', timeout=15)
        r.raise_for_status()
        return r.json()['models']
    except (httpx.HTTPError, ValueError, KeyError):
        raise ServiceError('ollama_unavailable',
                           f'Ollama 서버({s.ollama_base_url})에 연결하지 못했습니다. '
                           'Ollama 앱을 실행했는지, OLLAMA_BASE_URL 주소가 맞는지 확인하세요.') from None


class Providers:
    def __init__(self, s: Settings):
        self.s = s
        self.external_inference_attempts = 0
        self.deadline = None

    def timeout(self, default):
        if self.deadline is None:
            return default
        remaining=self.deadline-time.perf_counter()
        if remaining<=0:
            raise ServiceError('request_timeout','전체 요청 처리 시간이 초과됐습니다.',504)
        return min(default,remaining)

    def embed(self, texts: list[str], profile: str, *, query=False):
        model, dimensions, _ = self.s.profile(profile)
        if query and profile == 'qwen':
            texts = ['Instruct: Given a Korean shipping question, retrieve passages that answer the question.\nQuery: ' + t for t in texts]
        try:
            payload={'model': model, 'input': texts, 'truncate': False, 'keep_alive': 0 if self.s.low_vram_mode else '5m'}
            if self.s.embedding_context_tokens:
                payload['options']={'num_ctx':self.s.embedding_context_tokens}
            r = httpx.post(self.s.ollama_base_url.rstrip('/') + '/api/embed', json=payload, timeout=self.timeout(120))
            r.raise_for_status()
            vectors = r.json()['embeddings']
            import math
            if len(vectors) != len(texts) or any(len(v) != dimensions or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError('Embedding shape or values invalid')
            return vectors
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise ServiceError('embedding_error', 'Ollama embedding failed: ' + type(e).__name__) from None

    def _chat_once(self, messages, schema):
        if self.s.generation_llm_provider == 'openai':
            headers = {'Authorization': 'Bearer ' + self.s.openai_api_key.get_secret_value()}
            if self.s.openai_org_id:
                headers['OpenAI-Organization'] = self.s.openai_org_id
            if self.s.openai_project_id:
                headers['OpenAI-Project'] = self.s.openai_project_id
            payload = {'model': self.s.openai_model, 'messages': messages,
                       'max_completion_tokens': self.s.llm_max_output_tokens, 'stream': False}
            if schema:
                import copy
                schema = copy.deepcopy(schema)
                def strict_objects(node):
                    if isinstance(node, dict):
                        node.pop('default', None)
                        if node.get('type') == 'object':
                            node['additionalProperties'] = False
                            node['required'] = list(node.get('properties', {}))
                        for value in node.values():
                            strict_objects(value)
                    elif isinstance(node, list):
                        for value in node:
                            strict_objects(value)
                strict_objects(schema)
                payload['response_format'] = {'type': 'json_schema', 'json_schema':
                    {'name': 'rag_answer', 'strict': True, 'schema': schema}}
            self.external_inference_attempts += 1
            r = httpx.post(self.s.openai_base_url.rstrip('/') + '/chat/completions',
                           headers=headers, json=payload, timeout=self.timeout(self.s.llm_timeout_seconds))
            r.raise_for_status()
            d = r.json()
            choice = d['choices'][0]
            usage = d.get('usage') or {}
            return {'content': choice['message'].get('content') or '',
                    'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                    'model': d.get('model', self.s.openai_model), 'finish_reason': choice.get('finish_reason'),
                    'provider_request_id': r.headers.get('x-request-id'), 'response_id': d.get('id'),
                    'http_status': r.status_code, 'usage': usage}
        model = self.s.local_llm_model
        r = httpx.post(self.s.ollama_base_url.rstrip('/') + '/api/chat', json={
            'model': model, 'messages': messages, 'stream': False, 'think': False,
            'format': schema or 'json', 'keep_alive': 0 if self.s.low_vram_mode else '5m',
            'options': {'temperature': 0, 'num_ctx': self.s.local_context_tokens,
                        'num_predict': self.s.llm_max_output_tokens}},
            timeout=self.timeout(self.s.llm_timeout_seconds))
        r.raise_for_status()
        d = r.json()
        timing = {k.replace('_duration', '_seconds'): d[k]/1e9 for k in
                  ('load_duration', 'prompt_eval_duration', 'eval_duration', 'total_duration') if d.get(k) is not None}
        if d.get('eval_duration') and d.get('eval_count') is not None:
            timing['output_tokens_per_second'] = d['eval_count']/(d['eval_duration']/1e9)
        return {'content': d['message']['content'], 'input_tokens': d.get('prompt_eval_count'),
                'output_tokens': d.get('eval_count'), 'model': model, 'finish_reason': d.get('done_reason'),
                'http_status': r.status_code, 'timing': timing}

    def chat(self, messages, schema=None):
        from .observability import observation
        self.s.check_inference_policy()
        attempts = []
        model = self.s.openai_model if self.s.generation_llm_provider == 'openai' else self.s.local_llm_model
        for attempt in range(self.s.llm_max_retries + 1):
            started = time.perf_counter()
            # 시도별로 기록하여 재시도 시간과 사용량을 구분한다. 비밀 헤더는 기록하지 않는다.
            with observation('llm_attempt', metadata={'attempt': attempt+1,
                             'provider': self.s.generation_llm_provider, 'model': model}) as span:
                try:
                    self.timeout(self.s.llm_timeout_seconds)
                    result = self._chat_once(messages, schema)
                    info = {k: v for k, v in result.items() if k != 'content'}
                    info.update(attempt=attempt+1, status='ok', latency_ms=round((time.perf_counter()-started)*1000))
                    attempts.append(info)
                    if span:
                        span.update(metadata=info)
                    result['attempts'] = attempts
                    result['latency_ms'] = sum(a['latency_ms'] for a in attempts)
                    return result
                except (httpx.HTTPError, ServiceError) as exc:
                    response = getattr(exc, 'response', None)
                    code = response.status_code if response is not None else None
                    info = {'attempt': attempt+1, 'status': 'error', 'http_status': code,
                            'error_type': type(exc).__name__, 'error_code': getattr(exc, 'code', None),
                            'provider_request_id': response.headers.get('x-request-id') if response is not None else None,
                            'input_tokens': None, 'output_tokens': None,
                            'latency_ms': round((time.perf_counter()-started)*1000)}
                    attempts.append(info)
                    if span:
                        span.update(level='ERROR', status_message=info['error_type'], metadata=info)
                    if isinstance(exc, ServiceError):
                        exc.attempts = attempts
                        raise
                    if attempt >= self.s.llm_max_retries or (code is not None and code not in {429, 500, 502, 503, 504}):
                        error = ServiceError('llm_error', 'Model request failed; see attempt metadata',
                                             504 if isinstance(exc, httpx.TimeoutException) else 502)
                        error.attempts = attempts
                        raise error from None
        raise ServiceError('llm_error', 'No model response')
