import time

import httpx

from .config import Settings
from .schemas import ServiceError


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

    def chat(self, messages, schema=None):
        self.s.check_inference_policy()
        attempts, result = [], None
        for attempt in range(self.s.llm_max_retries + 1):
            started = time.perf_counter()
            try:
                if self.s.generation_llm_provider == 'ollama':
                    model = self.s.local_llm_model
                    r = httpx.post(self.s.ollama_base_url.rstrip('/') + '/api/chat', json={'model': model, 'messages': messages, 'stream': False, 'think': False, 'format': schema or 'json', 'keep_alive':0 if self.s.low_vram_mode else '5m', 'options': {'temperature': 0, 'num_ctx': self.s.local_context_tokens, 'num_predict': self.s.llm_max_output_tokens}}, timeout=self.timeout(self.s.llm_timeout_seconds))
                    r.raise_for_status()
                    d = r.json()
                    result = {'content': d['message']['content'], 'input_tokens': d.get('prompt_eval_count'), 'output_tokens': d.get('eval_count'), 'model': model, 'finish_reason': d.get('done_reason')}
                attempts.append({'attempt': attempt + 1, 'latency_ms': round((time.perf_counter() - started) * 1000), 'input_tokens': result['input_tokens'], 'output_tokens': result['output_tokens'], 'status': 'ok'})
                result['attempts'] = attempts
                return result
            except httpx.HTTPError as exc:
                code = getattr(getattr(exc, 'response', None), 'status_code', None)
                attempts.append({'attempt': attempt + 1, 'status': 'error', 'http_status': code, 'input_tokens': None, 'output_tokens': None, 'latency_ms': round((time.perf_counter() - started) * 1000)})
                if attempt >= self.s.llm_max_retries or (code is not None and code not in {429, 500, 502, 503, 504}):
                    error = ServiceError('llm_error', 'Model request failed; see attempt metadata', 504 if isinstance(exc, httpx.TimeoutException) else 502)
                    error.attempts = attempts
                    raise error from None
        raise ServiceError('llm_error', 'No model response')
