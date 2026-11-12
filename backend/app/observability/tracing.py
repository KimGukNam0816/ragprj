import hashlib
import time
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Literal, get_args

from ..config import get_settings
from . import logs

RunKind = Literal['service', 'comparison', 'evaluation', 'security', 'diagnostic']
RUN_KINDS = frozenset(get_args(RunKind))
ROOT_NAME = 'shipping_rag_query'


@lru_cache
def langfuse_client():
    """The single Langfuse client, or None when tracing is disabled or keys are missing."""
    s = get_settings()
    if not (s.langfuse_enabled and s.langfuse_base_url and s.langfuse_public_key and s.langfuse_secret_key.get_secret_value()):
        return None
    from langfuse import Langfuse
    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key.get_secret_value(), base_url=s.langfuse_base_url)


def _propagate(**fields):
    # propagate_attributes is module-level in the SDK, so tests replace this function together with langfuse_client.
    from langfuse import propagate_attributes
    return propagate_attributes(**fields)


@contextmanager
def observation(name: str, *, kind='span', input=None, metadata=None, model=None, prompt=None, model_parameters=None):
    """Open a child span. Yields the span, or None when tracing is off.

    prompt (ResolvedPrompt) links a generation to its Langfuse prompt version and merges prompt.config_fields
    into metadata. Pass it only on the answer generation."""
    client = langfuse_client()
    if client is None:
        yield None
        return
    extra = {}
    if prompt is not None:
        metadata = {**(metadata or {}), **prompt.config_fields}
        if prompt._link is not None:
            extra['prompt'] = prompt._link
    if model_parameters is not None:
        extra['model_parameters'] = model_parameters
    with client.start_as_current_observation(name=name, as_type=kind, input=input, metadata=metadata, model=model, **extra) as span:
        yield span


def _safe(key: str, value) -> str | None:
    """Propagated values must be ASCII and at most 200 chars or the SDK drops them silently."""
    if value is None:
        return None
    text = str(value)
    if not text.isascii() or len(text) > 200:
        logs.log_event('warning', 'propagated_value_dropped', key=key)
        return None
    return text


def _session_id(run_id) -> str | None:
    """run_id, or `run-<sha8>` when the SDK would drop it (non-ASCII or over 200 chars). The root keeps the real run_id."""
    if run_id is None:
        return None
    text = str(run_id)
    if text.isascii() and len(text) <= 200:
        return text
    return 'run-' + hashlib.sha256(text.encode()).hexdigest()[:8]


def _scope_fields(manifest: dict, kind: str, variant: str | None = None) -> tuple[list[str], str | None, dict]:
    """Tags, session id and trace metadata for one trace. Pure apart from warning logs; never raises."""
    config = manifest.get('config') or {}
    sha = config.get('prompt_sha256')
    labels = {'mode': config.get('retrieval_mode'), 'profile': config.get('embedding_profile'),
              'entrypoint': manifest.get('entrypoint'), 'variant': variant, 'prompt': sha[:8] if sha else None}
    tags = [kind] + [tag for key, value in labels.items() if value is not None and (tag := _safe(key, f'{key}:{value}'))]
    session = _session_id(manifest.get('run_id')) if kind != 'service' else None
    metadata = {key: v for key in ('question_id', 'request_id') if (v := _safe(key, manifest.get(key))) is not None}
    return tags, session, metadata


def _root_config(manifest: dict) -> dict:
    """Flat root metadata keys from manifest['config'], named for dashboard filters."""
    c = manifest.get('config') or {}
    fields = {'top_k': c.get('final_top_k'), 'context_max_tokens': c.get('context_max_tokens'),
              'generation_provider': c.get('generation_llm_provider'),
              'generation_model': c.get('local_llm_model') if c.get('generation_llm_provider') == 'ollama' else c.get('openai_model'),
              **{k: c.get(k) for k in ('embedding_profile', 'retrieval_mode', 'prompt_sha256', 'prompt_version', 'prompt_source')}}
    return {k: v for k, v in fields.items() if v is not None}


class QueryTrace:
    """One `shipping_rag_query` trace: one scope and one root span, opened together by `with trace:`.

    Exactly one kind tag per trace, because scope and root are opened by the same __enter__.
    Root writes happen while the span is open; finish() or record_context() after __exit__ log a warning.
    With Langfuse off every method is a no-op with the same return types."""

    def __init__(self, manifest: dict, kind: str, question: str, memory_enabled: bool, variant: str | None):
        self._manifest, self._kind, self._question = manifest, kind, question
        self._memory_enabled, self._variant = memory_enabled, variant
        self.enabled = langfuse_client() is not None
        self.id = None
        self._root = None
        self._stack = None
        self._closed = False
        self._log_fields: dict = {}

    @property
    def kind(self) -> str:
        return self._kind

    def __enter__(self) -> 'QueryTrace':
        self._stack = ExitStack()
        self._stack.enter_context(logs.request_scope())
        logs.bind(request_id=self._manifest.get('request_id'), run_id=self._manifest.get('run_id'))
        client = langfuse_client()
        if client is not None:
            tags, session, metadata = _scope_fields(self._manifest, self._kind, self._variant)
            self._stack.enter_context(_propagate(tags=tags, session_id=session, metadata=metadata))
            self._root = self._stack.enter_context(client.start_as_current_observation(
                name=ROOT_NAME, input={'question': self._question},
                metadata={**self._manifest, 'memory_enabled': self._memory_enabled, **_root_config(self._manifest)}))
            self.id = client.get_current_trace_id()
            logs.bind(trace_id=self.id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc is not None:
                code, reason = getattr(exc, 'code', 'pipeline_error'), getattr(exc, 'reason', None)
                if self._root is not None:
                    self._root.update(level='ERROR', metadata={'error_code': code, **({'error_reason': reason} if reason else {})})
                logs.bind(error_code=code, error_reason=reason)
                logs.log_event('error', 'query_failed', exc=exc)
        finally:
            # flush() runs in the caller's `finally`, after the log scope closes, so keep what was bound.
            self._log_fields = logs.current()
            self._closed = True
            self._stack.__exit__(exc_type, exc, tb)
        return False

    def _update_root(self, **fields) -> bool:
        if self._closed:
            logs.log_event('warning', 'trace_write_after_close', trace_id=self.id)
            return False
        if self._root is not None:
            self._root.update(**fields)
        return True

    def record_context(self, prepared, *, source: Literal['retrieval', 'fixed']) -> None:
        """Write context_source, context_count, context_reduced, dropped_count to root metadata. No span."""
        self._update_root(metadata={'context_source': source, 'context_count': len(prepared.contexts),
                                    'context_reduced': bool(prepared.dropped_chunk_ids),
                                    'dropped_count': len(prepared.dropped_chunk_ids)})

    def finish(self, response) -> None:
        """Success path, inside the with. Root output, trace IO, and status, citation_count, latency_ms (pre-flush)."""
        data = response.model_dump()
        if not self._update_root(output=data, metadata={'status': data['status'], 'citation_count': len(data['citations']),
                                                        'latency_ms': data['latency_ms']}):
            return
        logs.bind(status=data['status'])
        if self._root is not None:
            langfuse_client().set_current_trace_io(input={'question': self._question}, output=data)

    def flush(self, started: float) -> dict:
        """Call in `finally`. Returns service_latency_ms (measured before the flush) and flush_ms,
        plus flush_error (exception type name) on failure. Logs `query_done`. Never raises."""
        result = {'service_latency_ms': round((time.perf_counter() - started) * 1000)}
        client = langfuse_client()
        flush_started = time.perf_counter()
        if client is not None:
            try:
                client.flush()
            except Exception as exc:
                result['flush_error'] = type(exc).__name__
        result['flush_ms'] = round((time.perf_counter() - flush_started) * 1000)
        logs.log_event('warning' if 'flush_error' in result else 'info', 'query_done', **self._log_fields,
                       service_latency_ms=result['service_latency_ms'], flush_ms=result['flush_ms'])
        return result


def query_trace(manifest: dict, *, kind: RunKind = 'service', question: str,
                memory_enabled: bool = False, variant: str | None = None) -> QueryTrace:
    """Build the handle for one service-query trace. Opens nothing yet.

    Raises ValueError only for an unknown kind (programmer error, before any work starts)."""
    if kind not in RUN_KINDS:
        raise ValueError(f'Unknown run kind: {kind}')
    return QueryTrace(manifest, kind, question, memory_enabled, variant)


def _ragas_version() -> str | None:
    try:
        return package_version('ragas')
    except PackageNotFoundError:
        return None


@contextmanager
def evaluation_trace(*, run_id: str, question_id: str, mode: str, profile: str, source: dict | None):
    """One `quality_evaluation` trace per scored item, kind tag `evaluation`, session = run_id.

    Only request_id and trace_id are read from source (the service response dict). Yields the root span or None."""
    client = langfuse_client()
    if client is None:
        yield None
        return
    source = source or {}
    with _evaluation_root(client, 'quality_evaluation', 'evaluation', run_id, question_id, mode, profile,
                          {'source_request_id': source.get('request_id'), 'source_trace_id': source.get('trace_id'),
                           'ragas_version': _ragas_version()}) as root:
        yield root


@contextmanager
def security_evaluation_trace(*, run_id: str, task_id: str, mode: str, profile: str, source_trace_ids: list):
    """One `security_evaluation` trace per security task, kind tag `security`, session = run_id.

    Groups the post-hoc judge calls for the task's base and defense answers (open a `security_judge` observation per
    call inside it), so they are not orphan traces. source_trace_ids are the judged answers' service traces.
    Attack metadata may appear below this root (it is a judge trace, not a service trace). Yields the root or None."""
    client = langfuse_client()
    if client is None:
        yield None
        return
    with _evaluation_root(client, 'security_evaluation', 'security', run_id, task_id, mode, profile,
                          {'source_trace_ids': [t for t in source_trace_ids if t]}) as root:
        yield root


@contextmanager
def _evaluation_root(client, name, kind, run_id, question_id, mode, profile, root_metadata):
    manifest = {'run_id': run_id, 'question_id': question_id, 'config': {'retrieval_mode': mode, 'embedding_profile': profile}}
    tags, session, metadata = _scope_fields(manifest, kind)
    with _propagate(tags=tags, session_id=session, metadata=metadata), \
            client.start_as_current_observation(name=name, metadata=root_metadata) as root:
        yield root


def trace_url(trace_id: str | None) -> str | None:
    """Langfuse UI link for a trace. The SDK caches the project id; any failure returns None."""
    client = langfuse_client()
    if client is None or not trace_id:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None
