"""Offline Langfuse client that imitates the SDK's recording rules (plan 9).

- Writes to a closed span are dropped (recorded in `dropped`), as the OTel exporter does.
- Nested propagate_attributes scopes merge tags, as SDK propagation.py:566-573 does.
- Propagated values over 200 chars or non-ASCII are dropped.
- Spans run inside a recording OTel span, so Context's context_stage records (plan 5.1 P5).
"""
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider


def _valid(value) -> bool:
    return isinstance(value, str) and value.isascii() and len(value) <= 200


class FakeSpan:
    def __init__(self, client, name, as_type, parent, trace_id, scope, fields):
        self.client, self.name, self.type, self.parent, self.trace_id = client, name, as_type, parent, trace_id
        self.tags, self.session_id, self.trace_metadata = list(scope['tags']), scope['session_id'], dict(scope['metadata'])
        self.metadata = dict(fields.pop('metadata', None) or {})
        self.fields = {k: v for k, v in fields.items() if v is not None}
        self.closed = False

    def update(self, *, metadata=None, **fields):
        if self.closed:
            self.client.dropped.append((self.name, {'metadata': metadata, **fields}))
            return self
        # Langfuse stores metadata per top-level key, so updates merge.
        self.metadata.update(metadata or {})
        self.fields.update({k: v for k, v in fields.items() if v is not None})
        return self


class FakeLangfuse:
    def __init__(self, *, prompts=None, flush_error=None, observations=None, scores=None, score_configs=()):
        self.spans: list[FakeSpan] = []
        self.scores: list[dict] = []
        self.datasets: list[str] = []
        self.dataset_items: list[dict] = []
        self.dropped: list = []
        self.trace_io: list[dict] = []
        self.flushes = 0
        self.flush_error = flush_error
        self.prompts = prompts or {}
        self.prompt_calls: list[dict] = []
        self.created_prompts: list[dict] = []
        self._current: ContextVar[FakeSpan | None] = ContextVar(f'fake_span_{id(self)}', default=None)
        self._scope: ContextVar[dict] = ContextVar(f'fake_scope_{id(self)}',
                                                   default={'tags': [], 'session_id': None, 'metadata': {}})
        self._tracer = TracerProvider().get_tracer('fake_langfuse')
        # Without given rows, the query API serves what this fake recorded (spans, and scores with the last write per id).
        self.api = SimpleNamespace(observations=_Pages(observations if observations is not None else self._exported_spans),
                                   scores_v3=_Pages(scores if scores is not None else self._exported_scores, method='get_many_v3'),
                                   score_configs=_ScoreConfigs(score_configs))

    def install(self, monkeypatch):
        from backend.app.observability import tracing
        monkeypatch.setattr(tracing, 'langfuse_client', lambda: self)
        monkeypatch.setattr(tracing, '_propagate', self.propagate_attributes)
        return self

    @contextmanager
    def propagate_attributes(self, *, tags=None, session_id=None, metadata=None, **_):
        outer = self._scope.get()
        merged = {'tags': outer['tags'] + [t for t in tags or [] if _valid(t) and t not in outer['tags']],
                  'session_id': session_id if _valid(session_id) else outer['session_id'],
                  'metadata': {**outer['metadata'], **{k: v for k, v in (metadata or {}).items() if _valid(str(v))}}}
        current = self._current.get()
        if current is not None and not current.closed:
            current.tags, current.session_id = merged['tags'], merged['session_id']
            current.trace_metadata = dict(merged['metadata'])
        token = self._scope.set(merged)
        try:
            yield
        finally:
            self._scope.reset(token)

    @contextmanager
    def start_as_current_observation(self, *, name, as_type='span', **fields):
        parent = self._current.get()
        span = FakeSpan(self, name, as_type, parent, parent.trace_id if parent else uuid.uuid4().hex, self._scope.get(), fields)
        self.spans.append(span)
        token = self._current.set(span)
        try:
            with self._tracer.start_as_current_span(name):
                yield span
        finally:
            span.closed = True
            self._current.reset(token)

    def get_current_trace_id(self):
        span = self._current.get()
        return span.trace_id if span else None

    def set_current_trace_io(self, *, input=None, output=None):
        span = self._current.get()
        if span is None or span.closed:
            self.dropped.append(('trace_io', {'input': input, 'output': output}))
            return
        self.trace_io.append({'trace_id': span.trace_id, 'input': input, 'output': output})

    def create_score(self, **score):
        self.scores.append(score)

    def create_dataset(self, *, name, **_):
        self.datasets.append(name)

    def create_dataset_item(self, *, dataset_name, **item):
        if dataset_name not in self.datasets:
            raise LookupError(f'dataset {dataset_name} not found')
        self.dataset_items.append({'dataset_name': dataset_name, **item})

    def flush(self):
        if self.flush_error:
            raise self.flush_error
        self.flushes += 1

    def get_prompt(self, name, **kwargs):
        self.prompt_calls.append({'name': name, **kwargs})
        found = self.prompts.get(name)
        if isinstance(found, Exception):
            raise found
        if found is None:
            from langfuse.api import Prompt_Chat
            from langfuse.model import ChatPromptClient
            return ChatPromptClient(Prompt_Chat(name=name, version=0, prompt=kwargs['fallback'], type='chat',
                                                config={}, labels=[], tags=[]), is_fallback=True)
        return found

    def create_prompt(self, **prompt):
        self.created_prompts.append(prompt)

    def get_trace_url(self, *, trace_id):
        return f'https://fake.langfuse/project/p1/traces/{trace_id}'

    def _exported_spans(self):
        return [{'id': str(i), 'traceId': s.trace_id, 'name': s.name, 'startTime': f'{i:06d}', 'sessionId': s.session_id,
                 'tags': s.tags, 'level': s.fields.get('level'), 'metadata': s.metadata,
                 'input': s.fields.get('input'), 'output': s.fields.get('output')} for i, s in enumerate(self.spans)]

    def _exported_scores(self):
        latest = {s['score_id']: s for s in self.scores}
        return [{'id': s['score_id'], 'traceId': s['trace_id'], 'name': s['name'], 'value': s['value']} for s in latest.values()]

    # helpers for assertions
    def named(self, name) -> FakeSpan:
        matches = [s for s in self.spans if s.name == name]
        assert len(matches) == 1, f'{name}: {len(matches)} spans'
        return matches[0]

    def roots(self) -> list[FakeSpan]:
        return [s for s in self.spans if s.parent is None]


class _Pages:
    """SDK list endpoint stand-in: rows are split into pages of 2 with a cursor."""

    def __init__(self, rows, method='get_many'):
        self.rows, self.calls = rows, []
        setattr(self, method, self._get)

    def _get(self, *, cursor=None, **kwargs):
        self.calls.append({'cursor': cursor, **kwargs})
        start = int(cursor or 0)
        keys = {'trace_id': 'traceId', 'name': 'name', 'session_id': 'sessionId'}   # SDK .dict() uses camelCase aliases
        source = self.rows() if callable(self.rows) else self.rows
        rows = [r for r in source if all(r.get(keys[k]) == v for k, v in kwargs.items() if k in keys and v)]
        page = rows[start:start + 2]
        next_cursor = str(start + 2) if start + 2 < len(rows) else None
        return SimpleNamespace(data=[SimpleNamespace(dict=lambda r=r: r) for r in page], meta=SimpleNamespace(cursor=next_cursor))


class _ScoreConfigs:
    """api.score_configs stand-in: page-based listing (one name per page) and create."""

    def __init__(self, names):
        self.names, self.created = list(names), []

    def get(self, *, page=1, limit=100):
        data = [SimpleNamespace(name=n) for n in self.names[page - 1:page]]
        return SimpleNamespace(data=data, meta=SimpleNamespace(total_pages=max(len(self.names), 1)))

    def create(self, **config):
        self.created.append(config)
