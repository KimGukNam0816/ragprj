"""JSON line logs. Only the fields in LOG_FIELDS are written; there is no raw-text field."""
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

from ..config import ROOT

LOG_FIELDS = frozenset({'request_id', 'trace_id', 'run_id', 'error_code', 'error_reason', 'status', 'http_status',
                        'latency_ms', 'service_latency_ms', 'lock_wait_ms', 'flush_ms', 'method', 'path',
                        'key', 'error_type'})   # key: a field, score or endpoint name; error_type: an exception class name
LOG_DIR = ROOT / 'logs'

_logger = logging.getLogger('backend.observability')
_handlers: dict = {}
# One mutable dict per request. Sync endpoints run in a threadpool with a copy of the context,
# so bind() mutates this dict instead of calling _ctx.set, or values would not travel back.
_ctx: ContextVar[dict | None] = ContextVar('observability_log_context', default=None)
# CLI and batch runs never call setup_logging; without a handler Python's last-resort handler prints
# every warning and traceback to stderr.
_logger.addHandler(logging.NullHandler())


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        line = {'ts': datetime.fromtimestamp(record.created, timezone.utc).isoformat(), 'level': record.levelname.lower(),
                'logger': record.name, 'message': record.getMessage(), **getattr(record, 'fields', {})}
        if record.exc_info:
            line['stack'] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False, default=str)


def setup_logging(kind: str = 'api') -> None:
    """Write JSON lines to <repo>/logs/<kind>.log, rotated daily with 14 backups, INFO and above.

    The same handler is attached to the SDK `langfuse` logger. A second call adds no handler."""
    path = LOG_DIR / f'{kind}.log'
    handler = _handlers.get(path)
    if handler is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _handlers[path] = TimedRotatingFileHandler(path, when='midnight', backupCount=14, encoding='utf-8')
        handler.setLevel(logging.INFO)
        handler.setFormatter(_JsonFormatter())
    for logger in (_logger, logging.getLogger('langfuse')):
        if handler not in logger.handlers:
            logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def _allowed(fields: dict) -> dict:
    unknown = sorted(set(fields) - LOG_FIELDS)
    if unknown:
        _logger.warning('log_fields_dropped', extra={'fields': {'dropped_keys': unknown}})
    return {k: v for k, v in fields.items() if k in LOG_FIELDS and v is not None}


def log_event(level: str, message: str, *, exc: BaseException | None = None, **fields) -> None:
    """Write one JSON line. Fields outside LOG_FIELDS are dropped with a warning naming the keys.

    Fields bound to the current request (bind) are merged in. exc adds a `stack` field."""
    data = {**(_ctx.get() or {}), **_allowed(fields)}
    _logger.log(getattr(logging, level.upper()), message, extra={'fields': data},
                exc_info=(type(exc), exc, exc.__traceback__) if exc else None)


def bind(**fields) -> None:
    """Package-internal. Attach ids to the current request's log context. No-op outside a request scope."""
    current = _ctx.get()
    if current is not None:
        current.update(_allowed(fields))


def current() -> dict:
    """Package-internal. A copy of the fields bound to the current request."""
    return dict(_ctx.get() or {})


@contextmanager
def request_scope():
    """Package-internal. Reuse the middleware's request dict, or open a fresh one (CLI and batch runs)."""
    if _ctx.get() is not None:
        yield
        return
    token = _ctx.set({})
    try:
        yield
    finally:
        _ctx.reset(token)


async def access_log(request, call_next):
    """HTTP middleware: one entrance line per request with method, path, http_status, latency_ms.

    Ids bound while handling the request (request_id, trace_id, error_code, lock_wait_ms) are included.
    Query strings and bodies are never read."""
    started = time.perf_counter()
    token = _ctx.set({})
    status, error = 500, None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        error = exc
        raise
    finally:
        log_event('error' if error else 'info', 'request', exc=error, method=request.method, path=request.url.path,
                  http_status=status, latency_ms=round((time.perf_counter() - started) * 1000))
        _ctx.reset(token)


@contextmanager
def timed_lock(lock):
    """`with lock:` that also binds lock_wait_ms to the request log. Lock semantics are unchanged."""
    started = time.perf_counter()
    with lock:
        bind(lock_wait_ms=round((time.perf_counter() - started) * 1000))
        yield
