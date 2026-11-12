"""Read-only Langfuse queries through SDK v2 observations and v3 scores. Langfuse off or any error -> []."""
import json
from datetime import datetime

from . import logs, tracing
from .tracing import ROOT_NAME


def _all_pages(fetch, **kwargs) -> list[dict]:
    rows, cursor = [], None
    while True:
        page = fetch(cursor=cursor, **{k: v for k, v in kwargs.items() if v is not None})
        rows += [item.dict() for item in page.data]
        cursor = page.meta.cursor
        if not cursor:
            return rows


def _query(endpoint: str, method: str, **kwargs) -> list[dict]:
    client = tracing.langfuse_client()
    if client is None:
        return []
    try:
        return _all_pages(getattr(getattr(client.api, endpoint), method), **kwargs)
    except Exception as exc:
        logs.log_event('warning', 'langfuse_query_failed', key=endpoint, error_type=type(exc).__name__)
        return []


def root_of(rows: list[dict]) -> dict:
    """The shipping_rag_query root among one trace's observations, or {}."""
    return next((row for row in rows if row.get('name') == ROOT_NAME), {})


def get_trace(trace_id: str) -> list[dict]:
    """Observations of one trace with io, metadata and usage, ordered by start time."""
    rows = _query('observations', 'get_many', trace_id=trace_id, fields='core,basic,io,metadata,usage,model,prompt,trace_context',
                  expand_metadata='config')
    return sorted(rows, key=lambda row: str(row.get('startTime') or ''))


def list_traces(*, run_id=None, tags=None, name=ROOT_NAME, since: datetime | None = None,
                where: list[dict] | None = None) -> list[dict]:
    """Root observations of matching traces, all pages. run_id filters by session; tags must all be present.

    where adds SDK filter conditions as given, e.g.
    [{"type": "stringObject", "column": "metadata", "key": "context_source", "operator": "=", "value": "fixed"}]."""
    # A `filter` overrides every query-parameter filter (SDK observations docs), so all conditions go into it.
    conditions = [{'type': 'boolean', 'column': 'isRootObservation', 'operator': '=', 'value': True},
                  *([{'type': 'string', 'column': 'name', 'operator': '=', 'value': name}] if name else []),
                  *([{'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': run_id}] if run_id else []),
                  *([{'type': 'datetime', 'column': 'startTime', 'operator': '>=', 'value': since.isoformat()}] if since else []),
                  *([{'type': 'arrayOptions', 'column': 'tags', 'operator': 'all of', 'value': list(tags)}] if tags else []),
                  *(where or [])]
    return _query('observations', 'get_many', name=name, session_id=run_id, from_start_time=since, is_root_observation=True,
                  fields='core,basic,metadata,trace_context', filter=json.dumps(conditions))


def get_scores(*, trace_id=None, name=None) -> list[dict]:
    """Scores, all pages, optionally for one trace or one score name."""
    return _query('scores_v3', 'get_many_v3', trace_id=trace_id, name=name)
