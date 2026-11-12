import json
import time

from . import query
from .tracing import ROOT_NAME

RETRIEVAL_SPANS = frozenset({'query_embedding', 'dense_retrieval'})
CONTEXT_SPANS = frozenset({'context_budget', 'context_prompt'})   # span names from context_engineering/builder.py
ANSWER_SPANS = frozenset({'answer_generation', 'output_validation'})


def required_spans(context_source: str, context_count: int) -> frozenset[str]:
    """Spans a complete trace must have: retrieval, fixed-context, and zero-context paths."""
    spans = {ROOT_NAME, *CONTEXT_SPANS}
    if context_source == 'retrieval':
        spans |= RETRIEVAL_SPANS
    if context_count:
        spans |= ANSWER_SPANS
    return frozenset(spans)


def verify_trace(s, trace_id, request_id, started_at=None, *, attempts=5, wait_seconds=4):
    """Read the exported trace back; flush success is not ingestion. Retries while Langfuse is still ingesting.

    The root's context_source and context_count choose the required spans. Traces written before
    record_context existed have neither and are checked as the retrieval path. s and started_at are kept for callers."""
    if not trace_id:
        return {'verified': False, 'reason': 'No trace ID'}
    missing = []
    for attempt in range(attempts):
        rows = query.get_trace(trace_id)
        names = {row.get('name') for row in rows}
        root = query.root_of(rows)
        metadata = root.get('metadata') or {}
        # shortcut: the API may return metadata values as JSON strings ('"fixed"', '2'); not yet seen on Cloud for these keys.
        source = str(metadata.get('context_source', 'retrieval')).strip('"')
        required = required_spans(source, int(str(metadata.get('context_count', 1))))
        generation = next((row for row in rows if row.get('name') == 'answer_generation'), {})
        io_available = 'answer_generation' not in required or bool(generation.get('input') and generation.get('output'))
        missing = sorted(required - names)
        if rows and not missing and io_available and request_id in json.dumps(rows, default=str):
            return {'verified': True, 'api': 'v2/observations', 'trace_id': trace_id, 'observation_count': len(rows),
                    'names': sorted(n for n in names if n), 'generation_io_available': io_available}
        if attempt + 1 < attempts:
            time.sleep(wait_seconds)
    return {'verified': False, 'trace_id': trace_id, 'missing': missing,
            'reason': 'Required observations not readable yet; verify in Langfuse'}
