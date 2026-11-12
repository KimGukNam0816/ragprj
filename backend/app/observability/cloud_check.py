"""Check recording on Langfuse Cloud: python -m backend.app.observability.cloud_check --apply [--environment dev].

Runs one real run_query (kind `diagnostic`), reads the trace back, and sends one test score twice under the same id.
It writes to Langfuse Cloud, so --apply is required and each run needs the observability owner's approval. Prints check results only:
no keys, no trace input or output. A failed query (for example a missing index) still checks the error path.
Not exported (imports the pipeline)."""
import argparse
import json
import os
import time
from datetime import datetime, timezone

CHECK_SCORE = 'observability_check'   # its own name, so the probe never looks like a real result
CHECK_EVALUATOR = 'cloud-check'


def _poll(read, ready, *, timeout: float, interval: float):
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if ready(value) or time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def run_check(question: str, *, run_id: str, timeout: float = 120, interval: float = 10) -> dict:
    """Run the query, then return {'checks': {name: bool}, 'facts': {...}} from what Langfuse returns."""
    from ..config import get_settings
    from ..pipeline import run_query
    from ..schemas import QueryRequest
    from . import flush_scores, get_scores, get_trace, list_traces, record_score, trace_url, verify_trace
    from .query import root_of

    s = get_settings()
    response, error = None, None
    try:
        response, output = run_query(s, QueryRequest(question=question, mode='hybrid'), run_id=run_id,
                                     run_kind='diagnostic')
        saved = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    except Exception as exc:
        error = getattr(exc, 'code', 'pipeline_error')
        saved_files = list((s.runtime_dir / 'runs' / run_id).glob('*/result.json'))
        if not saved_files:   # failed before the run folder existed (e.g. Ollama down): no trace to check
            return {'checks': {'trace_created': False}, 'facts': {'query': f'failed:{error}', 'error_type': type(exc).__name__}}
        saved = json.loads(max(saved_files, key=lambda p: p.stat().st_mtime).read_text(encoding='utf-8'))
    trace_id = saved.get('trace_id')
    facts = {'query': 'ok' if response else f'failed:{error}', 'trace_id': trace_id, 'trace_url': trace_url(trace_id),
             'environment': os.environ.get('LANGFUSE_TRACING_ENVIRONMENT', 'default'),
             'flush_error': saved.get('flush_error')}
    if not trace_id:
        return {'checks': {'trace_created': False}, 'facts': facts}

    rows = _poll(lambda: get_trace(trace_id), lambda r: bool(root_of(r)),
                 timeout=timeout, interval=interval)
    root = root_of(rows)
    metadata = root.get('metadata') or {}
    listed = list_traces(run_id=run_id, tags=['diagnostic'])
    facts.update(span_names=sorted({x.get('name') for x in rows if x.get('name')}),
                 metadata_value_types={k: type(metadata.get(k)).__name__ for k in ('error_code', 'context_count', 'status')
                                       if k in metadata},
                 session_id=root.get('sessionId'), tags_on_root=root.get('tags'))
    checks = {'trace_readable': bool(root), 'session_is_run_id': root.get('sessionId') == run_id,
              'tags_filter_finds_root': any(x.get('traceId') == trace_id for x in listed)}
    if response:
        checks['required_spans'] = verify_trace(s, trace_id, response.request_id, attempts=1)['verified']
    else:
        checks['error_code_on_root'] = str(metadata.get('error_code', '')).strip('"') == error
        checks['error_level_on_root'] = root.get('level') == 'ERROR'

    # Same trace, name and evaluator -> same score_id. Send 0 then 1: one score with value 1 means overwrite.
    sent = [record_score(trace_id, CHECK_SCORE, v, comment='cloud_check', evaluator_sha256=CHECK_EVALUATOR) for v in (0, 1)]
    flush_scores()
    scores = _poll(lambda: get_scores(trace_id=trace_id, name=CHECK_SCORE),
                   lambda r: len(r) >= 1 and all(x.get('value') == 1 for x in r), timeout=timeout, interval=interval)
    facts['check_scores'] = [{'value': x.get('value'), 'id': x.get('id')} for x in scores]
    checks['score_sent'] = all(sent)
    checks['same_id_overwrites'] = len(scores) == 1 and scores[0].get('value') == 1
    return {'checks': checks, 'facts': facts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--apply', action='store_true', help='required: this writes a trace and scores to Langfuse')
    parser.add_argument('--environment', help='LANGFUSE_TRACING_ENVIRONMENT for this run, e.g. dev or production')
    parser.add_argument('--question', default='보관료 무료기간은 며칠인가요?')
    parser.add_argument('--timeout', type=float, default=120, help='seconds to wait for ingestion')
    args = parser.parse_args(argv)
    if args.environment:
        os.environ['LANGFUSE_TRACING_ENVIRONMENT'] = args.environment   # SDK reads os.environ when the client starts
    from . import tracing
    if tracing.langfuse_client() is None:
        print('Langfuse is not configured; nothing to check')
        return 1
    if not args.apply:
        print('preview: would run one diagnostic query and send one test score; add --apply to write to Langfuse')
        return 0
    run_id = 'obs-cloud-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = run_check(args.question, run_id=run_id, timeout=args.timeout)
    print(json.dumps({'run_id': run_id, **result}, ensure_ascii=False, indent=2, default=str))
    return 0 if all(result['checks'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
