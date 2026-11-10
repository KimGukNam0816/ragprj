import json
import time
from datetime import datetime, timedelta, timezone


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

