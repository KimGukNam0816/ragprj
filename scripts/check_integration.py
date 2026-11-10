"""Step-labelled setup diagnostics. Never starts a benchmark or creates a document index."""
import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['tools', 'env', 'data', 'connections', 'corpus', 'indexes',
          'reranker', 'query', 'trace', 'api', 'web']
HINTS = {
    'tools': 'Install Git, uv, Node.js 24 LTS and Ollama; reopen PowerShell. Run nvidia-smi.',
    'env': 'Copy .env.example to .env; set paths and Cloud keys. Run uv sync --locked --extra memory --extra dev.',
    'data': 'Copy all 120 Advanced PDFs and original evaluation JSON files into data/advanced/part folders.',
    'connections': 'Check Ollama models, Cloud URLs/keys, VPN/firewall and actual 1024-dim embeddings.',
    'corpus': 'Run 01_chunk_documents. Pass the printed CORPUS_DIR with --corpus. Inspect manifest failures.',
    'indexes': 'Run 02_build_indexes for BOTH profiles in this RUNTIME_DIR. Do not copy old active pointers.',
    'reranker': 'Check CUDA PyTorch, free VRAM and HF download access. On 8GB use batch_size=1, max_length=1024.',
    'query': 'Check active index/digest/context settings, free VRAM, local model and saved run error records.',
    'trace': 'Check Langfuse ingestion/read permissions and region; 429 may require waiting before retry.',
    'memory': 'Check mem0 installation, local model and dedicated Qdrant memory collection permissions.',
    'api': 'Start Uvicorn in terminal A, from repository root, using the SAME .venv and .env.',
    'web': 'Start Next.js in terminal B after npm ci/build. Check BACKEND_URL and ports 3000/8000.',
}


class CheckFailed(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def settings():
    from backend.app.config import get_settings
    return get_settings()


def check(stage, args):
    if stage == 'tools':
        missing = [x for x in ['git', 'uv', 'node', 'npm', 'ollama', 'nvidia-smi'] if not shutil.which(x)]
        require(not missing, 'Missing commands: ' + ', '.join(missing))
        require(sys.version_info >= (3, 11), 'Python >= 3.11 required')
        return {'python': sys.version.split()[0], 'commands': 'available'}
    if stage == 'env':
        require((ROOT / '.env').exists(), '.env missing')
        absent = [x for x in ['httpx', 'pydantic_settings', 'torch', 'transformers', 'mem0', 'fastapi', 'langfuse', 'ragas']
                  if importlib.util.find_spec(x) is None]
        require(not absent, 'Missing Python packages: ' + ', '.join(absent))
        s = settings(); s.check_inference_policy()
        require(s.force_local_only and not s.allow_external_llm, 'Local-only flags must remain enabled')
        require(all(p == 'ollama' for p in [s.generation_llm_provider, s.evaluation_llm_provider, s.mem0_llm_provider]), 'All providers must be ollama')
        require(s.langfuse_enabled, 'LANGFUSE_ENABLED must be true for trace verification')
        import torch
        require(torch.cuda.is_available(), 'CUDA unavailable in this Python environment')
        return {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'runtime': str(s.runtime_dir.resolve()), 'local_only': True}
    s = settings()
    s.check_inference_policy()
    if stage == 'data':
        from backend.app.ingestion import discover_documents
        files = discover_documents(s)
        for name, count in [('development.json', 30), ('additional.json', 30), ('security_tasks.json', 2),
                            ('development_questions.json', 30), ('additional_questions.json', 30)]:
            require(len(read(s.evaluation_dir / name)) == count, 'Evaluation count mismatch: ' + name)
        return {'pdfs': len(files), 'general_questions': 60, 'security_questions': 2}
    if stage == 'connections':
        from backend.app.connections import check_connections
        from backend.app.providers import Providers
        results = asyncio.run(check_connections(s))
        # Redis는 REDIS_ENABLED=true일 때만 필수 연결로 검사
        bad = {k: v['status'] for k, v in results.items() if k in ['ollama', 'llm', 'qdrant', 'langfuse'] + (['redis'] if s.redis_enabled else []) and v['status'] != 'ok'}
        require(not bad, 'Connection failed: ' + json.dumps(bad))
        provider = Providers(s)
        dims = {p: len(provider.embed(['해운 연결 확인'], p)[0]) for p in ['bge', 'qwen']}
        generated = provider.chat([{'role': 'user', 'content': 'Return JSON only: {"ok":true}'}])
        require(json.loads(generated['content']).get('ok') is True, 'Local JSON generation check failed')
        require(provider.external_inference_attempts == 0, 'External inference attempted')
        return {'services': 'ok', 'embedding_dimensions': dims, 'local_generation': 'ok'}
    if stage == 'corpus':
        require(args.corpus is not None, 'Pass --corpus with the printed CORPUS_DIR')
        corpus = read(args.corpus / 'manifest.json')
        require(corpus['failure_count'] == 0, 'Extraction failures exist')
        require(corpus['chunk_count'] > 0 and (args.corpus / 'chunks.jsonl').stat().st_size > 0, 'No chunks')
        if args.require_full:
            require(corpus['scope'] == 'full' and corpus['document_count'] == 120, 'Full 120-document corpus required')
        return {k: corpus[k] for k in ['scope', 'document_count', 'page_count', 'chunk_count']}
    if stage == 'indexes':
        from backend.app.indexing import QdrantStore
        from backend.app.providers import Providers
        from backend.app.indexing import retrieve
        result = {}
        store = QdrantStore(s)
        try:
            for profile in ['bge', 'qwen']:
                pointer_path = s.runtime_dir / 'indexes' / f'active_{profile}.json'
                require(pointer_path.exists(), profile + ' active pointer missing in configured RUNTIME_DIR')
                pointer = read(pointer_path)
                state = read(Path(pointer['index_dir']) / 'manifest.json')
                count = store.count(pointer['collection'])
                require(state['complete'] and count == state['expected_count'], profile + ' index incomplete/count mismatch')
                if args.require_full:
                    require(state['scope'] == 'full', profile + ' is a subset index')
                # Also validates installed model digest/context and actual dense retrieval.
                hits, _, _ = retrieve(s, Providers(s), '일반 건화물 컨테이너 종류', profile, 'naive')
                require(bool(hits), profile + ' retrieval empty')
                result[profile] = {'collection': pointer['collection'], 'points': count, 'scope': state['scope']}
        finally:
            store.close()
        return result
    if stage == 'reranker':
        from backend.app.reranker import get_reranker
        model = get_reranker(s)
        values = model.score('일반 건화물 컨테이너 종류', ['일반 건화물 장비는 20GP와 40GP입니다.', '점심은 김치찌개입니다.'])
        require(values[0] > values[1], 'Relevant passage score is not above unrelated passage')
        return {'device': model.device, 'scores': values}
    if stage == 'query':
        from backend.app.pipeline import run_query
        from backend.app.schemas import QueryRequest
        response, folder = run_query(s, QueryRequest(question='가온루트해운 부산–LA 안내에서 일반 건화물로 예약할 수 있는 컨테이너 종류는 무엇인가요?', mode='hybrid_rerank', memory_enabled=False))
        require(response.status == 'answered' and bool(response.citations), 'Expected grounded answer with citations')
        saved = {'result_dir': str(folder.resolve()), 'request_id': response.request_id, 'trace_id': response.trace_id}
        path = s.runtime_dir / 'checks/setup_query.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
        return saved
    if stage == 'trace':
        from backend.app.pipeline import verify_trace
        saved = read(s.runtime_dir / 'checks/setup_query.json')
        manifest = read(Path(saved['result_dir']) / 'manifest.json')
        result = verify_trace(s, saved['trace_id'], saved['request_id'], manifest['started_at'])
        require(result['verified'], 'Trace/required observations/input/output not verified')
        return result
    if stage == 'memory':
        from backend.app.memory import add_memory, search_memory
        user = 'setup-' + uuid.uuid4().hex[:12]
        added = add_memory('한국어로 짧고 간결한 답변을 선호합니다.', user)
        own = search_memory('답변 언어와 길이 선호', user)
        other = search_memory('답변 언어와 길이 선호', user + '-other')
        require(bool(added.get('results')) and bool(own.get('results')) and not other.get('results'), 'Memory add/search/isolation failed')
        return {'user_id': user, 'stored': len(added['results']), 'own_hits': len(own['results']), 'other_hits': 0}
    import httpx
    if stage == 'api':
        base = args.api_url.rstrip('/')
        health = httpx.get(base + '/health', timeout=15); health.raise_for_status()
        require(health.json().get('status') == 'ok' and health.json().get('local_only') is True, 'Backend health/local policy mismatch')
        invalid = httpx.post(base + '/query', json={'question': ''}, timeout=15)
        require(invalid.status_code == 422, 'Blank input must return 422')
        return {'health': 200, 'invalid_input': 422}
    if stage == 'web':
        base = args.web_url.rstrip('/')
        page = httpx.get(base, timeout=20); page.raise_for_status()
        require('해운 문서 도우미' in page.text, 'Frontend HTML missing expected title')
        proxy = httpx.get(base + '/api/backend/status', timeout=20); proxy.raise_for_status()
        require(all(proxy.json().get('indexes', {}).get(p) for p in ['bge', 'qwen']), 'Frontend proxy cannot see both active indexes')
        return {'html': 200, 'backend_proxy': 200, 'manual_check': 'Send a browser question and click a citation; HTTP checks do not prove rendering.'}
    raise CheckFailed('Unknown stage')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES + ['memory', 'all'], default='tools')
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--require-full', action='store_true', help='Reject subset corpora/indexes')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000')
    parser.add_argument('--web-url', default='http://127.0.0.1:3000')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    rows = []; failed = False
    for stage in STAGES if args.stage == 'all' else [args.stage]:
        if failed:
            rows.append({'stage': stage, 'status': 'SKIP'})
            print(f'[SKIP] {stage}: prerequisite failed', flush=True)
            continue
        print(f'[RUN ] {stage}', flush=True)
        started = time.perf_counter()
        try:
            detail = check(stage, args)
            row = {'stage': stage, 'status': 'PASS', 'detail': detail}
            print(f'[PASS] {stage}: ' + json.dumps(detail, ensure_ascii=False), flush=True)
        except Exception as exc:
            # Never echo arbitrary HTTP exception strings, credentials, or stack frames.
            detail = str(exc) if isinstance(exc, CheckFailed) else type(exc).__name__
            row = {'stage': stage, 'status': 'FAIL', 'error': detail, 'hint': HINTS[stage]}
            print(f'[FAIL] {stage}: {detail}\n       NEXT: {HINTS[stage]}', flush=True)
            failed = True
        row['seconds'] = round(time.perf_counter() - started, 2); rows.append(row)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        folder = settings().runtime_dir / 'checks'
    except Exception:
        folder = ROOT / 'runtime/setup-checks'
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / f'setup-{stamp}.json'
    report.write_text(json.dumps({'timestamp': stamp, 'success': not failed, 'checks': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT=' + str(report.resolve()))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
