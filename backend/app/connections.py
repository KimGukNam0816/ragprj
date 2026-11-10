import asyncio
import importlib.util

import httpx

from .config import Settings
from .redis_client import check_redis


async def check_connections(s: Settings) -> dict:
    async def probe(name, base, path, headers=None, auth=None, missing=None, local_llm=False):
        if missing:
            return name, {'status': 'missing', 'message': '설정 필요: ' + ', '.join(missing)}
        try:
            async with httpx.AsyncClient(timeout=s.connection_timeout_seconds) as client:
                r = await client.get(base.rstrip('/') + path, headers=headers, auth=auth)
                if not r.is_success:
                    return name, {'status': 'error', 'message': f'HTTP {r.status_code} 응답. 주소·권한을 확인하세요.'}
                result = {'status': 'ok', 'message': '연결 확인'}
                if name == 'ollama':
                    models = [m.get('name', '') for m in r.json().get('models', [])]
                    result['models'] = models
                    required = [s.embedding_model, s.comparison_embedding_model]
                    absent = [m for m in required if m not in models and m + ':latest' not in models]
                    if absent:
                        result.update(status='warning', message='모델 설치 필요: ' + ', '.join(absent))
                if name == 'qdrant':
                    result['collections'] = [c['name'] for c in r.json().get('result', {}).get('collections', [])]
                if name == 'llm':
                    if local_llm:
                        ids = [m.get('name') for m in r.json().get('models', [])]
                        result['model_listed'] = s.local_llm_model in ids or s.local_llm_model+':latest' in ids
                        result['message'] = 'Ollama 로컬 생성 모델 확인' if result['model_listed'] else '로컬 생성 모델 설치 필요'
                    else:
                        ids = [m.get('id') for m in r.json().get('data', [])]
                        result['model_listed'] = s.openai_model in ids
                    if not result['model_listed']:
                        result.update(status='warning', message='엔드포인트 연결됨. 지정 모델 사용 가능 여부는 프롬프트 호출로 확인하세요.')
                return name, result
        except (httpx.HTTPError, ValueError):
            return name, {'status': 'error', 'message': '연결 실패. 서비스 실행 상태·주소·네트워크를 확인하세요.'}

    lf_missing = [n for n, v in [('LANGFUSE_BASE_URL', s.langfuse_base_url), ('LANGFUSE_PUBLIC_KEY', s.langfuse_public_key), ('LANGFUSE_SECRET_KEY', s.langfuse_secret_key.get_secret_value())] if not v]
    llm_missing = [n for n, v in [('OPENAI_BASE_URL', s.openai_base_url), ('OPENAI_API_KEY', s.openai_api_key.get_secret_value())] if not v]
    checks = await asyncio.gather(
        probe('llm',s.ollama_base_url,'/api/tags',local_llm=True) if s.generation_llm_provider=='ollama' else probe('llm', s.openai_base_url, '/models', headers={'Authorization': 'Bearer ' + s.openai_api_key.get_secret_value()}, missing=llm_missing),
        probe('ollama', s.ollama_base_url, '/api/tags'),
        probe('qdrant', s.qdrant_url, '/collections', headers={'api-key': s.qdrant_api_key.get_secret_value()} if s.qdrant_api_key.get_secret_value() else {}),
        probe('langfuse', s.langfuse_base_url, '/api/public/projects', auth=(s.langfuse_public_key, s.langfuse_secret_key.get_secret_value()), missing=lf_missing) if s.langfuse_enabled else asyncio.sleep(0, result=('langfuse', {'status': 'disabled', 'message': '비활성'})),
    )
    result = dict(checks)
    result['redis'] = await asyncio.to_thread(check_redis, s)  # redis 클라이언트는 동기 방식이라 별도 스레드에서 실행
    result['mem0'] = {'status': 'configured' if importlib.util.find_spec('mem0') else 'missing', 'message': 'OSS 패키지 설치됨. 기억 저장·검색은 별도 확인 필요.' if importlib.util.find_spec('mem0') else 'uv sync --extra memory 필요'}
    result['reranker'] = {'status': 'configured' if importlib.util.find_spec('transformers') else 'missing', 'message': '대상 GPU에서 모델 적재 확인 필요' if importlib.util.find_spec('transformers') else '대상 GPU 머신에서 uv sync --extra reranker 필요'}
    return result
