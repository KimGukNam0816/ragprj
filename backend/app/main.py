import json
import threading
from hmac import compare_digest
import time
import logging
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .connections import check_connections
from .memory.service import add_memory
from .pipeline import run_query
from .schemas import MemoryRequest, QueryRequest, QueryResponse, ServiceError

@asynccontextmanager
async def lifespan(app):
    app.state.reranker_warmup = {'status': 'skipped', 'message': 'RERANKER_WARMUP_ON_STARTUP=false'}
    if s.reranker_warmup_on_startup:
        def warmup():
            from .retrieval.reranker import get_reranker
            started = time.perf_counter()
            model = get_reranker(s)
            model.score('보관료 면제 조건은 무엇인가요?', ['보관료 면제에는 별도 승인이 필요합니다.'])
            return {'status': 'ready', 'model': s.reranker_model, 'device': model.device,
                    'duration_ms': round((time.perf_counter()-started)*1000)}
        try:
            # 미리 적재해 첫 사용자 요청에 모델 다운로드/적재 시간을 미루지 않는다.
            app.state.reranker_warmup = await run_in_threadpool(warmup)
            logging.getLogger(__name__).info('Reranker ready: %s', app.state.reranker_warmup)
        except Exception as exc:
            app.state.reranker_warmup = {'status': 'error', 'model': s.reranker_model,
                                         'error_type': type(exc).__name__,
                                         'message': 'hybrid_rerank 모드를 쓸 수 없습니다. naive/hybrid는 정상 동작합니다.'}
            logging.getLogger(__name__).warning('Reranker warmup failed (%s): %s', type(exc).__name__, exc)
            # 웹 배포에서는 리랭커 없이도 서비스가 떠야 한다. 엄격 모드는 설정으로 켠다.
            if s.reranker_warmup_fail_fast:
                raise
    yield


app=FastAPI(title='해운 문서 도우미',version='0.2.0',lifespan=lifespan)
s=get_settings()
gpu_lock=threading.Lock()

# 공개 배포 시 최소한의 접근 차단. PUBLIC_ACCESS_TOKEN이 비어 있으면 아무것도 하지 않는다(로컬 기본값).
PROTECTED_PATHS = ('/query', '/memory')


@app.middleware('http')
async def access_token_guard(request, call_next):
    expected = s.public_access_token.get_secret_value()
    if expected and request.method != 'OPTIONS' and request.url.path in PROTECTED_PATHS:
        header = request.headers.get('authorization', '')
        supplied = request.headers.get('x-access-token') or (header[7:] if header.lower().startswith('bearer ') else '')
        if not compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={'detail': {'code': 'unauthorized', 'message': '접근 토큰이 필요합니다.'}})
    return await call_next(request)


app.add_middleware(CORSMiddleware,allow_origins=s.cors_origins(),allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Entrypoint','X-Access-Token','Authorization'])

# 평가 화면용 읽기 전용 API (구현: evaluation/retrieval/api.py, 권오륜)
from evaluation.retrieval.api import router as evaluation_router
app.include_router(evaluation_router)


@app.exception_handler(ServiceError)
async def service_error(request,exc):
    return JSONResponse(status_code=exc.http_status,content={'detail':{'code':exc.code,'message':exc.message}})


@app.get('/health')
def health():
    return {'status':'ok','scope':'smoke','local_only':s.force_local_only,'metric_eligible':False,'model':s.local_llm_model,
            'reranker':getattr(app.state,'reranker_warmup',{'status':'unknown'}).get('status','unknown')}


@app.get('/status')
def status():
    indexes={}
    for profile in ['bge','qwen']:
        path=s.runtime_dir/'indexes'/f'active_{profile}.json'
        indexes[profile]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    return {'scope':'smoke','model':s.local_llm_model,'local_only':s.force_local_only,'indexes':{k:{'collection':v['collection'],'scope':v['scope']} if v else None for k,v in indexes.items()},'retrieval':{'candidate_k':s.dense_candidate_k,'top_n':s.final_top_k},'reranker':getattr(app.state,'reranker_warmup',{'status':'unknown'}),'auth_enabled':bool(s.public_access_token.get_secret_value())}


@app.get('/connections')
async def connections():
    return await check_connections(s)


@app.post('/query',response_model=QueryResponse)
def query(request:QueryRequest,x_entrypoint:str=Header(default='api')):
    started=time.perf_counter()
    entry='web' if x_entrypoint=='web' else 'api'
    try:
        with gpu_lock:
            response,_=run_query(s,request,entrypoint=entry,deadline=started+s.request_timeout_seconds)
        response.latency_ms=round((time.perf_counter()-started)*1000)
        return response
    except ServiceError:
        raise
    except Exception as exc:
        raise HTTPException(502,detail={'code':'pipeline_error','message':'답변 처리 중 오류가 발생했습니다. 저장된 실행 기록을 확인하세요.','type':type(exc).__name__}) from None


@app.post('/memory')
def save_memory(request:MemoryRequest):
    try:
        with gpu_lock:
            return add_memory(request.text,request.user_id)
    except ServiceError:
        raise
    except Exception as exc:
        raise HTTPException(502,detail={'code':getattr(exc,'code','memory_error'),'message':getattr(exc,'message','Memory operation failed'),'type':type(exc).__name__}) from None
