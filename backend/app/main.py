import json
import threading
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
from .observability import access_log, setup_logging, timed_lock
from .pipeline import run_query
from .providers import GPU_LOCK
from .schemas import MemoryRequest, QueryRequest, QueryResponse, ServiceError

@asynccontextmanager
async def lifespan(app):
    setup_logging('api')
    if s.reranker_warmup_on_startup:
        from .retrieval.reranker import get_reranker
        def warmup():
            started = time.perf_counter()
            model = get_reranker(s)
            model.score('보관료 면제 조건은 무엇인가요?', ['보관료 면제에는 별도 승인이 필요합니다.'])
            return {'model': s.reranker_model, 'device': model.device,
                    'duration_ms': round((time.perf_counter()-started)*1000)}
        # 준비 실패 시 서버도 시작하지 않아 첫 사용자 요청에 실패를 미루지 않는다.
        app.state.reranker_warmup = await run_in_threadpool(warmup)
        logging.getLogger(__name__).info('Reranker ready: %s', app.state.reranker_warmup)
    yield


# 관측 담당(2026-11-12): FastAPI 자체 OTel 추적을 끈다. Langfuse가 첫 질의 때 전역 tracer provider를 설정하면
# 그 뒤 요청마다 FastAPI가 서버 span을 열고, shipping_rag_query가 Langfuse로 내보내지지 않는 그 span의 자식이 되어
# 루트로 보이지 않는다(Is Root Observation 필터, list_traces에서 빠짐). 요청 기록은 access_log가 남긴다.
app=FastAPI(title='해운 문서 도우미',version='0.2.0',lifespan=lifespan,telemetry={'tracing':False})
app.middleware('http')(access_log)
s=get_settings()
app.add_middleware(CORSMiddleware,allow_origins=[s.frontend_origin,'http://localhost:3000'],allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Entrypoint'])
gpu_lock=GPU_LOCK   # 관측 담당: live judge의 임베딩도 같은 락을 쓴다(providers.GPU_LOCK)

# 평가 화면용 읽기 전용 API (구현: evaluation/retrieval/api.py, 권오륜)
from evaluation.retrieval.api import router as evaluation_router
app.include_router(evaluation_router)


@app.exception_handler(ServiceError)
async def service_error(request,exc):
    return JSONResponse(status_code=exc.http_status,content={'detail':{'code':exc.code,'message':exc.message}})


@app.get('/health')
def health():
    # 관측 담당: live_judge가 true면 local_only여도 답변 판정 호출은 OpenAI로 나간다(LIVE_JUDGE_ENABLED와 OpenAI 주소·키).
    return {'status':'ok','scope':'smoke','local_only':s.force_local_only,'live_judge':s.live_judge_enabled and s.llm_ready(),'metric_eligible':False,'model':s.local_llm_model}


@app.get('/status')
def status():
    indexes={}
    for profile in ['bge','qwen']:
        path=s.runtime_dir/'indexes'/f'active_{profile}.json'
        indexes[profile]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    return {'scope':'smoke','model':s.local_llm_model,'local_only':s.force_local_only,'indexes':{k:{'collection':v['collection'],'scope':v['scope']} if v else None for k,v in indexes.items()},'retrieval':{'candidate_k':s.dense_candidate_k,'top_n':s.final_top_k},'auth_enabled':False}


@app.get('/connections')
async def connections():
    return await check_connections(s)


@app.post('/query',response_model=QueryResponse)
def query(request:QueryRequest,x_entrypoint:str=Header(default='api')):
    started=time.perf_counter()
    entry='web' if x_entrypoint=='web' else 'api'
    try:
        with timed_lock(gpu_lock):
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
        with timed_lock(gpu_lock):
            return add_memory(request.text,request.user_id)
    except ServiceError:
        raise
    except Exception as exc:
        raise HTTPException(502,detail={'code':getattr(exc,'code','memory_error'),'message':getattr(exc,'message','Memory operation failed'),'type':type(exc).__name__}) from None
