# 담당별 작업 시작 안내

이 문서는 담당자별 리팩토링 후의 실제 소스 구조와 공통 계약 v1을 설명합니다. 호칭 없이 이름을 사용합니다. 담당별 기능 개선은 이 기준에서 시작하며, 담당 밖의 파일이 필요하면 통합 작업으로 분리합니다.

## 담당 범위

| 순서 | 담당 | 수정할 구현 | 전용 테스트 | 범위 |
|---|---|---|---|---|
| 저장 기반 | 이정로 | `backend/app/storage/` | `backend/tests/storage/` | Redis 연결·사용자 정보·합의된 캐시. 현재는 계약과 예약 영역만 있으며 미구현 |
| 문서 검색 | 권오륜 | `backend/app/retrieval/`, `evaluation/retrieval/` | `backend/tests/retrieval/` | Qdrant 검색·적재, BM25 캐시, RRF, reranker, 검색 지표·방식별 실행 |
| 사용자 선호 | 정김기보 | `backend/app/memory/` | `backend/tests/memory/` | Mem0 연결·명시 선호 저장·조회 |
| Context·검증 | 김국남 | `backend/app/context_engineering/`, `evaluation/security/` | `backend/tests/context_engineering/`, `backend/tests/security/` | 입력 예산·프롬프트·출처/출력 검증·고정 Context 보안 비교 |
| 실행 관측 | 노영우 | `backend/app/observability/` | `backend/tests/observability/` | Langfuse trace·usage 변환·점수 전송·trace 재조회 |

통합 담당자는 아직 별도로 지정하지 않았습니다. 공통 파일의 변경은 별도 통합 PR/작업으로 분리합니다. 이번 리팩토링에서 새로운 답변 품질 평가 담당자나 프런트엔드 담당자를 임의로 배정하지 않았습니다.

## 실제 소스 트리

```text
backend/app/
  contracts.py                  공통 계약 v1
  main.py                       공통: FastAPI·시작 워밍업·라우팅
  pipeline.py                   공통: 검색 → Context → 모델 → 검증 → 응답
  config.py, schemas.py         공통 설정·HTTP 요청/응답
  providers.py                  공통 OpenAI/Ollama 호출
  ingestion.py, artifacts.py    공통 문서 추출·산출물 저장
  connections.py                공통 연결 점검
  storage/                     이정로 (Redis 미구현)
    redis_client.py, user_store.py, cache.py
  retrieval/                   권오륜
    service.py                 활성 인덱스 확인·적재·검색 모드 연결
    dense.py                   QdrantStore
    bm25.py                    토큰화·버전별 BM25 캐시
    fusion.py                  RRF
    reranker.py                모델 준비·scoring·rank
  memory/                      정김기보
    mem0_client.py              Mem0 모델·저장소 연결
    service.py                 add_memory/search_memory
    preferences.py             Mem0 응답 → 프롬프트 선호 문자열 (외부 연결 없음)
    records.py                 Mem0 응답 행·metadata 읽기 도우미
  context_engineering/         김국남
    builder.py                 prepare_context
    budget.py                  tokenizer·토큰 추정·입력 예산
    prompts.py                 답변·직접 호출 프롬프트
    validation.py              validate_answer
  observability/               노영우
    tracing.py                 Langfuse client·observation
    verification.py            실제 trace 재조회
    usage.py                   usage·generation metadata 변환
    scores.py                  점수 전송 진입점 (자동 채점 연결은 별도)

evaluation/
  retrieval/benchmark.py       권오륜: 세 검색 방식 실행
  retrieval/metrics.py         권오륜: 문서 기준 Hit/Recall
  security/runner.py           김국남: 고정 Context 보안 실행
  quality.py                   공통 유지: RAGAS·Judge, 담당 추가 합의 필요
  report.py                    공통 유지: 최종 비교 집계
```

## 공통 입출력 v1

정의 위치는 `backend/app/contracts.py`입니다. HTTP 계약은 기존 `schemas.py`를 그대로 사용합니다. 새 계약은 Python 자료구조와 타입 힌트이며 모든 경계에서 자동 런타임 검사를 수행한다는 뜻은 아닙니다. 출력 검증은 `validate_answer`가 수행합니다.

### 1. 검색 결과

```python
retrieve(settings, provider, question, profile, mode) -> RetrievalResult
# profile: bge | qwen
# mode: naive | hybrid | hybrid_rerank

RetrievalResult(
    selected=[chunk, ...],      # 상위 final_top_k. 높은 순위 먼저
    candidates=[chunk, ...],    # 해당 모드의 순위가 적용된 후보
    index={...},               # 기존 manifest 원본 필드 보존
)

chunk = {
    "chunk_id": "stable-id",
    "book_id": "AS001",        # PDF 확장자를 뺀 이름
    "page": 1,                 # PDF 첫 페이지가 1
    "text": "원문",
    # score, rerank_score 및 기존 원문 메타데이터는 추가 가능
}
```

`index`에는 최소 `fingerprint`, `scope`가 있어야 하며 실제 manifest에는 collection/config 등의 필드가 더 있습니다. scope는 full/subset입니다. 기존 tuple 언패킹도 유지합니다. 검색 담당은 정답 JSON의 근거를 검색 입력에 넣지 않습니다.

### 2. 사용자 선호

`search_memory(question, user_id) -> MemoryResult`, `add_memory(text, user_id) -> MemoryResult`입니다. 기존 Mem0 dict/list 응답을 그대로 반환합니다.

공통 pipeline은 이 응답을 `memory.preferences.select_preferences(settings, result)`로 선호 문장만 뽑아 `memory_text`로 전달합니다. 원본 JSON(`id`·`hash`·`score`·`created_at`·`metadata`)은 프롬프트에 들어가지 않고 실행 기록 `result.json`의 `memory_results`·`memory_selection`에만 남습니다. 선별·정규화 규칙은 [memory 문서](memory.md)에 있습니다. 남는 선호가 없으면 기존과 같은 `'사용하지 않음'`입니다.

Context 담당은 Mem0를 직접 호출하지 않습니다. `QueryRequest.memory_enabled=false`면 기존처럼 조회하지 않고, 서버 설정 `MEMORY_ENABLED=false`면 요청 플래그로도 켤 수 없습니다. 선호는 표현 방식 참고용이며 업무 사실의 출처가 아닙니다.

### 3. Context 구성

```python
prepare_context(settings, question, selected_chunks,
                memory_text="사용하지 않음", defense_addition=None)
# -> PreparedContext
# contexts: 실제 모델에 남길 청크 원문
# messages: [{role: system/user, content: str}, ...]
# estimated_input_tokens: 입력 토큰 추정값
# context_hash: 최종 청크들의 digest
# dropped_chunk_ids: 예산 때문에 제외한 청크 ID
# warnings: 제외 사실 등 경고
```

입력 청크를 직접 수정하지 않습니다. 모델에는 book_id/page/chunk_id/text만 넣고 score나 평가 정답은 넣지 않습니다. 기존 메시지·토큰 계산·하위 청크 제거 정책을 유지했습니다. 중복 제거·압축·시점 판단의 새 정책은 이후 김국남의 별도 개선 작업입니다. tokenizer를 인자로 주입해 외부 다운로드 없는 테스트가 가능합니다.

현재 OpenAI 예산은 기존 context_max_tokens와 tiktoken 근사값이며 실제 사용량은 API usage입니다. 로컬 예산은 기존 출력 예약 토큰을 반영합니다. 모델 한도·계산 정책의 새로운 변경은 이 리팩토링에 섞지 않았습니다.

### 4. 생성과 검증

`Providers.chat(messages, schema) -> GenerationResult`는 content/model/input_tokens/output_tokens/attempts를 반환하며 provider request ID·usage 상세는 선택 필드입니다. 미확인 토큰은 null입니다. 자동 fallback 및 외부 호출 허용 정책은 바꾸지 않았습니다.

`validate_answer(content, final_contexts) -> ModelAnswer`는 JSON, 기존 응답 스키마와 실제 전달 출처를 검사합니다. status는 answered/needs_clarification/abstained를 유지합니다. answered는 실제 출처가 최소 1개 필요합니다. 허위 인용·잘못된 출력은 기존처럼 예외가 됩니다. 형식·출처 범위 검사가 내용의 사실성까지 보장하지는 않습니다.

### 5. 관측과 저장소

`observation(name, kind, input, metadata, model)`과 `langfuse_client()`의 기존 호출 계약은 유지합니다. `usage_details(generated)`는 알려진 토큰만 반환하고 `generation_metadata`는 본문을 제외한 실행 정보를 반환합니다. 모든 단계는 공통 request_id/trace_id로 연결합니다. `verify_trace`는 flush 성공과 Cloud 수신을 구분합니다.

Redis는 이번 작업에서 새로 구현하지 않았습니다. UserStore의 get_user/save_user, CacheStore의 get/set/delete 계약을 정의했으며 cache.set에는 TTL(초)을 받습니다. 사용자 정보 조회 실패·인증·캐시 키 정책과 실제 요청 경로 연결은 이정로 구현 후 통합 담당자가 연결합니다. Redis가 연동된 것처럼 통과값을 반환하지 않습니다.

## 작업 규칙

1. 각 담당은 전용 폴더·전용 테스트·본인의 docs 문서만 수정합니다. `__init__.py`도 해당 폴더 담당이 관리합니다.
2. 공통 계약/응답/설정/의존성/진입점 변경이 필요하면 별도 통합 작업으로 요청합니다. 다른 담당의 내부 구현을 import하지 말고 위 공개 service 함수를 사용합니다.
3. `main.py`, `pipeline.py`, `contracts.py`, `config.py`, `schemas.py`, `providers.py`, 공통 문서 추출, `.env.example`, `pyproject.toml`, `uv.lock`, CI, 기존 혼합 테스트는 공통 관리입니다.
4. 기존 `indexing.py`, `reranker.py`, `prompts.py`, `evaluation/benchmark.py`는 호환용입니다. 새 기능은 이 파일들에 넣지 않습니다. 기존 memory/observability/security import도 패키지의 공개 함수로 계속 동작합니다.
5. 본인 테스트 후 전체 `python -m pytest -q`를 실행합니다. 통합 테스트는 가짜 서비스로 API→Context→출력·산출물까지 확인하고 실제 Cloud 성공과 구분합니다.
6. 파일 담당 구분은 협업 규칙입니다. Git 편집 권한을 강제로 제한하는 기능은 아니므로 변경 파일 목록을 검토해야 합니다.

## 로컬 작업 시작

현재 리팩토링은 `refactor/team-modules` 로컬 브랜치입니다. push/PR/main 병합은 하지 않습니다. 로컬 확인 후 이 변경을 기준으로 합의한 브랜치에 반영하고, 각 담당은 같은 기준 커밋에서 개인 브랜치를 시작합니다. 커밋 전 상태를 두고 다른 브랜치로 작업을 분산하지 마세요.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_integration --help
```

변경 후 서버는 완전히 재시작합니다. 기존 endpoint·컬렉션명·인덱스 fingerprint·청크·모델·적재 데이터·.env·uv.lock은 변경하지 않았으며, 리팩토링 때문에 재임베딩할 필요는 없습니다. 소스 파일 경로가 달라져 실행 source hash는 바뀌므로 이전 비교 실행 폴더의 config hash를 억지로 수정하지 말고 새로운 run_name으로 비교합니다. 실제 GPU·OpenAI·Qdrant·Langfuse 연동은 별도 환경 검증이 필요합니다.
