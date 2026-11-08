# Advanced 해운 RAG

새 PC 설치·청킹·색인·API·채팅 실행: **[SETUP.md](SETUP.md)**. 단계별 진단은 `python -m scripts.check_integration --help`를 참조하세요.

구현 경로: **연결 점검 → 청킹·색인 → 실제 리랭커 → 6개 구성 비교 → RAGAS/Judge·운영·보안 집계 → FastAPI/Uvicorn → Next.js/React 채팅**.
현재 실행은 RTX4090 연동 smoke입니다. 전체 문서 검색 성능이나 RTX5060 공식 지표가 아닙니다. Redis·인증·권한승인은 후속 단계입니다.

## 실행 환경

- 현재 RTX 4090: 개발 검증 전용. 공식 품질/운영 지표로 사용하지 않습니다.
- 최종 RTX 5060 8GB: 최종 비교 환경. 이 장비에서 GPU·모델·시간·메모리 재검증이 필요합니다.
- 임베딩: Ollama `bge-m3`, `qwen3-embedding:0.6b` (각 1024차원).
- 로컬 생성: `qwen3:4b`. 생성·평가·Mem0 제공자 설정을 검사하며 local_smoke에서 외부 추론으로 fallback하지 않습니다.
- Qdrant Cloud·Langfuse Cloud 연결은 유지합니다. 외부 추론 차단은 인터넷 연결 차단과 다릅니다.

```powershell
uv sync --extra dev --extra memory
```

`.env.example`을 참고해 `.env`에 실제 키를 입력합니다. 키는 Git·터미널 출력·결과 보고서에 기록하지 않습니다.
기존 `.env`가 있으면 덮어쓰지 마세요. `LOCAL_LLM_MODEL=qwen3:4b`, `EXECUTION_STAGE=local_smoke`,
`GENERATION_LLM_PROVIDER=ollama`, `ALLOW_EXTERNAL_LLM=false`, `PUBLISH_METRICS=false`로 시작합니다.

## 0. 연결 점검

```powershell
uv run python -m scripts.00_check_connections --embed --generate
```

두 임베딩 실제 출력·로컬 생성 JSON·Qdrant·Langfuse 인증을 확인합니다. `/models` 조회는 메타데이터 확인이며 외부 추론 호출이 아닙니다.
보고서: `$RUNTIME_DIR/checks/connections.json`. 외부 추론 시도 횟수를 확인합니다. 로컬 모드에서는 외부 LLM 모델 목록도 조회하지 않습니다.

## 1. 청킹

```powershell
uv run python -m scripts.01_chunk_documents
# 대표 문서만 처리하려면: --books AS007
```

`data/advanced/*/documents/AS001.pdf`~`AS120.pdf`만 수집합니다. 다른 난이도와 평가 정답은 색인하지 않습니다.
두 임베딩의 실제 tokenizer.json을 내려받아 캐시하고 동일 청크 경계를 만듭니다. tokenizer 파일 해시·snapshot을 설정 해시에 포함합니다.
목표 600 tokens, overlap 100, 제목 포함 hard limit 1000. 문서·페이지·원본 해시를 보존합니다.
PDF는 pypdf 페이지 텍스트 추출을 사용합니다. 복잡한 표의 행열·단위·각주 보존은 대표 원문과 별도로 대조해야 하며, 추출 오류 0이 의미적 품질 보장을 뜻하지 않습니다.

출력 경로는 `CORPUS_DIR=...`로 표시됩니다. 각 문서 체크포인트로 재개하며, `manifest.json`에 전체 페이지·청크·실패 수를 기록합니다.

## 2. 색인 및 복구

```powershell
# <CORPUS_DIR>를 앞 단계에서 출력된 실제 경로로 치환
uv run python -m scripts.02_build_indexes --corpus <CORPUS_DIR> --embedding bge --books AS007
uv run python -m scripts.02_build_indexes --corpus <CORPUS_DIR> --embedding qwen --books AS007
# 전체 색인: --books 옵션 없이 실행. 약 8만여 청크이므로 시간·Cloud 용량 확인 필요.
```

첫 수직 경로는 명시적 문서 subset으로 검증합니다. subset은 원본 전체 평가로 소개하지 않습니다.
원본·청킹·모델 digest·선택 문서가 컬렉션 이름의 해시에 포함되며 기존 컬렉션을 삭제하지 않습니다.
point ID는 결정적 UUID입니다. 네트워크 실패 후 같은 명령으로 재개하면 마지막 미확정 batch만 멱등 upsert합니다.
청크 변경 시 새 컬렉션을 생성하므로 예전 청크가 새 색인에 남지 않습니다.
완료 후 `runtime/indexes/active_bge.json` 또는 `active_qwen.json`이 사용 가능한 색인을 가리킵니다.
다른 PC로 옮기면 로컬 경로가 달라질 수 있으므로 같은 색인 명령으로 로컬 manifest·체크포인트를 재구성합니다.

## 3. 질문 하나와 Langfuse 확인

```powershell
# AS007 subset에서 먼저 검증할 수 있는 질문
uv run python -m scripts.03_query_once --question "가온루트해운 부산–LA 안내에서 일반 건화물로 예약할 수 있는 컨테이너 종류는 무엇인가요?" --embedding bge --mode naive --verify-trace
# 핵심 10문항의 질문 실행은 필요한 문서를 모두 색인한 후
uv run python -m scripts.03_query_once --question-id AS-DEV02 --embedding bge --mode hybrid --verify-trace
```

Naive: dense 후보20→최종4. Hybrid: dense20+BM25 20→동일 가중치 RRF(상수60)→최종4.
한국어 BM25는 원형·문자 bigram과 코드 보존 tokenizer를 사용하며 구성 버전을 기록합니다.
`--mode hybrid_rerank`는 RRF 상위20개를 Qwen3-Reranker-0.6B의 yes/no logits으로 실제 재정렬해 최종4개를 사용합니다.

`backend.app.pipeline.run_query()`가 향후 CLI·API·web 공통 진입점입니다. `entrypoint`만 달리하고 요청1건 trace1개로 연결합니다.
LangChain 프롬프트로 질문·문맥을 구성하고, 실제 문맥의 문서/페이지/chunk만 인용 가능한지 검증합니다.
공식 평가와 다른 개발용 결과이며 `metric_eligible=false`로 기록됩니다.

결과 위치: `runtime/runs/<run_id>/<request_id>/`.

- `manifest.json`: 설정 사본·해시, 장비, 시각, 질문/요청/색인 식별자.
- `result.json`: 후보·실제 최종 문맥·실제 모델 messages·원 응답·검증 답변·오류·usage·외부 추론 시도 횟수.
- `trace_verification.json`: flush 이후 Cloud에서 trace를 다시 읽어 요청 ID 존재 확인.

`--verify-trace` 실패는 기록 완료로 간주하지 않습니다. Langfuse의 generation 입력/출력·사용량과 검색·문맥을 직접 확인하세요.
Cloud 전파 지연 시 모델을 다시 호출할 필요 없이 `uv run python -m scripts.03_query_once --verify-existing <RESULT_DIR>`로 재조회합니다.
조회는 기존 Traces API 대신 시간 범위를 제한한 Observations API v2를 사용합니다.
로컬 생성 실패 시 외부 API로 전환하지 않습니다. 구조화 출력/출처 실패는 실패로 저장하며 오답을 성공 결과로 바꾸지 않습니다.

## 테스트

```powershell
uv run pytest
```

외부 추론 차단, 평가·Mem0 제공자 검증, 인용 위조 차단, RRF 중복·순위, 코드 토큰 보존을 검사합니다.
실제 Cloud·모델 smoke test는 위 0~3 명령으로 별도 수행합니다.

## 4. 4090 전체 연동 smoke

`.env`에서 `FORCE_LOCAL_ONLY=true`, `SMOKE_RUN=true`, `ALLOW_EXTERNAL_LLM=false`, `PUBLISH_METRICS=false`를 유지합니다.
생성·평가·Mem0 모델은 모두 `qwen3:4b` 로컬이며, 외부 추론으로 fallback하지 않습니다.
Qdrant/Langfuse Cloud는 저장·추적용으로만 연결합니다.

smoke 데이터는 `runtime/smoke/integration`와 `smoke_` 접두사의 새 Cloud 컬렉션으로 분리합니다.
전체120문서의 앞2페이지를 고정 선택한 색인은 평가 정답으로 검색을 유도한 것이 아니며, 전체83,526청크 색인과 구분됩니다.

```powershell
uv run python -m scripts.02_build_indexes --corpus runtime/corpora/advanced-276cbbaddf-all --embedding bge --max-page 2
uv run python -m scripts.02_build_indexes --corpus runtime/corpora/advanced-276cbbaddf-all --embedding qwen --max-page 2
uv run python -m scripts.check_reranker
uv run python -m scripts.04_run_comparison --set core10 --run-name smoke-core10-local-v2
uv run python -m scripts.05_evaluate_quality --comparison runtime/smoke/integration/comparisons/smoke-core10-local-v2
uv run python -m scripts.07_build_report --comparison runtime/smoke/integration/comparisons/smoke-core10-local-v2
uv run python -m scripts.04_run_comparison --set final60 --run-name smoke-final60-local
uv run python -m scripts.05_evaluate_quality --comparison runtime/smoke/integration/comparisons/smoke-final60-local
uv run python -m scripts.07_build_report --comparison runtime/smoke/integration/comparisons/smoke-final60-local
uv run python -m scripts.06_evaluate_security --run-name smoke-security-local
```

CLI에서 `--set`을 생략하면 `EVAL_SELECTION`·`EVAL_QUESTION_LIMIT`·`EVAL_QUESTION_IDS_FILE`을 사용합니다.
예: `EVAL_SELECTION=core10`, `EVAL_QUESTION_LIMIT=10`, `EVAL_QUESTION_IDS_FILE=evaluation/selection/core10.json`.
명시적 `--set final60`은 기본적으로60문항 전체를 실행하며, `--limit`는 smoke 부분 검증에만 사용합니다.
성공한 비교 항목은 같은 설정으로 재개 시 건너뛰며, 실패한 항목만 재시도합니다. 설정이 달라지면 새 실행 이름을 사용하세요.

자동 평가는 저장된 답변에 실제 RAGAS Faithfulness와 별도 로컬 Judge를 적용합니다. 두 평가는 서비스 생성 시간/토큰에 합산하지 않습니다.
해당 없음·평가 불가·서비스 실패는 구분하며, CSV에 유효 분모를 함께 기록합니다. 로컬 Judge의 사실·노출 판정은 사람이 원문과 대조해야 합니다.
보안은 두 질문 x 기본/방어 총4응답, 동일 최종 문맥 해시 검증. 공격 미노출·생성 실패를 방어 성공으로 세지 않습니다.

## 5. 실제 채팅 서버

각 터미널에서 실행합니다. 개발 서비스는 localhost에만 바인딩하며 아직 로그인/권한 서비스가 아닙니다.

```powershell
# 백엔드 / 프로젝트 루트
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
# 프런트엔드 / frontend 폴더
npm install
npm run build
npm start
```

채팅: http://127.0.0.1:3000 / API 문서: http://127.0.0.1:8000/docs
Next.js 서버가 `/api/backend/*`를 FastAPI로 프록시하므로 브라우저에 키를 전달하지 않습니다. `BACKEND_URL`은 필요할 때 frontend 서버 환경변수로 지정합니다.
검색 방식·임베딩 선택, 인용 원문 패널, 실제 처리 시간·토큰·trace ID, 로컬 대화 기록을 제공합니다.
사용자 기억은 설정 패널에서 명시적으로 선호를 저장하고 켤 수 있습니다. 과제 비교에서는 항상 OFF입니다.

주요 API: GET `/health`, `/status`, `/connections`; POST `/query`, `/memory`.
web→FastAPI는 entrypoint=web인 trace1개로 처리합니다. CLI와 API도 같은 `run_query`를 사용합니다.

## 최종 RTX5060 실행

4090 smoke 결과를 그대로 공식 결과로 쓰지 않습니다. RTX5060에서 모델 GPU 호환·전체120 PDF의 두 완전 색인·동결 설정을 재검증한 후 실행합니다.
`PUBLISH_METRICS=true`는 smoke가 꺼져 있고 실제5060이 확인될 때만 허용합니다. subset 색인으로 공식 실행하는 것도 차단합니다.
이번 사용자 지시에 따라 기본값은 최종 답변까지 모두 로컬 추론입니다. 외부 API 키가 남아 있어도 로컬 모드에서 사용하지 않습니다.

5060 8GB에서는 `LOW_VRAM_MODE=true`로 Ollama의 임베딩/생성 모델을 호출 후 내려 여러 모델의 GPU 동시 상주를 줄일 수 있습니다.
`EMBEDDING_CONTEXT_TOKENS=2048`로 임베딩 문맥 할당을 제한하는 안도 검증하세요. 이 설정이 바뀌면 새 색인을 만들도록 버전이 구분됩니다.
이는 대상 장비를 위한 실행 옵션이며, 4090에서 5060의 메모리 사용량을 검증했다는 뜻은 아닙니다.

### 로컬 추론 고정

생성·최종 답변·RAGAS·Judge·Mem0 추론은 모든 단계에서 Ollama 로컬 모델만 사용합니다. 기존 API 설정이 남아 있어도 외부 추론은 실행하지 않습니다. `ALLOW_EXTERNAL_LLM=true`, `FORCE_LOCAL_ONLY=false`로 바꿔도 외부 제공자는 정책 검사에서 차단되며 외부 생성 요청 코드도 제거했습니다. Qdrant Cloud와 Langfuse Cloud는 저장·관측 용도로 유지합니다.
