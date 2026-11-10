# 시작 준비와 LLM 관측

## reranker

`RERANKER_WARMUP_ON_STARTUP=true`가 기본입니다. FastAPI lifespan에서 모델 로딩과 짧은 추론을 마친 뒤 요청을 받습니다. 준비 실패 시 서버 시작도 실패하며, 준비 결과는 서버 로그에 모델·device·duration_ms로 남습니다. 준비 시간은 사용자 요청 latency에 포함되지 않습니다. 직접 CLI를 실행하면 별도 프로세스이므로 이 준비를 공유하지 않습니다.

개발 시 모델 없이 API만 검사하려면 false로 설정할 수 있습니다. 이때 첫 rerank 요청에서 지연 로딩합니다. 8GB GPU에서는 우선 uvicorn worker 1개로 실행하세요. worker마다 모델·BM25 메모리가 따로 필요하며 reload/재시작 시 다시 준비합니다. 시작 워밍업은 이후 메모리 부족이나 실제 긴 입력의 성능까지 보장하지 않습니다.

## BM25

활성 인덱스의 `chunks.jsonl` 경로·manifest fingerprint·파일 크기·수정 시각으로 청크와 BM25Okapi 객체를 캐시합니다. 같은 버전 요청은 파일 읽기·전체 토큰화·통계 계산을 반복하지 않습니다. 질문별 점수 계산은 계속 수행합니다. 새 활성 버전/수정 파일은 다시 구성하며, 프로세스당 최근 두 버전을 보관합니다. Qdrant 벡터 인덱스에는 변경이 없습니다.

## OpenAI 답변 생성 선택

실제 `.env`는 자동 변경하지 않습니다. 사용자가 다음 값을 모두 설정해야 외부 답변 생성이 허용됩니다. 평가·Mem0는 로컬 정책을 유지하고 자동 fallback은 없습니다.

```dotenv
GENERATION_LLM_PROVIDER=openai
ALLOW_EXTERNAL_LLM=true
FORCE_LOCAL_ONLY=false
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=여기에_본인의_키
OPENAI_MODEL=gpt-6-luna
```

OpenAI Chat Completions 호환 `/chat/completions`를 사용합니다. 모델의 계정 접근 권한과 해당 endpoint/structured output 지원은 실제 환경에서 확인해야 합니다. 모델을 임의로 대체하지 않습니다. 입력 예산은 tiktoken 근사값이며 실제 사용량은 API 응답 usage입니다. API 모델로 바꿔도 질문 임베딩은 Ollama, reranker는 Transformers를 사용합니다. 변경 후 서버를 재시작하세요.

## Langfuse 확인

trace → `answer_generation` → Metadata에서 provider, model, latency_ms, finish_reason, provider_request_id, response_id, usage, attempts를 확인합니다. Usage에는 제공받은 입력·출력 토큰만 기록합니다. API usage의 cached_tokens/reasoning_tokens 등 세부 값은 Metadata의 usage에 원래 구조로 남습니다. 실패한 호출은 사용량을 알 수 없으므로 null로 두며 합계에 0으로 추정하지 않습니다.

`answer_generation` 아래 `llm_attempt`에는 시도별 latency_ms, http_status, error_type/error_code와 provider_request_id가 기록됩니다. 시도 span은 generation 타입이 아니므로 상위 generation과 토큰이 이중 집계되지 않습니다. 키·인증 헤더·원문 HTTP 오류 응답은 Metadata에 넣지 않습니다. 실제 입력/출력 기록은 기존 프로젝트 정책을 따릅니다.

OpenAI에서는 모델 로딩·입력 처리·출력 생성의 분리 시간을 만들지 않습니다. 현재 비스트리밍이므로 TTFT도 측정하지 않습니다. 비용은 임의 단가로 계산하지 않으며 Langfuse 모델 가격 설정/지원 여부를 확인하세요. Ollama를 유지하면 timing에 load_seconds, prompt_eval_seconds, eval_seconds, total_seconds 및 출력 tokens/sec를 기록합니다.

전체 요청 예산 120초, LLM 호출 timeout 90초, 최대 재시도 1회 기본값은 바꾸지 않았습니다. 실제 `.env` 설정이 우선합니다. 외부 HTTP timeout은 남은 요청 예산으로 제한하며, 로컬 BM25/GPU 연산 전체를 강제로 중단하는 하드 deadline은 아닙니다.

## 검증

`uv run pytest`는 외부 호출 없이 캐시 재사용/갱신, 시작 워밍업, API usage·request ID·재시도·deadline 및 비밀 비노출을 검사합니다. 실제 GPU 워밍업 시간·OpenAI 유료 호출·Langfuse Cloud 수신 확인과는 구분합니다. 같은 질문의 최초/반복 요청을 비교하되 답변 내용·토큰 수가 달라지면 이를 함께 기록하세요.

API 형식 참고: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
