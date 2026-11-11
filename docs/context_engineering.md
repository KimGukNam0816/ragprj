# 김국남 담당: Context·프롬프트·출력 검증

구현은 `backend/app/context_engineering/`, 테스트는 `backend/tests/context_engineering/`입니다. 공통 `PreparedContext`/HTTP 응답 스키마를 바꾸지 않고 다음을 보강했습니다. 공통 pipeline·검색·저장소·Providers·Langfuse 전송 구현·설정·의존성은 수정하지 않습니다.

## 단계별 동작

| 파일 | 처리 |
|---|---|
| builder.py | 출처 필드 검증 → 동일 출처/본문/버전 중복 제거 → 입력 예산 내 상위 청크 선택 → 실제 메시지 구성 |
| budget.py | 역할·메시지 경계·출력 스키마를 포함한 길이 추정과 여유분 256. 로컬 출력 예약량 반영 |
| prompts.py | context-v2: 조건·단위·시점·비용별 승인·부분 답변·보류·문서 내 지시 분리 |
| validation.py | 엄격한 필드/타입·중복 JSON 키·공백 응답·출처 tuple·중복 인용·본문 표준 인용 불일치 검사 |
| telemetry.py | 기존 기록 중인 trace에만 단계 observation 추가. 독립 단위 테스트는 Cloud 전송 없음 |

원문 중복은 같은 book_id/page/text/version/effective_date 기준입니다. 다른 페이지·버전이나 서로 보완하는 조건은 삭제하지 않습니다. 같은 chunk_id가 서로 다른 출처/본문/버전을 가리키면 `context_conflict`로 중단합니다. 충돌을 높은 검색 점수나 최신 날짜로 임의 해결하지 않습니다.

예산 초과 시 낮은 순위의 청크 전체를 제외하며 문장·표를 중간에서 자르지 않습니다. 제외 ID와 경고를 남깁니다. 이는 의미를 이해한 압축 알고리즘이 아닙니다. 필요한 보완 근거도 예산 때문에 빠질 수 있어 결과를 확인하고 입력 예산 또는 후보 구성을 조정해야 합니다. 검색 순위를 바꾸지 않습니다.

Local tokenizer는 현재 qwen3:4b에 매핑되어 있습니다. 다른 로컬 모델을 설정하면 해당 tokenizer 매핑을 추가하거나 테스트에서 주입해야 합니다. OpenAI는 tiktoken 모델 매핑 또는 o200k_base 근사값을 사용합니다. `context_max_tokens`는 기존 입력 한도이며 OpenAI 모델 전체 context 한도라고 주장하지 않습니다. 모델 전체 한도·출력 예약과 API 지원은 실제 모델 전환 시 확인해야 합니다. 모든 카운트는 서버 chat template와 다를 수 있는 추정값이고 실제 사용량은 API usage로 확인합니다.

`needs_clarification`/`abstained`의 reason은 프롬프트에서 요구하되 기존 공통 스키마가 선택 필드이므로 검증기에서 새로운 필수 조건으로 바꾸지는 않았습니다. 이유 존재 여부를 trace에 남깁니다. 본문에 `[문서ID p.페이지]` 표기가 있으면 citations와 일치해야 합니다. 형식·인용 범위 통과는 사실성·계산 정확성·주입 방어 성공을 뜻하지 않으며 별도 원문 검토/RAGAS/Judge 대상입니다. 개인정보 전체 탐지·권한 시스템은 구현하지 않았습니다.

## 별도 웹 앱 대신 로컬 보고서와 Langfuse

새 프런트엔드 페이지나 API를 추가하지 않았습니다. 진단 CLI가 만든 `report.html`을 브라우저에서 열면 각 문항의 후보·최종 Context·예산·실제 메시지·샘플 출력 검사·실제 모델 결과를 펼쳐볼 수 있습니다. HTML은 입력/출력을 escape해 문서의 스크립트를 실행하지 않습니다.

기존 API 요청에 Langfuse 기록이 활성화되어 있으면 다음 단계가 기존 trace 아래에 나타납니다.

```text
context_deduplication       유지 ID·중복 제거 ID
context_budget              입력 추정량·예산·출력 예약·예산 제외 ID
context_prompt              prompt_version·Context/메시지 hash·출처 수
context_output_validation   상태·인용 수·이유 존재 여부·검사 결과
```

실제 생성 입력/출력은 기존 `answer_generation`/`output_validation` 기록을 함께 확인합니다. 기록 중인 parent trace가 없거나 sampling에서 제외되면 이 자식 단계도 보내지 않습니다. 자체적인 평가 점수 이름이나 팀 보안 점수 S4를 새로 정의하지 않았습니다.

## 별도 진단 질문 24개

`diagnostics/questions.json`은 기존 development/additional/security 평가셋과 분리된 합성 근거입니다. CEFIX 문서 ID는 실제 PDF가 아닙니다. Qdrant 검색·임베딩·기존 평가 자료의 변경/색인을 하지 않습니다.

- 단일 근거, 동일 ID/본문 중복, ID 충돌, 페이지 차이, 보완 조건, 문서 충돌
- 예산에 따른 청크 제외, 질문 자체 초과, 빈 근거, 추가 확인, 사용자 선호 분리
- 표 헤더/단위, 기준일 전후 자료, 문서 내 공격 문구와 방어 지시
- 허위 출처, 인용 없는 답변, 본문 인용 불일치, 추가 필드, 잘못된 타입, 공백 답변, 중복 인용, 잘못된 청크

`sample_output`은 검증기 입력용 샘플이며 실제 모델 답변이 아닙니다. 정답 필드처럼 생성 입력에 보내지 않습니다. 기대 규칙 통과는 구조 처리 검증이며 실제 검색/답변 정확도·보안 성공률이 아닙니다. 기존 과제 보안 질문 2개의 고정 Context 비교는 여전히 별도로 수행해야 합니다.

### 1. 기본: 모델/Cloud 호출 없이 전체 구조 진단

```powershell
.\.venv\Scripts\python.exe -X utf8 -m backend.app.context_engineering.diagnostics --mode offline
```

offline은 외부 tokenizer 다운로드도 피하기 위해 테스트용 문자 카운터를 사용합니다. 보고서의 길이를 실제 모델 토큰이나 운영 성능으로 해석하지 마세요. 질문별 결과 JSONL, summary.json, report.html이 설정된 RUNTIME_DIR/context_diagnostics/실행이름 아래에 생성됩니다. 이전 실행을 덮어쓰지 않습니다. `.env` 값은 바꾸지 않습니다.

### 2. 명시적으로 실제 모델 답변 확인

```powershell
.\.venv\Scripts\python.exe -X utf8 -m backend.app.context_engineering.diagnostics --mode model --limit 3
```

현재 `.env`의 provider/model을 그대로 사용합니다. `--limit`은 1..24로 명시해야 합니다. 로컬 모델은 Ollama, OpenAI는 기존 외부 호출 허용 설정이 있어야 하며 자동 전환하지 않습니다. 모델/API 호출 또는 tokenizer 다운로드가 발생할 수 있고 설정된 재시도도 적용됩니다. 기본은 Langfuse 전송을 CLI 프로세스에서 비활성화합니다. 모델 결과는 샘플 검사와 별도 저장하고 내용 정확성은 원문/조건을 직접 검토합니다.

### 3. Langfuse에도 명시적으로 전송

```powershell
.\.venv\Scripts\python.exe -X utf8 -m backend.app.context_engineering.diagnostics --mode model --limit 3 --trace
```

유효한 Langfuse 설정과 enabled=true가 필요합니다. `--mode offline --trace`는 합성 샘플·문자 카운트의 구조 진단만 전송합니다. trace는 `evaluation`, `context_diagnostic` 태그와 실행명 session으로 묶고 서비스·비교 운영 지표에 합산하지 않습니다. flush 요청 후에도 Cloud 저장 건수/본문은 별도 확인해야 하며 summary의 cloud_verified는 자동으로 true가 되지 않습니다.

CLI 이외에서 runner를 직접 호출할 경우 관측 설정은 호출 프로세스의 기존 설정을 따릅니다. 독립 실행·전송 제어는 위 CLI 사용을 권장합니다. 기존 서비스의 마스킹·태그 규약을 변경하지 않습니다.

## 로컬 검증

### Mem0 연결 전 더미 선호 시험

현재 경계 형식은 `search_memory() -> dict 또는 list`, 공통 pipeline의 `json.dumps(..., ensure_ascii=False)[:1500] -> str`, `prepare_context(..., memory_text=str)`입니다. builder에는 Mem0 객체나 dict를 직접 넘기지 않습니다. PDF 청크 목록은 별도 인자로 유지합니다.

실제 Mem0를 붙이기 전에 다음 설정으로 고정 응답만 시험할 수 있습니다. 실제 `.env`는 자동 변경하지 않았습니다.

```dotenv
MEMORY_BACKEND=dummy
```

서버 재시작 후 Swagger `/query`에서 다음처럼 요청합니다. 일반 웹 요청이 memory_enabled=false를 보내면 더미도 조회하지 않습니다.

```json
{
  "question": "보관료 면제 조건을 알려 주세요.",
  "mode": "hybrid_rerank",
  "embedding_profile": "bge",
  "memory_enabled": true,
  "user_id": "local-demo-user"
}
```

`dummy_preferences.py`가 실제 `search_memory` 호출을 대신해 `{"results":[{"id":"dummy-language","user_id":"...","memory":"한국어로 간결하게 답변해 주세요.","metadata":{"source":"dummy","scope":"expression_preference"}}, ...]}`를 반환합니다. 두 번째 항목은 비용 비교 시 표 형식 선호입니다. 모든 사용자는 같은 테스트 선호를 받되 요청 user_id로 구분하고 실제 사용자 기억이라고 취급하지 않습니다.

Mem0 임베딩·기억 벡터 검색·저장·갱신은 호출하지 않습니다. **문서용 질문 임베딩·Qdrant 검색·reranker·답변 생성은 계속 실행**합니다. 기존 기본값 `MEMORY_BACKEND=mem0`와 요청의 memory_enabled=false는 유지했습니다. 실제 기능을 붙일 때 backend를 mem0로 되돌리고 동일 dict/list 반환 계약을 유지하면 Context 코드를 바꿀 필요가 없습니다. 더미는 읽기 연결 전용이며 `/memory` 저장 API를 더미로 바꾸지 않습니다.

기록 중인 Langfuse trace에는 `memory_preference_fixture` 단계가 나타나며, 실행 manifest에 memory_backend가 기록됩니다. 실제 메시지의 사용자 표현 선호 영역과 PDF 검색 자료 영역이 분리되어 있는지 확인하세요. 더미 활성 경고는 response warnings에 포함됩니다. publish_metrics=true인 실행에서는 더미 선호 조회를 차단합니다. 과제 비교에서는 memory_enabled=false로 고정하거나 실제로 합의한 동일 선호를 사용하세요.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

외부 HTTP 없는 진단 24개, 기존 HTTP/검색/관측 회귀 테스트, 추가 엄격 출력/HTML escape/trace 단계/모델 결과 분리 검사를 실행합니다. 모의 모델 성공은 실제 LLM·Cloud 검증으로 보고하지 않습니다. 수정 후 기존 서버는 재시작해야 하며 prompt/source hash가 바뀌므로 최종 비교는 새로운 run_name에서 세 검색 방식에 같은 설정으로 실행하세요.
