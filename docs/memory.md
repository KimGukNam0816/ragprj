# 정김기보 담당: memory

구현은 backend/app/memory/, 테스트는 backend/tests/memory/에서 관리합니다. 상세 공통 입출력·연결 방식·기존 경로 호환은 [팀 작업 안내](TEAM_WORKFLOW.md)를 참고하세요.

이 문서는 담당 기능 개선 내용과 실행 방법을 추가하는 전용 문서입니다. 공통 설정이나 다른 담당 폴더 변경이 필요하면 별도 통합 작업으로 분리합니다.

## 구성

| 파일 | 역할 | 외부 연결 |
|---|---|---|
| `mem0_client.py` | Mem0 설정 조립, 로컬 LLM·임베딩 어댑터 연결 | Ollama, Qdrant, SQLite |
| `service.py` | `search_memory` / `add_memory` | Mem0 경유 |
| `preferences.py` | Mem0 응답 → 프롬프트 선호 문자열 변환 | **없음** (순수 변환) |
| `records.py` | Mem0 응답 행·metadata 읽기 도우미 | **없음** |

`preferences.py`는 설정을 읽지 않고 네트워크도 쓰지 않습니다. 설정 객체를 인자로 받으므로 단위 테스트에서 그대로 실행할 수 있습니다.

## 프롬프트에 들어가는 값

`contracts.MemoryResult`는 Mem0 버전별 응답을 그대로 보존하고, 프롬프트에는 사용자가 밝힌 표현 선호 문장만 넣습니다. 이 변환을 `select_preferences`/`format_preferences`가 담당합니다.

```python
from backend.app.memory import format_preferences

format_preferences(settings, {'results': [
    {'id': '7f3c', 'hash': 'deadbeef', 'score': 0.82,
     'created_at': '2026-11-10T00:00:00Z', 'memory': '한국어로 짧게 답변'},
]})
# '- 한국어로 짧게 답변'
```

`id`, `hash`, `score`, `created_at`, `metadata`는 프롬프트에 넣지 않습니다. 원본 응답은 실행 기록 `result.json`의 `memory_results`에, 선별 근거는 `memory_selection`에 남으므로 나중에 어떤 기억이 왜 빠졌는지 확인할 수 있습니다.

남는 선호가 없으면 `'사용하지 않음'`(`memory.NO_PREFERENCE`)을 돌려줍니다. 이 값은 `prepare_context`가 `memory_used`를 판정하는 기준과 같은 문자열입니다.

### 선별 순서

1. 본문이 빈 기억 제거
2. `MEMORY_MIN_RELEVANCE_SCORE` 미만 제거 — **Mem0가 점수를 돌려줄 때만** 적용합니다. 점수 없는 응답(구버전·고정 응답)은 그대로 통과시킵니다.
3. 정규화 — 제어문자를 공백으로 바꾸고, chat template 경계(`<|…|>`, `[INST]`)는 `[제거됨]`으로 바꾸고, 모든 공백을 한 칸으로 접어 한 줄로 만듭니다. 줄 구조가 사라지므로 기억 안에서 가짜 프롬프트 구획(`PDF 검색 자료:`)을 만들 수 없습니다.
4. `MEMORY_PREFERENCE_MAX_CHARS` 초과분 자르기
5. 같은 문장 중복 제거
6. `MEMORY_TOP_K` 건수 제한
7. `MEMORY_CONTEXT_MAX_TOKENS` 토큰 예산 제한

정규화는 구조만 다루고 내용은 판단하지 않습니다. 지시문을 실행하지 않는 책임은 계속 `RAG_SYSTEM`과 `DEFENSE_ADDITION`이 집니다.

## 설정

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MEMORY_ENABLED` | `true` | 서버 차단 스위치. `false`면 `QueryRequest.memory_enabled=true`로도 켤 수 없고 경고만 남습니다 |
| `MEMORY_BACKEND` | `mem0` | `dummy`는 고정 응답이며 저장소가 아닙니다 |
| `MEMORY_TOP_K` | `3` | Mem0 검색 건수이자 프롬프트에 넣을 최대 건수 |
| `MEMORY_CONTEXT_MAX_TOKENS` | `500` | 선호 문자열 토큰 예산 |
| `MEMORY_MIN_RELEVANCE_SCORE` | `0.3` | 점수가 있는 기억의 하한 |
| `MEMORY_PREFERENCE_MAX_CHARS` | `200` | 기억 한 건의 최대 길이 |
| `MEM0_HISTORY_DB_PATH` | `./runtime/mem0/history.db` | 상대 경로는 `RUNTIME_DIR` 기준으로 풉니다 |

토큰 예산은 토크나이저를 주입하면 그것으로 세고, 없으면 한글 1자를 1토큰으로 보는 넉넉한 어림을 씁니다. 예산을 넘기는 쪽보다 기억을 한 건 덜 넣는 쪽을 택한 것입니다.

## dummy 백엔드의 쓰기

`MEMORY_BACKEND=dummy`에서 `POST /memory`는 422 `memory_backend_readonly`로 거절합니다. 읽기가 고정 응답을 쓰는 동안 쓰기만 실제 Mem0에 저장되면, 저장한 선호가 조회되지 않아 연결 점검 결과를 오해하게 됩니다.

## 실행

```bash
uv run pytest backend/tests/memory -q
```

`preferences.py` 테스트는 외부 연결이 없습니다. Mem0 실연동은 `scripts/check_integration.py`에서 확인합니다.

## 남은 작업

- 명시 선호를 지우는 경로(`delete`)가 없습니다. 현재는 Mem0가 `UPDATE`/`DELETE` 이벤트를 스스로 판단할 때만 갱신됩니다.
- `user_id`는 아직 검증된 인증 주체가 아니라 요청이 보낸 값입니다. 저장·조회 격리는 `scripts/check_integration.py`가 다른 `user_id`로 확인만 합니다.
- 선호 추출 프롬프트(`custom_fact_extraction_prompt`)의 품질 측정 기준이 없습니다.
