# Langfuse 기록 사용법 (담당별)

기록 규칙의 원본은 [`docs/observability.md`](../../../docs/observability.md)다. 이 파일은 코드에서 부르는 방법만 적는다.

## import
- 다른 패키지는 `from backend.app.observability import ...`로만 가져온다. `observability.tracing`처럼 하위 모듈을 직접 import하면 `test_public_surface.py`가 실패한다.
- 공개 이름은 `__init__.py`의 `__all__`에 있다.

## 담당별 사용 예시

**자식 span (검색, 메모리, Context 담당).** 기존 방식 그대로다. Langfuse가 꺼져 있으면 `span`은 None이다.
```python
from backend.app.observability import observation, rank_list

with observation('dense_retrieval', input={'query': q}) as span:
    rows = search(q)
    if span:
        span.update(output=rows, metadata={'ranks': rank_list(rows)})   # rank_list: 순위 목록, 본문 없음
```

**서비스 질의 trace (pipeline).** `with trace:` 하나가 실행 범위(태그, session)와 루트 span `shipping_rag_query`를 함께 연다. 그래서 trace마다 kind 태그는 하나다. 오류 코드는 span이 닫히기 전에 `with trace:`가 기록하므로 `except`에서 따로 쓰지 않는다.
```python
from backend.app.observability import query_trace, ResolvedPrompt, resolve_prompt, observation, generation_parameters

# 지금(D5 연결 전): pipeline은 로컬 템플릿을 쓴다. prepare_context(prompt=)가 생기면 resolve_prompt로 바꾼다.
prompt = ResolvedPrompt.local('rag_answer', RAG_PROMPT)
# 나중: prompt = resolve_prompt('rag_answer', RAG_PROMPT, version=prompt_version, strict=run_kind != 'service')
trace = query_trace(manifest, kind=run_kind, question=request.question, memory_enabled=request.memory_enabled)
with trace:
    prepared = prepare_context(...)
    trace.record_context(prepared, source='retrieval')      # 루트 metadata: context_source, context_count, context_reduced
    with observation('answer_generation', kind='generation', input=messages, model=model,
                     prompt=prompt, model_parameters=generation_parameters(s)) as generation:
        ...
    trace.finish(response)                                   # 루트 output, status, citation_count, latency_ms
record.update(trace.flush(started))                          # finally에서: service_latency_ms, flush_ms
```
`kind`는 `service`, `comparison`, `evaluation`, `security`, `diagnostic` 중 하나다. 그 밖의 값은 `ValueError`.

**채점 trace (채점 담당).** 문항 하나에 `quality_evaluation` trace 하나. ragas와 judge span이 그 안에 들어간다.
```python
from backend.app.observability import evaluation_trace, publish_scores, flush_scores

with evaluation_trace(run_id=folder.name, question_id=item['question_id'], mode=item['mode'],
                      profile=item['profile'], source=item.get('response')):
    result = asyncio.run(evaluate_item(s, item, reference))
write_json(path, result)                    # 로컬 파일이 기준값이다. 먼저 쓴다
publish_scores(result, run_id=folder.name)  # row['trace_id'](서비스 trace)에 점수를 붙인다
...
flush_scores()
```

**점수 한 건 (보안 담당 등).** 이름은 `SCORE_NAMES`에 있어야 하고, 형식(NUMERIC, BOOLEAN, CATEGORICAL)은 `scores.SCORES` 표가 정한다. 형식이 맞지 않으면 보내지 않고 False를 돌려준다.
```python
from backend.app.observability import record_score
record_score(trace_id, 'security_case_status', 'defended', comment=reason)
```

**로그 (API).** `logs/<kind>.log`에 JSON 한 줄씩 쓴다. 허용 필드(`logs.LOG_FIELDS`) 밖의 필드는 버리고 키 이름만 경고한다. 질문, 답변 같은 원문 필드는 없다.
```python
from backend.app.observability import setup_logging, access_log, timed_lock, log_event
setup_logging('api')                     # lifespan 시작
app.middleware('http')(access_log)       # 모든 요청에 입구 로그 한 줄
with timed_lock(gpu_lock): ...           # lock_wait_ms
```

**조회 (보고서, 확인 스크립트).** 조회 함수는 Langfuse가 꺼져 있거나 실패하면 빈 목록을 돌려준다.
```python
from backend.app.observability import get_trace, list_traces, get_scores, trace_url, verify_trace
list_traces(run_id='core10-run', tags=['comparison'])
```

## 실서비스 답변 판정 (live judge)
- 실서비스 `/query` 답변이 출력 검증을 통과하면 `pipeline.py`가 `submit_live_judge(...)`로 판정을 예약한다. 응답은 기다리지 않는다.
- 백그라운드 스레드(최대 2개)가 OpenAI(`LIVE_JUDGE_MODEL`, 기본 gpt-6-luna)에 질문, 번호 붙은 최종 문맥, 답을 보낸다. 모델은 숫자를 매기지 않고 답변 문장마다 `supported`·`relevant`, 문맥마다 `used`만 판정한다. 비율은 코드가 계산한다(`live_judge.metrics`).
- 점수(NUMERIC 0~1): `live_faithfulness`(문맥이 뒷받침하는 문장 비율), `live_answer_relevancy`(RAGAS 방식: 답변만 보고 만든 질문 3개와 원래 질문의 bge-m3 코사인 유사도 평균. 답변 상태가 `answered`일 때만. 질문 생성은 별도 호출이고 span `live_judge_relevancy`), `live_context_precision`(쓰인 문맥의 순위별 평균 정밀도, RAGAS의 정답 없는 context precision), `live_decision`(judge가 질문과 문맥만 보고 고른 응답 종류 `answered`·`abstained`·`needs_clarification`이 실제 응답 상태와 같으면 1, 다르면 0. 배치 `judge_decision`의 정답 없는 판). 판정하지 못했을 때만 `live_judge_status`(`not_applicable`, `unscorable`). comment에는 그 지표를 깎은 항목을 코드가 적는다(뒷받침되지 않은 문장, 답변에서 만든 질문, 쓰인 문맥 번호, 기대·실제 응답 종류). 문장별 전체 판정은 trace의 span `live_judge`(루트 아래) 출력에 있다.
- 공격 문장 감시: 같은 판정에서 질문·문맥에 모델을 향한 지시문이 있는지(`live_injection_exposed`, 0/1)와 답변이 따랐는지(`live_injection_followed`, 노출됐을 때만)를 낸다. 노출된 질의는 `<runtime>/security_cases/<trace_id>.json`과 Langfuse Dataset `live_security_cases`(원본 trace 연결, `review: pending`)에 사례로 남는다. 사람이 검토해 기대 응답을 채우면 `security_tasks.json` 형식으로 옮겨 기존 보안 실행(base/defense 비교, `security_*` 점수)에 넣는다. 기존 보안 실행은 바뀌지 않는다.
- 로컬 RAGAS(`faithfulness`)와 계산 방식은 같지만 판정 모델이 달라 값을 그대로 비교하지 않는다. 지표 안정성 측정과 선택 근거: `docs/designs/observability/reports/live-judge-metric-stability.md`. 로컬 사본은 실행 폴더의 `live_judge.json`.
- 켜기(기본 꺼짐): `.env`에 `LIVE_JUDGE_ENABLED=true`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`. 주소나 키가 없으면 판정만 건너뛰고 `live_judge_not_configured` 경고를 한 번 남긴다. 질의는 그대로 처리된다. `FORCE_LOCAL_ONLY`, `ALLOW_EXTERNAL_LLM`(생성·배치 평가·Mem0 규칙)은 바꾸지 않는다. 외부 허용은 judge 호출 하나로 한정된다. `/health`의 `live_judge`는 실제로 판정할 때만 true다. 답변 생성 모델(`GENERATION_LLM_PROVIDER`)과는 따로 켜고 끈다.
- 실서비스 트래픽만 판정한다. `run_kind`가 `service`이고 진입점이 api·web인 질의다. 비교·보안·진단 실행, CLI(`03_query_once.py`, `check_integration.py`), Langfuse가 꺼진 경우는 판정하지 않는다. 판정의 bge-m3 임베딩은 API 요청과 같은 GPU 락(`providers.GPU_LOCK`)을 잡아 사용자 요청과 겹치지 않는다. 문맥이 없거나 인용 없는 되묻기는 `not_applicable`.
- 새 점수 이름이라 `setup --apply`로 score config를, `dashboards --apply`로 위젯 `[운영] 실서비스 답변 품질 지표`을 한 번 반영한다.
- 판정 대기열은 메모리에 있다. 서버를 끄면 남은 판정이 끝날 때까지 종료가 늦어진다(1건 최대 `LLM_TIMEOUT_SECONDS` × 시도 수). 강제 종료하면 남은 판정은 점수 없이 남는다.
- 판정 호출도 `llm_attempt` span을 만든다. `[운영] LLM 시도 수`와 `[운영] 단계별 지연 p95`에는 서비스 생성 시도와 판정 시도가 같이 잡힌다(비교 채점, 보안 판정도 같음). span 부모(`live_judge`)로 구분한다.

## 프롬프트 원본
- `PROMPT_SOURCE=local`(기본)이면 코드의 `RAG_PROMPT`를 쓰고 네트워크를 쓰지 않는다. `prompt_version`은 `prompts.PROMPT_VERSION`이다.
- `PROMPT_SOURCE=langfuse`면 Langfuse의 `rag_answer`를 가져온다. 가져오기에 실패하면 `source='fallback'`이 되고 60초 동안 다시 가져오지 않는다. 비교, 보안 실행(`strict=True`)에서는 실패로 끝난다.
- 두 모드 모두 `prompt_sha256`(system과 human 메시지 전체)과 태그 `prompt:<앞 8자>`를 남긴다. 프롬프트별 점수는 이 해시로 나눠 본다.

## Langfuse 설정
- `uv run python -m backend.app.observability.setup`은 미리보기만 한다(읽기).
- `--apply`는 Cloud에 프롬프트와 score config를 만든다. 실행마다 관측 담당 승인을 받는다(D8).
- 대시보드: `uv run python -m backend.app.observability.dashboards`(미리보기), `--apply`(생성). 대시보드 4개(`[관측] 서비스 운영`, `[관측] 평가 지표`, `[관측] 보안 지표`, `[관측] Context engineering`)와 위젯 25개를 `dashboards.py`의 표대로 만든다. 같은 이름이 있으면 건너뛰고, 정의가 바뀐 위젯만 갱신한다. 프로젝트 단위라 environment는 Langfuse 화면에서 고른다. SDK의 `api.unstable` 기능이라 Langfuse가 바꾸면 고쳐야 할 수 있다.
- 기록 확인: `uv run python -m backend.app.observability.cloud_check --apply --environment dev`. 실제 질의 1건(kind `diagnostic`, session `obs-cloud-check-<시각>`)을 보내고 다시 읽어 session, 태그 필터, 오류 코드(성공이면 필수 span), 같은 score id 덮어쓰기를 확인한다. 운영은 운영 키로 `--environment production`. 결과만 출력하고 키와 원문은 출력하지 않는다. Cloud에 쓰므로 실행마다 승인을 받는다.

## 주의
- 기록은 Cloud에 15~30초 뒤에 반영된다. 바로 조회하면 비어 있을 수 있다. `verify_trace`는 몇 번 다시 읽는다.
- 무료 요금제라면 Observations 조회가 분당 30회로 제한될 수 있다. 실행 하나(요청 360개)를 API로 모두 읽지 말고 로컬 파일을 쓴다.
- 구 API를 쓰지 않는다: `/api/public/traces`, `/api/public/v2/scores`, SDK v2의 `langfuse.trace()`. 조회는 `api.observations`(v2)와 `api.scores_v3`를 쓴다.
- 닫힌 span에 쓴 값은 버려진다. `with` 블록 안에서 `update`한다.
- 태그, session, trace metadata 값은 ASCII 200자 이하여야 한다. 넘으면 SDK가 조용히 버린다. `query_trace`는 그런 값을 빼고 키 이름만 경고한다.
- 테스트는 `backend/tests/observability/fake_langfuse.py`의 `FakeLangfuse().install(monkeypatch)`를 쓴다. 네트워크를 쓰지 않는다.
