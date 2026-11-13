# 이정로 담당: storage

구현은 backend/app/storage/, 테스트는 backend/tests/storage/에서 관리합니다. 상세 공통 입출력·연결 방식·기존 경로 호환은 [팀 작업 안내](TEAM_WORKFLOW.md)를 참고하세요.

이 문서는 담당 기능 개선 내용과 실행 방법을 추가하는 전용 문서입니다. 공통 설정이나 다른 담당 폴더 변경이 필요하면 별도 통합 작업으로 분리합니다.

## 시맨틱 캐시(Redis Stack): main 머지 후 절차

비슷한 질문이 들어오면 문서 검색과 LLM 호출을 건너뛰고 Redis에 저장된 답을 돌려줍니다. 구현은 `backend/app/storage/answer_cache.py`, 연결은 `backend/app/storage/redis_client.py`입니다.

### 1. 캐시 없이 그냥 실행만 할 때

```powershell
git pull
uv sync --extra dev --extra memory   # redis, redisvl 패키지가 추가됐습니다
```

`.env`는 손대지 않아도 됩니다. `REDIS_VECTOR_URL`이 비어 있으면 캐시를 조용히 건너뛰고 예전과 똑같이 검색·생성합니다. 경고도 지연도 없습니다.

### 2. 캐시를 켤 때

1. 검색 기능이 있는 Redis Stack을 띄웁니다(일반 Redis는 안 됩니다).

   ```powershell
   docker run -d --name shipping-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v8
   ```

2. `.env`에 두 줄을 추가합니다(기존 `.env`는 덮어쓰지 말고 줄만 추가).

   ```
   SEMANTIC_CACHE_ENABLED=true
   REDIS_VECTOR_URL=redis://127.0.0.1:6379
   ```

   선택값: `SEMANTIC_CACHE_DISTANCE_THRESHOLD`(기본 0.1, 작을수록 엄격), `SEMANTIC_CACHE_TTL_SECONDS`(기본 3600), `SEMANTIC_CACHE_LLM_CHECK`(기본 true, 후보를 LLM에게 같은 질문인지 한 번 더 확인).

3. 백엔드를 다시 시작하고 연결을 확인합니다. 웹 설정 패널의 "연결 확인" 또는 아래 명령에서 `redis`가 `ok`(연결·검색 기능 확인)이면 됩니다.

   ```powershell
   uv run python -m scripts.00_check_connections
   ```

4. 같은 질문을 두 번 보내면 두 번째 답에 "캐시 응답"이 붙고, 경고 목록에 저장된 질문과 거리가 나옵니다.

### 알아 둘 것

- 기억(선호) 검색을 켠 요청, 공식 평가(`PUBLISH_METRICS=true`), 고정 문맥·방어 문구 실행은 캐시를 쓰지 않습니다.
- 인덱스를 다시 만들면 지문이 바뀌어 옛 답은 나오지 않습니다. 저장된 답은 TTL이 지나면 사라집니다.
- 숫자·날짜·문서 코드·회사명이 다른 질문은 벡터가 가까워도 hit로 보지 않습니다.
- Redis가 죽어 있으면 답변마다 "Semantic cache unavailable" 경고가 붙고 평소대로 답합니다. 컨테이너를 살리거나 `.env`에서 `REDIS_VECTOR_URL`을 비우면 됩니다.
