# 이정로 담당: storage

구현은 backend/app/storage/, 테스트는 backend/tests/storage/에서 관리합니다. 상세 공통 입출력·연결 방식·기존 경로 호환은 [팀 작업 안내](TEAM_WORKFLOW.md)를 참고하세요.

이 문서는 담당 기능 개선 내용과 실행 방법을 추가하는 전용 문서입니다. 공통 설정이나 다른 담당 폴더 변경이 필요하면 별도 통합 작업으로 분리합니다.

## 사용자 기억 저장소 정의서 (초안 v0.3)

Mem0 기억 기능 기획서 v0.3의 선호·맥락을 저장하는 규칙입니다. 기획서의 **slot은 category로**(`slot_key` → `category`), **source 값 explicit·auto는 user_saved·auto_extracted로** 이름을 바꿨습니다. 6절은 memory 담당(정김기보)과 확정할 항목입니다.

### 1. 원칙

- **기억 데이터를 DB에 읽고 쓰는 곳은 storage 저장소 객체뿐입니다.** memory 담당은 이 객체의 메서드만 호출하고, Mem0의 add·update·delete로 DB에 직접 쓰지 않습니다.
  - 이유: Mem0 update는 metadata를 통째로 바꿔서 type·category·source·last_used_at이 사라집니다(실제 Redis에서 확인).
- **공개 메서드는 할 일마다 하나씩이고, `MemoryStore`(schema.py)에 한 번만 정의합니다.** 값 검사와 오류 변환도 여기서 합니다. `RedisMemoryStore`·`QdrantMemoryStore`는 이를 물려받아 실제 읽기·쓰기(`_`로 시작하는 메서드)만 구현합니다. 그래서 두 DB의 사용법·검사·반환이 다를 수 없고, 부하 테스트도 같은 호출로 두 DB를 비교합니다.
  - 종류를 글자 인자로 받는 범용 저장 함수는 공개하지 않습니다. 정해진 종류 밖의 기억이나 오타(`preefrence`)를 넣을 길을 없애기 위함입니다.
  - 연결·Lua 스크립트·이름·HNSW 값은 객체를 만들 때 한 번 준비하고 메서드(`self.…`)에서 재사용합니다. 설정이 다른 객체를 여러 개 만들 수 있어 부하 테스트에서 HNSW 값을 바꿔 가며 비교할 수 있습니다.
- **벡터는 호출하는 쪽이 만들어 넘깁니다**(bge-m3, 1024차원). storage는 임베딩하지 않습니다. 부하 테스트에서 미리 만든 벡터로 DB만 재기 위함입니다.
- **자주 바뀌는 값은 벡터 문서와 분리합니다.** Redis는 HASH의 색인된 칸을 하나만 바꿔도 벡터까지 문서 전체를 다시 색인합니다(사용 시각 갱신 50번 → 내부 문서 번호 50 증가, 실측). 그래서 질문마다 바뀌는 `last_used_at`은 Redis에서 사용자별 Sorted Set에 둡니다.

**파일**

| 파일 | 내용 | 상태 |
|---|---|---|
| `backend/app/storage/schema.py` | 공개 메서드·값 검사·오류 변환을 가진 `MemoryStore`, 업무 항목 목록 `MemoryCategory`, category 고정 id·시각 변환 | 완료 |
| `backend/app/storage/redis_client.py` | Redis 연결, `RedisMemoryStore` | 완료 (이력 제외) |
| `backend/app/storage/qdrant.py` | Qdrant 연결, `QdrantMemoryStore` | 완료 (이력 제외) |
| `backend/app/storage/__init__.py` | `get_memory_store(settings)`: `REDIS_ENABLED`에 따라 두 저장소 중 하나를 돌려줌(프로세스당 1개) | 완료 |
| `backend/app/storage/history.py` | 변경 이력(4절). 두 구현이 같이 사용 | 다음 버전 |
| `backend/tests/storage/` | 같은 시나리오를 두 DB에 돌리는 테스트 + DB 없이 도는 단위 테스트 | 완료 |
| `evaluation/storage/locustfile.py` | 부하 테스트(5절) | 예정 |

memory 담당은 `from backend.app.storage import MemoryCategory, get_memory_store`만 가져옵니다. 어느 DB인지는 `REDIS_ENABLED`가 정하므로 memory 코드는 몰라도 됩니다(7절).

### 2. 기억 한 건

| 필드 | 형식 | 뜻 | Redis | Qdrant (point) |
|---|---|---|---|---|
| `id` | str (UUID) | 식별자 | HASH 키 `shipping:memory:<id>` | point id |
| `user_id` | str, 빈 값 금지 | 사용자 | HASH, TAG 색인 | payload, 사용자 분리용(tenant) 색인 |
| `memory` | str | 본문. 문장 또는 category 값 | HASH (색인 안 함) | payload |
| `type` | `preference` \| `context` | 주입 경로 | HASH, TAG 색인 | payload, keyword 색인 |
| `category` | `route` `equipment` `carrier` `cargo_type` `team` 또는 없음 | 있으면 category, 없으면 note | HASH, TAG 색인. 없으면 칸을 저장하지 않고 `ismissing`으로 검색 | payload, keyword 색인. 없으면 비어 있음 |
| `source` | `user_saved` \| `auto_extracted` | 사용자가 저장하라고 한 것(설정 패널 입력, "기억해" 명령)인지, 대화에서 자동으로 뽑은 것인지. 상한 삭제 보호 | HASH, TAG 색인 | payload, keyword 색인 |
| `created_at` `updated_at` | 함수 입출력은 ISO 8601(UTC) 글자, DB에는 초 단위 숫자 | 생성·수정 시각 | HASH (색인 안 함¹) | payload, float 색인 |
| `last_used_at` | 위와 같음 | 마지막 사용 시각 | **Sorted Set** `shipping:memory-used:<user_id>` (점수 = 시각) | payload, float 색인 |
| `embedding` | float32 × 1024 | 검색용 색인 | HASH, VECTOR (HNSW, cosine) | vector (HNSW, cosine) |

¹ 검색 조건이 아니라서 색인하지 않습니다. NUMERIC으로 색인하면 FT.SEARCH가 값을 유효숫자 12자리로 잘라 돌려줍니다(실측).

- **종류**: preference = `type=preference` / category = `type=context` + `category` 있음 / note = `type=context` + `category` 없음.
- **category는 사용자·항목마다 1개**: id를 (user_id, category)로 고정 생성(uuid5)합니다. 같은 항목에 다시 저장하면 자동으로 덮어쓰고(created_at·last_used_at 유지, updated_at 갱신), 동시에 저장해도 2개가 생기지 않습니다.
- **last_used_at 시작값 = created_at**: 새 기억이 저장 직후 "가장 오래 안 쓴 것"으로 지워지지 않게 합니다.
- **이름**: Redis 인덱스 `idx:shipping:memory:v2`(키 접두사 `shipping:memory:`), Qdrant 컬렉션 `shipping_memory_v2`. Mem0가 쓰는 이름(`idx:shipping:memory:v1` 등)과 겹치지 않게 합니다. Mem0는 시작할 때마다 자기 인덱스를 다시 만들기 때문입니다.
- **인덱스는 모든 사용자가 같이 쓰는 1개**이고 user_id로 거릅니다. 서버를 다시 켜도 유지되지만(Redis 스냅샷), 볼륨 없는 컨테이너를 지우면 사라집니다.
- **HNSW 값은 두 DB를 같게 둡니다**: M=16, ef_construction=200. 부하 테스트에서 바꿔 가며 측정합니다.

### 3. 메서드

`store = get_memory_store(settings)`로 받은 저장소 객체에서 부릅니다(7절). 반환하는 기억 dict는 2절 필드에서 `embedding`을 뺀 모양입니다.

| 할 일 | 메서드 | 성공하면 | 대상이 없으면 |
|---|---|---|---|
| 선호 저장 | `save_preference(user_id, text, vector, is_user_saved=False)` | id | – |
| 업무 항목 저장 | `save_category(user_id, category, text, vector, is_user_saved=False)` | id. 같은 항목이 있으면 그 자리를 덮어씀 | – |
| note 저장 | `save_note(user_id, text, vector, is_user_saved=False)` | id | – |
| 답변 전 선호·업무 항목 | `list_preferences_and_categories(user_id)` | 기억 목록(최대 100건, 순서 없음) | 빈 목록 |
| note 전부(정리용) | `list_notes(user_id)` | 기억 목록(최대 100건, 순서 없음) | 빈 목록 |
| 질문과 비슷한 note | `find_similar_notes(user_id, question_vector, top_k=3, min_similarity=0.5)` | 비슷한 순 목록 + `similarity`(0~1) | 빈 목록 |
| 본문 고치기 | `update_memory_text(user_id, memory_id, text, vector)` | `True` | `False` |
| 사용자 저장으로 바꾸기 | `mark_memory_user_saved(user_id, memory_id)` | `True` | `False` |
| 답변에 쓴 기억 표시 | `mark_memories_used(user_id, memory_ids, used_at=None)` | 바꾼 개수 | `0` |
| 1건 삭제 | `delete_memory(user_id, memory_id)` | `True` | `False` |
| 퇴사 처리 | `delete_all_memories_of_user(user_id)` | 지운 개수(복구 불가) | `0` |

- **결과 약속**: 함수는 동기 방식이고, 값이 돌아오면 DB 반영이 끝난 것입니다("진행 중" 상태 없음. Qdrant도 `wait=True`로 반영을 기다림).
- **실패**: DB 연결·요청 실패는 DB 종류와 상관없이 팀 공통 `ServiceError`(code `memory_store_error`) 하나로 냅니다.
- **잘못된 값은 DB에 닿기 전에 막습니다**: 빈 user_id·빈 본문·1024개가 아닌 벡터·목록에 없는 category는 `ValueError`, bool이 아닌 `is_user_saved`는 `TypeError`입니다. 빈 user_id를 막는 이유는 빈 조건이 전체 사용자 검색이 되기 때문입니다.
- **category**는 `MemoryCategory.ROUTE`처럼 목록에서 고릅니다(`"route"` 글자도 목록에 있으면 허용).
- **남의 기억은 건드릴 수 없습니다**: 수정·삭제·사용 표시는 모두 user_id를 받고, 다른 사용자의 id면 `False`/`0`입니다. 없는 id도 새로 만들지 않습니다.
- type·category·user_id는 바꿀 수 없습니다(지우고 새로 저장).
- 인덱스·컬렉션은 `get_memory_store`가 처음 한 번 `create_index_if_missing()`으로 만듭니다. 직접 부를 일은 없습니다.
- 상한(preference 5, note 30), 중복 판정(0.90/0.75), 가장 오래 안 쓴 note 고르기는 memory 담당 규칙입니다. 판단에 필요한 값은 위 조회 메서드로 모두 얻을 수 있습니다.

### 4. 변경 이력 (다음 버전, 아직 기록하지 않음)

- 저장·수정·삭제 메서드가 **스스로** 기록합니다. 호출하는 쪽이 따로 부르지 않으므로, 추가되어도 7절 사용법은 바뀌지 않습니다.
- 위치: `RUNTIME_DIR/memory/history.db` (SQLite, Git 제외). Redis·Qdrant 구현이 같은 모듈을 씁니다.

| 테이블 | 칸 | 보관 |
|---|---|---|
| `history` | id, memory_id, user_id, event(ADD/UPDATE/DELETE), old_memory, new_memory, reason, at | 본문(개인정보)이 있어 퇴사 처리 때 함께 삭제 |
| `audit` | id, action(`purge`), user_id, count, at | 본문 없음. 계속 보관 |

- 순서: 이력 기록 시작 → DB에 씀 → 성공하면 이력 확정, 실패하면 취소. 일어나지 않은 변경이 이력에 남지 않습니다.
- Mem0의 history.db는 user_id 칸이 없어 사용자 단위 삭제가 어렵기 때문에 쓰지 않습니다.

### 5. 부하 테스트 (Locust, Redis vs Qdrant)

**방식**

- Locust의 사용자(User)가 HTTP 대신 **3절 저장소 메서드를 직접 호출**하고, 호출마다 걸린 시간을 Locust에 보고합니다(request 이벤트). 같은 locustfile로 대상 DB만 바꿔 실행합니다.
- 실행: `uv run --with locust==<고정 버전> locust -f evaluation/storage/locustfile.py --headless -u <유저 수> -r <초당 생성 수> -t <시간> --csv <경로> --html <경로>`
  - `--with`는 실행할 때만 Locust를 설치하므로 공통 파일(pyproject.toml)을 바꾸지 않습니다.

**유저 1명의 동작** (질문 1번 기준, 기획서 UC-02·UC-03)

- 질의 보강: `list_preferences_and_categories` 1번 → `find_similar_notes` 1번 → `mark_memories_used` 1번(사용한 기억)
- 자동 추출이 있는 경우: `find_similar_notes` 1번(중복 확인) → `save_note` 1번
- 두 동작의 비율은 locustfile 위쪽 값으로 정하고 결과에 기록합니다.

**바꿔 가며 잴 값**

| 값 | Locust 옵션 | 무엇을 보나 |
|---|---|---|
| 유저 수 | `-u` | 동시에 붙는 사용자 수에 따른 지연·처리량 |
| 생성 속도 | `-r` | 초당 몇 명씩 늘리는지. 크게 하면 순간적으로 몰리는 상황(스파이크) |
| `between(a, b)` | `wait_time` | 요청 사이에 a~b초 무작위로 쉼(실제 사용자 흉내). 응답이 느려지면 요청 수도 함께 줄어듦 |
| `constant_throughput(x)` | `wait_time` | 유저 1명이 초당 **최대** x번 실행. 전체 목표 처리량 ≈ 유저 수 × x. 두 DB를 같은 요청량에서 비교할 때 사용 |

- 비교 원칙: 같은 데이터·유저 수·`-r`·`wait_time`에서 DB만 바꿉니다. 워밍업 구간은 집계에서 빼고, 조건마다 3번 반복합니다.
- 한계점 찾기: `constant_throughput`으로 목표 처리량을 단계적으로 올리면서 p95가 급히 늘거나 실패가 생기는 지점을 찾습니다.

**남길 기록** (`runtime/loadtest/<실행 이름>/`, Git 제외)

결과를 공유·게시할 수 있도록 실행마다 남기고, 게시용 요약 표만 따로 옮깁니다. 데이터는 합성 데이터만 씁니다.

- **환경**: CPU·메모리, Docker 제한(`--cpus`, `--memory`), 이미지 태그(redis/redis-stack-server:7.4.0-v8, qdrant/qdrant:v1.20.0), 패키지·Locust 버전
- **조건**: 사용자 수 × 사용자당 건수 = 전체 벡터 수, 차원, HNSW 값, `-u`·`-r`·`wait_time`, 동작 비율, 워밍업, 반복 횟수
- **Locust 출력**: `--csv`(요약·시간별·실패), `--html` 보고서
- **정밀 지연**: 요청별 지연시간 CSV(아래 함정 1)
- **요약 표**: p50/p95/p99 지연, 처리량, 실패율, recall@k(정확 계산 대비), DB 메모리, 적재 시간
- **한계**: 로컬 한 대, 합성 데이터 등

**측정 함정**

1. Locust는 100ms 미만 응답시간을 **1ms 단위로 반올림**해 백분위수를 계산합니다. 1ms 안팎인 DB 호출은 p50이 0이나 1로 뭉개지므로, locustfile에서 `time.perf_counter`로 잰 정밀 지연을 따로 CSV로 남기고 백분위수는 그것으로 계산합니다.
2. Locust 자체가 CPU를 다 쓰면(CPU 90% 이상 경고) DB가 아니라 Locust를 잰 것입니다. `--processes`로 늘리고, 경고가 난 실행은 무효로 합니다.
3. Redis 검색은 기본 500ms를 넘기면 부분 결과를 오류 없이 돌려줍니다(`search_query_timeout_ms:500`, `timeout_policy:return`). 측정할 때는 시간 초과를 실패로 처리하도록 바꿉니다.
4. Qdrant는 REST로 호출합니다. gRPC는 Locust가 쓰는 gevent와 함께 쓰려면 별도 설정이 필요하기 때문입니다. 프로토콜 차이(Redis 전용 프로토콜 vs HTTP)는 실제 사용 방식 그대로이므로 한계에 적습니다.
5. Locust와 DB가 같은 PC에서 CPU를 나눠 씁니다. 컨테이너 제한(`--cpus`)으로 DB 몫을 고정하고 한계에 적습니다.

### 6. 확인 필요 (memory 담당과)

1. Mem0는 문장에서 선호·맥락을 뽑고 판정하는 데만 쓰고, DB 쓰기는 storage 저장소로만 한다.
2. 변경 이력을 storage가 SQLite로 남기는 방식(4절).
3. 3절 메서드와 반환 모양으로 기획서 UC-01~08을 구현할 수 있는지.
4. 공통 설정 추가 요청(통합 작업): 기억용 Qdrant 주소 1개. 지금은 문서 검색과 같은 `QDRANT_URL`을 쓰는데, 부하 테스트는 로컬 컨테이너끼리 비교해야 하므로 분리가 필요합니다. 저장소 선택은 지금 있는 `REDIS_ENABLED`를 씁니다.
5. 부하 테스트 폴더 `evaluation/storage/` 생성(팀 확인). `evaluation/retrieval/`, `evaluation/security/`와 같은 규칙입니다.
6. 계약 `MemoryStore`를 공통 계약 파일(`backend/app/contracts.py`)로 옮길지(통합 작업). 지금은 `storage/schema.py`에 있습니다.

### 7. 사용법 (memory 담당용)

**준비** (한 번)

1. 패키지: `uv sync --extra memory` (redis, redisvl)
2. 로컬 DB(일반 터미널):
   ```powershell
   docker run -d --name shipping-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v8
   docker run -d --name shipping-qdrant -p 6333:6333 qdrant/qdrant:v1.20.0
   ```
3. `.env`
   - Redis: `REDIS_ENABLED=true`, `REDIS_VECTOR_URL=redis://127.0.0.1:6379`
   - Qdrant: `REDIS_ENABLED=false`. `QDRANT_URL`의 Qdrant에 `shipping_memory_v2` 컬렉션이 생깁니다(6절 4번).

**코드**

```python
from backend.app.config import get_settings
from backend.app.memory import memory_client
from backend.app.storage import MemoryCategory, get_memory_store

store = get_memory_store(get_settings())       # RedisMemoryStore 또는 QdrantMemoryStore. 처음 한 번만 만들고 재사용
embed = memory_client().embedding_model.embed  # 문장 → 1024차원 벡터(bge-m3). Mem0는 임베딩·추출에만 사용

# 저장 (UC-01, UC-02)
store.save_preference(user_id, '3줄 이내로 답해주세요', embed('3줄 이내로 답해주세요'), is_user_saved=True)
store.save_category(user_id, MemoryCategory.ROUTE, '부산–LA', embed('부산–LA'))  # 같은 항목이면 덮어씀
store.save_note(user_id, '화요일 출항을 선호한다', embed('화요일 출항을 선호한다'))

# 질의 보강 (UC-03): preference·category는 목록으로, note는 의미 검색으로
memories = store.list_preferences_and_categories(user_id)
preferences = [memory for memory in memories if memory['type'] == 'preference']
categories = [memory for memory in memories if memory['category']]
notes = store.find_similar_notes(user_id, embed(question), top_k=3, min_similarity=0.5)
store.mark_memories_used(user_id, [memory['id'] for memory in categories + notes])  # 실제로 쓴 기억만

# 중복 판정 (UC-04): 저장 전에 가장 비슷한 note의 similarity를 봄
same_notes = store.find_similar_notes(user_id, new_vector, top_k=1, min_similarity=0.75)

# 상한 (UC-05): 자동 추출 note 중 가장 오래 안 쓴 것
auto_notes = [note for note in store.list_notes(user_id) if note['source'] == 'auto_extracted']
oldest = min(auto_notes, key=lambda note: note['last_used_at'], default=None)

# 갱신·삭제 (UC-04, UC-07, UC-08)
store.update_memory_text(user_id, memory_id, '새 본문', embed('새 본문'))
store.mark_memory_user_saved(user_id, memory_id)
store.delete_memory(user_id, memory_id)
store.delete_all_memories_of_user(user_id)  # 퇴사: 전부 삭제하고 지운 개수 반환. 복구 불가
```

- 반환 dict: `id, user_id, memory, type, category, source, created_at, updated_at, last_used_at`(시각은 ISO 글자). `find_similar_notes`는 `similarity`가 더 붙습니다.
- 잘못된 값은 `ValueError`(형이 틀리면 `TypeError`)입니다(3절).
- DB에 연결할 수 없으면 `CONNECTION_TIMEOUT_SECONDS` 안에 `ServiceError`가 납니다. 기획서 F2-3대로 호출하는 쪽에서 잡아 경고만 남기고 답변은 계속 만드세요.
- **Mem0의 add·update·delete로 이 데이터에 쓰지 마세요**(1절).
- 테스트: `uv run python -m pytest backend/tests/storage -q`. 같은 시나리오를 로컬 Redis·Qdrant에 각각 돌립니다. 로컬 DB가 없으면 그 테스트만 건너뜁니다.
