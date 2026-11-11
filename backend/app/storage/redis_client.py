"""Redis 접근 코드를 모아 둔 파일.

- 연결: get_redis, check_redis
- Mem0 설정: build_mem0_vector_store
- 사용자 기억 저장소: RedisMemoryStore (공개 메서드는 schema.py의 MemoryStore, 정의서 docs/storage.md 3절)

redis·redisvl 패키지는 memory 묶음(uv sync --extra memory)에만 있어서, 패키지 없이도 서버가 뜨도록 함수 안에서 import한다.
"""

import itertools
import struct
import time
from functools import lru_cache

from ..config import Settings
from .schema import (
    CATEGORIES,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    VECTOR_SIZE,
    MemoryStore,
    build_memory_id,
    check_user_id,
    to_iso_time,
)

INDEX_NAME = "idx:shipping:memory:v2"  # Mem0 인덱스(v1)와 겹치지 않는 이름
KEY_PREFIX = "shipping:memory:"  # 이 접두사로 시작하는 HASH만 인덱스에 들어감(기억 1건 = HASH 1개)
USED_AT_KEY_PREFIX = "shipping:memory-used:"  # 사용자별 사용 시각 Sorted Set. 인덱스 접두사와 겹치지 않게
RETURN_FIELDS = (
    "user_id",
    "memory",
    "type",
    "category",
    "source",
    "created_at",
    "updated_at",
)  # 벡터는 안 돌려줌
MAX_LIST_SIZE = 100  # 기획상 사용자당 최대 40건
PURGE_BATCH_SIZE = 1000
KIND_QUERIES = {
    "preference": "@type:{preference}",
    "category": "@category:{" + "|".join(CATEGORIES) + "}",
    "note": "@type:{context} ismissing(@category)",  # category 칸이 없는 context
}

# 저장: 처음이면 created_at, 이미 있으면(category 덮어쓰기) updated_at을 쓰고, 사용 시각은 처음 한 번만 넣는다(NX).
SAVE_SCRIPT = """
local is_created = redis.call('HSETNX', KEYS[1], 'created_at', ARGV[1])
if is_created == 0 then redis.call('HSET', KEYS[1], 'updated_at', ARGV[1]) end
redis.call('HSET', KEYS[1], unpack(ARGV, 3))
redis.call('ZADD', KEYS[2], 'NX', ARGV[1], ARGV[2])
return is_created
"""
# 이 사용자의 기억일 때만 고친다(없는 키면 HGET이 false). 확인과 수정이 한 번에 일어나서
# 그 사이 지워진 키가 반쪽짜리로 되살아나지 않는다.
UPDATE_IF_OWNED_SCRIPT = """
if redis.call('HGET', KEYS[1], 'user_id') ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
return 1
"""
# 이 사용자의 기억일 때만 기억 HASH와 사용 시각 Sorted Set의 항목을 같이 지운다.
DELETE_IF_OWNED_SCRIPT = """
if redis.call('HGET', KEYS[1], 'user_id') ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[2])
return 1
"""


@lru_cache(maxsize=4)  # 같은 주소·타임아웃이면 연결 객체를 새로 만들지 않고 재사용
def _create_client(redis_url: str, timeout_seconds: float):
    import redis

    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,  # 응답을 bytes 대신 str로 받음
        socket_connect_timeout=timeout_seconds,  # 서버가 없을 때 무한 대기하지 않게
        socket_timeout=timeout_seconds,
    )


def get_redis(settings: Settings):
    """다 같이 쓰는 Redis 연결. redis-py가 안에 연결 풀을 두고 재사용한다."""
    redis_url = (
        settings.redis_vector_url.get_secret_value()
    )  # SecretStr에서 실제 주소를 꺼냄
    return _create_client(redis_url, settings.connection_timeout_seconds)


def check_redis(settings: Settings) -> dict:
    """Redis 연결과 검색 기능(FT.*)을 확인한다. 주소에 비밀번호가 있을 수 있어 메시지에 주소를 넣지 않는다."""
    if not settings.redis_enabled:
        return _build_check_result("disabled", "비활성")
    if not settings.redis_vector_url.get_secret_value():
        return _build_check_result("missing", "설정 필요: REDIS_VECTOR_URL")
    try:
        import redis
    except ImportError:
        return _build_check_result("missing", "uv sync --extra memory 필요")
    return _check_connection_and_search(get_redis(settings), redis)


def build_mem0_vector_store(settings: Settings) -> dict:
    """Mem0의 'redis' 저장소 설정. 저장 구조는 Mem0가 정한다(키 mem0:<인덱스 이름>:<id>, FLAT·cosine)."""
    return {
        "provider": "redis",
        "config": {
            "redis_url": settings.redis_vector_url.get_secret_value(),
            "collection_name": settings.redis_memory_index,  # Redis 인덱스 이름이자 키 접두사
            "embedding_model_dims": settings.mem0_embedding_dimensions,  # 임베딩 모델 차원과 같아야 함
        },
    }


class RedisMemoryStore(MemoryStore):
    """사용자 기억 저장소(Redis). 공개 메서드와 값 검사는 MemoryStore에 있다.

    - 기억 1건 = HASH 1개(본문·종류·벡터처럼 거의 안 바뀌는 칸) → 인덱스로 검색
    - 사용 시각 = 사용자별 Sorted Set(질문마다 바뀜) → 바꿔도 벡터를 다시 색인하지 않음
    연결·스크립트·이름은 만들 때 한 번 준비하고 메서드에서 재사용한다.
    """

    def __init__(
        self,
        client,
        index_name: str = INDEX_NAME,
        key_prefix: str = KEY_PREFIX,
        used_at_key_prefix: str = USED_AT_KEY_PREFIX,
        hnsw_m: int = HNSW_M,
        hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION,
    ):
        import redis

        self.db_error_types = (redis.RedisError,)  # 연결 실패·명령 오류의 공통 부모
        self.client = client
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.used_at_key_prefix = used_at_key_prefix
        self.hnsw_m = hnsw_m  # 부하 테스트에서 바꿔 가며 비교
        self.hnsw_ef_construction = hnsw_ef_construction
        self._save_script = client.register_script(
            SAVE_SCRIPT
        )  # 한 번만 준비하고 이후 EVALSHA로 실행
        self._update_script = client.register_script(UPDATE_IF_OWNED_SCRIPT)
        self._delete_script = client.register_script(DELETE_IF_OWNED_SCRIPT)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RedisMemoryStore":
        """설정의 Redis 주소로 저장소를 만든다."""
        return cls(get_redis(settings))

    def _create_index_if_missing(self) -> bool:
        if self._index_exists():
            return False
        return self._create_index()

    def _save_memory(self, user_id, text, vector, memory_type, category, source) -> str:
        """처음이면 created_at, category를 다시 저장하면 updated_at을 쓴다(SAVE_SCRIPT)."""
        memory_id = build_memory_id(user_id, category)
        fields = _build_memory_fields(
            user_id, text, vector, memory_type, category, source
        )
        self._save_script(
            keys=[self._build_memory_key(memory_id), self._build_used_at_key(user_id)],
            args=[time.time(), memory_id, *_flatten_fields(fields)],
        )
        return memory_id

    def _list_memories(self, user_id, kinds) -> list[dict]:
        """고른 종류만 Redis 안에서 걸러 최대 100건."""
        query = f"{_build_user_query(user_id)} ({_build_kinds_query(kinds)})"
        documents, used_at_by_id = self._search_with_used_at(
            user_id, query, "LIMIT", 0, MAX_LIST_SIZE
        )
        return [
            _to_memory_record(memory_id, fields, used_at_by_id)
            for memory_id, fields in documents
        ]

    def _find_similar_notes(self, user_id, question_vector, top_k) -> list[dict]:
        documents, used_at_by_id = self._search_with_used_at(
            user_id,
            _build_similar_notes_query(user_id, top_k),
            "PARAMS", 2, "vector", _vector_to_bytes(question_vector),
            "SORTBY", "distance",  # 거리가 작을수록 비슷함
            "LIMIT", 0, top_k,
            return_fields=(*RETURN_FIELDS, "distance"),
        )  # fmt: skip
        return [
            _to_note_record_with_similarity(memory_id, fields, used_at_by_id)
            for memory_id, fields in documents
        ]

    def _update_memory_fields(self, user_id, memory_id, fields, vector) -> bool:
        changed_fields = dict(fields)
        if vector is not None:
            changed_fields["embedding"] = _vector_to_bytes(vector)
        is_updated = self._update_script(
            keys=[self._build_memory_key(memory_id)],
            args=[user_id, *_flatten_fields(changed_fields)],
        )
        return bool(is_updated)

    def _mark_memories_used(self, user_id, memory_ids, used_at_seconds) -> int:
        """Sorted Set만 고치므로 벡터를 다시 색인하지 않는다."""
        used_at_key = self._build_used_at_key(user_id)
        pipeline = self.client.pipeline()  # MULTI/EXEC: 개수 세기와 갱신을 한 번에
        pipeline.zmscore(used_at_key, memory_ids)
        pipeline.zadd(
            used_at_key, dict.fromkeys(memory_ids, used_at_seconds), xx=True
        )  # XX: 이 사용자 Sorted Set에 이미 있는 id만 갱신(남의 id는 여기 없음)
        existing_scores, _ = pipeline.execute()
        return sum(score is not None for score in existing_scores)

    def _delete_memory(self, user_id, memory_id) -> bool:
        is_deleted = self._delete_script(
            keys=[self._build_memory_key(memory_id), self._build_used_at_key(user_id)],
            args=[user_id, memory_id],
        )
        return bool(is_deleted)

    def _delete_all_memories_of_user(self, user_id) -> int:
        user_query = _build_user_query(user_id)
        deleted_count = 0
        memory_keys = self._find_memory_keys(user_query)
        while memory_keys:  # 지우면 인덱스에서도 바로 빠지므로 빌 때까지 반복
            deleted_count += self.client.delete(*memory_keys)
            memory_keys = self._find_memory_keys(user_query)
        self.client.delete(self._build_used_at_key(user_id))
        return deleted_count

    def _build_memory_key(self, memory_id: str) -> str:
        return self.key_prefix + memory_id

    def _build_used_at_key(self, user_id: str) -> str:
        return self.used_at_key_prefix + user_id

    def _search_with_used_at(
        self, user_id: str, query: str, *options, return_fields=RETURN_FIELDS
    ):
        """FT.SEARCH와 사용 시각 조회를 한 번에 보낸다(왕복 1번).

        [(id, {칸: 값}), ...]과 {id: 사용 시각}을 돌려준다.
        """
        pipeline = self.client.pipeline(transaction=False)
        pipeline.execute_command(
            "FT.SEARCH", self.index_name, query,
            "RETURN", len(return_fields), *return_fields,
            *options,
            "DIALECT", 2,  # 파라미터($vector)와 ismissing을 쓰려면 2 이상
        )  # fmt: skip
        pipeline.zrange(self._build_used_at_key(user_id), 0, -1, withscores=True)
        search_result, used_at_scores = pipeline.execute()
        return _parse_search_result(search_result, self.key_prefix), dict(
            used_at_scores
        )

    def _find_memory_keys(self, user_query: str) -> list[str]:
        """사용자의 기억 키를 최대 PURGE_BATCH_SIZE개 찾는다."""
        search_result = self.client.execute_command(
            "FT.SEARCH", self.index_name, user_query,
            "NOCONTENT", "LIMIT", 0, PURGE_BATCH_SIZE, "DIALECT", 2,
        )  # fmt: skip
        return search_result[1:]  # [총개수, 키1, 키2, ...]

    def _index_exists(self) -> bool:
        import redis

        try:
            self.client.ft(self.index_name).info()
        except redis.ResponseError:  # 인덱스가 없으면 오류
            return False
        return True

    def _create_index(self) -> bool:
        import redis

        try:
            self.client.execute_command(
                "FT.CREATE", self.index_name, *self._build_index_schema()
            )
        except redis.ResponseError as error:
            if "already exists" in str(error):  # 다른 프로세스가 방금 만든 경우만 정상
                return False
            raise
        return True

    def _build_index_schema(self) -> list:
        """정의서 2절 구조. 질문마다 바뀌는 사용 시각, 검색 조건이 아닌 created_at·updated_at은 넣지 않는다.

        (created_at을 NUMERIC으로 넣으면 FT.SEARCH가 값을 유효숫자 12자리로 잘라 돌려준다.)
        """
        return [
            "ON", "HASH", "PREFIX", 1, self.key_prefix,
            "SCHEMA",
            "user_id", "TAG",
            "type", "TAG",
            "category", "TAG", "INDEXMISSING",  # note는 이 칸이 없음 → ismissing(@category)로 찾음
            "source", "TAG",
            "embedding", "VECTOR", "HNSW", 10,  # 10 = 뒤에 오는 설정 단어 개수
            "TYPE", "FLOAT32", "DIM", VECTOR_SIZE, "DISTANCE_METRIC", "COSINE",
            "M", self.hnsw_m, "EF_CONSTRUCTION", self.hnsw_ef_construction,
        ]  # fmt: skip


def _check_connection_and_search(client, redis_module) -> dict:
    """1단계 연결, 2단계 검색 기능(FT.*)을 확인한다."""
    try:
        client.ping()
    except redis_module.RedisError:
        return _build_check_result(
            "error", "연결 실패. Redis 실행 상태·주소를 확인하세요."
        )
    try:
        index_names = client.execute_command(
            "FT._LIST"
        )  # 일반 Redis면 unknown command 오류
    except redis_module.ResponseError:
        return _build_check_result(
            "error", "검색 기능 없음. Redis Stack 이미지인지 확인하세요."
        )
    return {
        **_build_check_result("ok", "연결·검색 기능 확인"),
        "indexes": sorted(index_names),
    }


def _build_check_result(status: str, message: str) -> dict:
    return {"status": status, "message": message}


def _build_user_query(user_id: str) -> str:
    """user_id 검색 조건. '-' 같은 특수문자 앞에 \\를 자동으로 붙인다(직접 쓰면 0건이 나옴)."""
    from redisvl.query.filter import Tag

    check_user_id(user_id)
    return str(Tag("user_id") == user_id)


def _build_kinds_query(kinds: tuple[str, ...]) -> str:
    """종류별 조건을 OR(|)로 묶는다. 예: (@type:{preference}) | (@category:{...})"""
    return " | ".join(f"({KIND_QUERIES[kind]})" for kind in kinds)


def _build_similar_notes_query(user_id: str, top_k: int) -> str:
    """이 사용자의 note 중 질문 벡터와 가까운 top_k개를 찾는 검색 조건."""
    note_query = f"{_build_user_query(user_id)} {KIND_QUERIES['note']}"
    return f"({note_query})=>[KNN {top_k} @embedding $vector AS distance]"


def _build_memory_fields(user_id, text, vector, memory_type, category, source) -> dict:
    fields = {
        "user_id": user_id,
        "memory": text,
        "type": memory_type,
        "source": source,
        "embedding": _vector_to_bytes(vector),
    }
    if category is not None:
        fields["category"] = category  # note는 이 칸을 아예 저장하지 않음
    return fields


def _vector_to_bytes(vector) -> bytes:
    """인덱스가 float32라서 같은 형식의 바이트(숫자 1개 = 4바이트)로 바꾼다."""
    return struct.pack(f"<{VECTOR_SIZE}f", *vector)


def _flatten_fields(fields: dict) -> list:
    """{칸: 값} → [칸, 값, 칸, 값, ...] (Redis 명령 인자 모양)"""
    return list(itertools.chain.from_iterable(fields.items()))


def _parse_search_result(
    search_result: list, key_prefix: str
) -> list[tuple[str, dict]]:
    """FT.SEARCH 결과 [총개수, 키1, [칸, 값, ...], 키2, [...], ...] → [(id, {칸: 값}), ...]"""
    keys = search_result[1::2]
    field_lists = search_result[2::2]
    documents = []
    for key, field_list in zip(keys, field_lists):
        memory_id = key.removeprefix(key_prefix)
        documents.append((memory_id, _unflatten_fields(field_list)))
    return documents


def _unflatten_fields(flat_pairs: list) -> dict:
    """[칸, 값, 칸, 값, ...] → {칸: 값}"""
    return dict(zip(flat_pairs[0::2], flat_pairs[1::2]))


def _to_memory_record(memory_id: str, fields: dict, used_at_by_id: dict) -> dict:
    """Redis에서 읽은 칸 → 정의서 모양 dict. 없는 칸(note의 category 등)은 None."""
    return {
        "id": memory_id,
        "user_id": fields.get("user_id"),
        "memory": fields.get("memory"),
        "type": fields.get("type"),
        "category": fields.get("category"),
        "source": fields.get("source"),
        "created_at": to_iso_time(fields.get("created_at")),
        "updated_at": to_iso_time(fields.get("updated_at")),
        "last_used_at": to_iso_time(used_at_by_id.get(memory_id)),
    }


def _to_note_record_with_similarity(
    memory_id: str, fields: dict, used_at_by_id: dict
) -> dict:
    similarity = 1 - float(fields["distance"])  # Redis cosine 거리 = 1 - 유사도
    return {
        **_to_memory_record(memory_id, fields, used_at_by_id),
        "similarity": similarity,
    }
