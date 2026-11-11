"""Redis 전용 테스트: 연결 점검, Mem0 설정, 인덱스 만들기, 사용 표시가 벡터를 다시 색인하지 않는지."""

import uuid
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.schemas import ServiceError
from backend.app.storage import build_mem0_vector_store, check_redis, get_memory_store

LOCAL_REDIS_URL = "redis://127.0.0.1:6379"


def test_check_redis_reports_disabled_and_missing_settings():
    disabled = Settings(_env_file=None, redis_enabled=False)
    missing_url = Settings(_env_file=None, redis_enabled=True, redis_vector_url="")

    assert check_redis(disabled)["status"] == "disabled"
    assert check_redis(missing_url)["status"] == "missing"


def test_mem0_vector_store_config_is_accepted_by_mem0():
    """Mem0 설정 클래스는 모르는 키를 거부하므로, Redis를 띄우지 않고도 설정 오타를 잡는다."""
    redis_db_config = pytest.importorskip(
        "mem0.configs.vector_stores.redis"
    ).RedisDBConfig
    settings = Settings(
        _env_file=None,
        redis_vector_url=LOCAL_REDIS_URL,
        redis_memory_index="idx:test",
        mem0_embedding_dimensions=1024,
    )

    config = build_mem0_vector_store(settings)

    assert config["provider"] == "redis"
    redis_db_config(**config["config"])


def test_create_index_if_missing_creates_index_once_and_never_overwrites():
    """Redis 없이 가짜 연결로: 없으면 만들고, 있으면 그대로 두고, 동시에 만들어진 경우도 정상 처리한다."""
    redis = pytest.importorskip("redis")
    from backend.app.storage.redis_client import RedisMemoryStore

    fake_redis = _FakeIndexRedis(redis)
    store = RedisMemoryStore(
        fake_redis
    )  # 연결을 밖에서 넣을 수 있어서 가짜로 바꿔 끼울 수 있음

    assert store.create_index_if_missing() is True
    create_args = fake_redis.created[0]
    assert create_args[:2] == ("FT.CREATE", "idx:shipping:memory:v2")
    assert "INDEXMISSING" in create_args and "HNSW" in create_args
    assert (
        "last_used_at" not in create_args
    )  # 질문마다 바뀌는 사용 시각은 벡터 문서에 두지 않음
    assert (
        "created_at" not in create_args
    )  # 숫자 칸으로 넣으면 검색 결과에서 정밀도가 잘림

    fake_redis.has_index = True
    assert store.create_index_if_missing() is False
    assert len(fake_redis.created) == 1

    fake_redis.has_index = False
    fake_redis.create_error = "Index already exists"
    assert store.create_index_if_missing() is False

    fake_redis.create_error = "Unknown argument"
    with pytest.raises(ServiceError):
        store.create_index_if_missing()  # 그 밖의 오류는 숨기지 않고 ServiceError로 알림


def test_mark_memories_used_does_not_reindex_vector_document():
    """로컬 Redis: 사용 표시는 Sorted Set만 고치므로 벡터 문서의 내부 번호(max_doc_id)가 그대로여야 한다."""
    pytest.importorskip("redis")
    store = _local_redis_store_or_skip()
    user_id = f"used-{uuid.uuid4().hex[:8]}"
    memory_id = store.save_note(user_id, "화요일 출항을 선호한다", [1.0] + [0.0] * 1023)
    try:
        doc_id_before = _max_doc_id(store)

        for _ in range(20):
            assert store.mark_memories_used(user_id, [memory_id]) == 1

        assert (
            _max_doc_id(store) == doc_id_before
        )  # 바뀌었다면 사용 표시마다 벡터까지 다시 색인된 것
    finally:
        store.delete_all_memories_of_user(user_id)


class _FakeIndexRedis:
    """create_index_if_missing()이 쓰는 명령만 흉내 내는 가짜 Redis 연결."""

    def __init__(self, redis_module):
        self.redis_module = redis_module
        self.has_index = False
        self.create_error = None
        self.created = []

    def register_script(self, script):
        return None

    def ft(self, index_name):
        return SimpleNamespace(info=self._info)

    def execute_command(self, *args):
        if self.create_error:
            raise self.redis_module.ResponseError(self.create_error)
        self.created.append(args)

    def _info(self):
        if not self.has_index:
            raise self.redis_module.ResponseError("Unknown index name")


def _local_redis_store_or_skip():
    settings = Settings(
        _env_file=None,
        redis_enabled=True,
        redis_vector_url=LOCAL_REDIS_URL,
        connection_timeout_seconds=2,
    )
    try:
        return get_memory_store(settings)
    except Exception as error:
        pytest.skip(f"로컬 Redis 없음: {type(error).__name__}")


def _max_doc_id(store) -> int:
    info = store.client.execute_command("FT.INFO", store.index_name)
    info_by_name = dict(zip(info[0::2], info[1::2]))
    return int(info_by_name["max_doc_id"])
