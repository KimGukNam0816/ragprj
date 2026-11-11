"""이정로: 저장소 담당 영역. 다른 폴더는 여기서 공개한 것만 가져다 쓴다.

사용자 기억(preference·category·note)은 get_memory_store(settings)가 돌려주는 저장소 객체로 읽고 쓴다(정의서 docs/storage.md).
"""

from ..config import Settings
from .redis_client import build_mem0_vector_store, check_redis, get_redis
from .schema import MemoryStore

# (DB 종류, 주소) → 준비가 끝난 저장소. 프로세스당 1개만 만들어 재사용
_stores_by_target: dict[tuple[str, str], MemoryStore] = {}


def get_memory_store(settings: Settings) -> MemoryStore:
    """REDIS_ENABLED=true면 RedisMemoryStore, 아니면 QdrantMemoryStore를 돌려준다. 두 객체의 메서드는 같다.

    처음 부를 때만 만들고 인덱스·컬렉션이 없으면 만든다(있으면 그대로). 이후에는 같은 객체를 돌려준다.
    """
    target = _get_store_target(settings)
    if target not in _stores_by_target:
        _stores_by_target[target] = _create_ready_store(settings)
    return _stores_by_target[target]


def _get_store_target(settings: Settings) -> tuple[str, str]:
    if settings.redis_enabled:
        return ("redis", settings.redis_vector_url.get_secret_value())
    return ("qdrant", settings.qdrant_url)


def _create_ready_store(settings: Settings) -> MemoryStore:
    store = _choose_store_class(settings).from_settings(settings)
    store.ensure()
    return store


def _choose_store_class(settings: Settings):
    if settings.redis_enabled:
        from .redis_client import RedisMemoryStore

        return RedisMemoryStore
    from .qdrant import QdrantMemoryStore

    return QdrantMemoryStore


__all__ = [
    "MemoryStore",
    "build_mem0_vector_store",
    "check_redis",
    "get_memory_store",
    "get_redis",
]
