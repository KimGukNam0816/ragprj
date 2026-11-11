"""Redis·Qdrant 저장소가 같이 쓰는 계약과 규칙(정의서 docs/storage.md 2·3절).

두 구현이 똑같이 동작하도록 값 검사, id 만들기, 시각 변환을 여기 한 곳에 둔다.
"""

import time
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Protocol

VECTOR_SIZE = 1024  # bge-m3 벡터 크기
HNSW_M = 16  # 두 DB에 같은 값을 써야 공정하게 비교됨
HNSW_EF_CONSTRUCTION = 200
MEMORY_TYPES = ("preference", "context")
CATEGORIES = (
    "cargo_type",
    "carrier",
    "equipment",
    "route",
    "team",
)  # 사용자·항목마다 값 1개
SOURCES = (
    "user_saved",
    "auto_extracted",
)  # 사용자가 저장하라고 한 것 / 대화에서 자동으로 뽑은 것
KINDS = (
    "preference",
    "category",
    "note",
)  # category = context + category 있음, note = context + category 없음


class MemoryStore(Protocol):
    """사용자 기억 저장소 계약. RedisMemoryStore와 QdrantMemoryStore가 같은 메서드를 가진다."""

    def ensure(self) -> bool: ...

    def save_memory(
        self,
        user_id: str,
        memory: str,
        vector: Sequence[float],
        memory_type: str,
        category: str | None = None,
        source: str = "auto_extracted",
    ) -> str: ...

    def list_memories(
        self, user_id: str, kinds: Iterable[str] | None = None
    ) -> list[dict]: ...

    def search_notes(
        self,
        user_id: str,
        vector: Sequence[float],
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[dict]: ...

    def update_memory(
        self,
        memory_id: str,
        memory: str | None = None,
        vector: Sequence[float] | None = None,
        source: str | None = None,
    ) -> bool: ...

    def touch(
        self, user_id: str, memory_ids: Iterable[str], used_at: str | None = None
    ) -> int: ...

    def delete_memory(self, memory_id: str) -> bool: ...

    def purge_user(self, user_id: str) -> int: ...


def check_user_id(user_id: str) -> None:
    """빈 user_id로 검색하면 전체 사용자가 대상이 되므로 막는다."""
    if not user_id:
        raise ValueError("user_id is required")


def check_vector(vector: Sequence[float]) -> None:
    """길이가 틀린 벡터는 Redis가 오류 없이 인덱스에서만 빼 버리므로 미리 막는다."""
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"vector must have {VECTOR_SIZE} numbers")


def check_record(user_id, memory, vector, memory_type, category, source) -> None:
    """저장할 기억 한 건이 정의서 규칙에 맞는지 확인한다."""
    check_user_id(user_id)
    check_vector(vector)
    _check_memory_text(memory)
    _check_choice("memory_type", memory_type, MEMORY_TYPES)
    _check_choice("source", source, SOURCES)
    _check_category(memory_type, category)


def build_update_fields(memory, vector, source) -> dict:
    """update_memory 인자를 확인하고, 바꿀 값(memory·source)만 담은 dict를 돌려준다.

    벡터는 DB마다 저장 방식이 달라 각 구현이 따로 처리한다.
    """
    if memory is None and vector is None and source is None:
        raise ValueError("nothing to update")
    if memory is not None and vector is None:
        raise ValueError(
            "vector is required when memory changes"
        )  # 본문과 벡터가 어긋나면 검색이 틀어짐
    if vector is not None:
        check_vector(vector)
    fields = {}
    if memory is not None:
        _check_memory_text(memory)
        fields["memory"] = memory
    if source is not None:
        _check_choice("source", source, SOURCES)
        fields["source"] = source
    return fields


def parse_kinds(kinds: Iterable[str] | None) -> tuple[str, ...]:
    """kinds 인자를 확인해 튜플로 돌려준다. None이면 모든 종류."""
    if kinds is None:
        return KINDS
    kinds = tuple(kinds)
    if not kinds or not set(kinds) <= set(KINDS):
        raise ValueError(f"kinds must be chosen from {list(KINDS)}")
    return kinds


def parse_top_k(top_k: int) -> int:
    """검색 개수를 확인해 정수로 돌려준다."""
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1")
    return int(top_k)


def build_memory_id(user_id: str, category: str | None) -> str:
    """category 기억은 (user_id, category)마다 항상 같은 id라서 다시 저장하면 같은 자리를 덮어쓴다.

    그 외 기억은 무작위 id를 쓴다.
    """
    if category is None:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"shipping-memory:{user_id}:{category}"))


def is_uuid(value) -> bool:
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    return True


def to_epoch_seconds(iso_time: str | None) -> float:
    """ISO 시각 글자를 DB에 저장할 초 단위 숫자로 바꾼다. 없으면 지금 시각."""
    if iso_time is None:
        return time.time()
    return datetime.fromisoformat(iso_time).timestamp()


def to_iso_time(epoch_seconds) -> str | None:
    """DB의 초 단위 숫자를 함수가 돌려줄 ISO 시각 글자(UTC)로 바꾼다. 값이 없으면 None."""
    if epoch_seconds in (None, ""):
        return None
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat()


def _check_memory_text(memory) -> None:
    if not memory:
        raise ValueError("memory is required")


def _check_choice(name: str, value, choices: tuple) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}")


def _check_category(memory_type: str, category: str | None) -> None:
    """category는 context 기억에만 있고, 정해진 항목 중 하나여야 한다."""
    if category is None:
        return
    if memory_type != "context" or category not in CATEGORIES:
        raise ValueError(
            f"category must be one of {list(CATEGORIES)} with memory_type=context"
        )
