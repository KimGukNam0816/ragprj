"""Redis·Qdrant 저장소가 같이 쓰는 계약과 규칙(정의서 docs/storage.md 2·3절).

다른 모듈은 MemoryStore의 공개 메서드만 쓴다. 값 검사와 오류 변환은 여기 한 곳에 두고,
각 DB 클래스는 실제 읽기·쓰기(_로 시작하는 추상 메서드)만 구현한다.
"""

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum

from ..schemas import ServiceError

VECTOR_SIZE = 1024  # bge-m3 벡터 크기
HNSW_M = 16  # 두 DB에 같은 값을 써야 공정하게 비교됨
HNSW_EF_CONSTRUCTION = 200


class MemoryCategory(StrEnum):
    """업무 항목. 사용자·항목마다 값 1개만 저장한다."""

    CARGO_TYPE = "cargo_type"
    CARRIER = "carrier"
    EQUIPMENT = "equipment"
    ROUTE = "route"
    TEAM = "team"


CATEGORIES = tuple(category.value for category in MemoryCategory)


class MemoryStore(ABC):
    """사용자 기억 저장소. 다른 모듈은 아래 공개 메서드만 쓴다.

    결과 약속: 값이 돌아오면 DB 반영이 끝난 것이다(진행 중 상태 없음).
    잘못된 값은 ValueError(형이 틀리면 TypeError), DB 연결·요청 실패는
    ServiceError("memory_store_error").
    """

    db_error_types: tuple[type[Exception], ...] = ()  # 각 DB 클래스가 채움

    def create_index_if_missing(self) -> bool:
        """인덱스(Qdrant는 컬렉션)가 없으면 만들고 True, 있으면 그대로 두고 False."""
        with self._raise_service_error_on_db_failure():
            return self._create_index_if_missing()

    def save_preference(
        self,
        user_id: str,
        text: str,
        vector: Sequence[float],
        is_user_saved: bool = False,
    ) -> str:
        """답변 방식 선호를 새로 저장하고 id를 돌려준다."""
        return self._check_and_save(
            user_id, text, vector, "preference", None, is_user_saved
        )

    def save_category(
        self,
        user_id: str,
        category: MemoryCategory,
        text: str,
        vector: Sequence[float],
        is_user_saved: bool = False,
    ) -> str:
        """업무 항목 값을 저장하고 id를 돌려준다. 같은 항목이 있으면 값을 바꾼다."""
        # 정해진 항목이 아니면 여기서 ValueError(오타 'routee'도 저장 전에 막힘)
        category_value = MemoryCategory(category).value
        return self._check_and_save(
            user_id, text, vector, "context", category_value, is_user_saved
        )

    def save_note(
        self,
        user_id: str,
        text: str,
        vector: Sequence[float],
        is_user_saved: bool = False,
    ) -> str:
        """업무 항목이 아닌 맥락 문장을 새로 저장하고 id를 돌려준다."""
        return self._check_and_save(
            user_id, text, vector, "context", None, is_user_saved
        )

    def list_preferences_and_categories(self, user_id: str) -> list[dict]:
        """답변 전에 넣을 선호·업무 항목을 전부 돌려준다. 없으면 빈 목록."""
        return self._check_and_list(user_id, ("preference", "category"))

    def list_notes(self, user_id: str) -> list[dict]:
        """note를 전부 돌려준다(오래 안 쓴 note 정리용). 없으면 빈 목록."""
        return self._check_and_list(user_id, ("note",))

    def find_similar_notes(
        self,
        user_id: str,
        question_vector: Sequence[float],
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """질문과 비슷한 note를 비슷한 순으로 최대 top_k건 돌려준다.

        similarity(0~1)가 min_similarity 미만인 note는 뺀다. 없으면 빈 목록.
        """
        check_user_id(user_id)
        check_vector(question_vector)
        checked_top_k = parse_top_k(top_k)
        with self._raise_service_error_on_db_failure():
            notes = self._find_similar_notes(user_id, question_vector, checked_top_k)
        return [note for note in notes if note["similarity"] >= min_similarity]

    def update_memory_text(
        self, user_id: str, memory_id: str, text: str, vector: Sequence[float]
    ) -> bool:
        """이 사용자 기억의 본문과 벡터를 바꾼다. 없거나 남의 기억이면 False."""
        check_user_id(user_id)
        _check_memory_text(text)
        check_vector(vector)  # 본문과 벡터가 어긋나면 검색이 틀어지므로 둘 다 받음
        fields = {"memory": text, "updated_at": time.time()}
        with self._raise_service_error_on_db_failure():
            return self._update_memory_fields(user_id, memory_id, fields, vector)

    def mark_memory_user_saved(self, user_id: str, memory_id: str) -> bool:
        """자동으로 뽑은 기억을 사용자가 저장한 것으로 바꾼다. 없거나 남의 것: False."""
        check_user_id(user_id)
        fields = {"source": "user_saved", "updated_at": time.time()}
        with self._raise_service_error_on_db_failure():
            return self._update_memory_fields(user_id, memory_id, fields, None)

    def mark_memories_used(
        self, user_id: str, memory_ids: Iterable[str], used_at: str | None = None
    ) -> int:
        """답변에 쓴 기억의 사용 시각을 바꾸고 바꾼 개수를 돌려준다.

        없는 id와 남의 id는 건너뛴다. used_at은 ISO 시각 글자(기본: 지금).
        """
        check_user_id(user_id)
        memory_ids = list(memory_ids)
        if not memory_ids:
            return 0
        used_at_seconds = to_epoch_seconds(used_at)
        with self._raise_service_error_on_db_failure():
            return self._mark_memories_used(user_id, memory_ids, used_at_seconds)

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """이 사용자 기억 1건을 지운다. 없거나 남의 기억이면 False."""
        check_user_id(user_id)
        with self._raise_service_error_on_db_failure():
            return self._delete_memory(user_id, memory_id)

    def delete_all_memories_of_user(self, user_id: str) -> int:
        """사용자의 기억을 전부 지우고 지운 개수를 돌려준다(퇴사 처리). 복구 불가."""
        check_user_id(user_id)
        with self._raise_service_error_on_db_failure():
            return self._delete_all_memories_of_user(user_id)

    # 아래 추상 메서드는 각 DB 클래스가 구현한다. 값 검사가 끝난 뒤에만 불린다.
    @abstractmethod
    def _create_index_if_missing(self) -> bool: ...

    @abstractmethod
    def _save_memory(
        self, user_id, text, vector, memory_type, category, source
    ) -> str: ...

    @abstractmethod
    def _list_memories(self, user_id: str, kinds: tuple[str, ...]) -> list[dict]:
        """kinds는 "preference"·"category"·"note" 중에서 고른다.

        category = context + category 있음, note = context + category 없음.
        """

    @abstractmethod
    def _find_similar_notes(self, user_id, question_vector, top_k) -> list[dict]:
        """similarity(0~1)가 붙은 note를 비슷한 순으로 돌려준다."""

    @abstractmethod
    def _update_memory_fields(self, user_id, memory_id, fields, vector) -> bool:
        """이 사용자 기억이면 fields(와 vector가 있으면 벡터)를 바꾸고 True."""

    @abstractmethod
    def _mark_memories_used(self, user_id, memory_ids, used_at_seconds) -> int: ...

    @abstractmethod
    def _delete_memory(self, user_id: str, memory_id: str) -> bool: ...

    @abstractmethod
    def _delete_all_memories_of_user(self, user_id: str) -> int: ...

    def _check_and_save(
        self, user_id, text, vector, memory_type, category, is_user_saved
    ) -> str:
        check_user_id(user_id)
        _check_memory_text(text)
        check_vector(vector)
        source = _to_source(is_user_saved)
        with self._raise_service_error_on_db_failure():
            return self._save_memory(
                user_id, text, vector, memory_type, category, source
            )

    def _check_and_list(self, user_id: str, kinds: tuple[str, ...]) -> list[dict]:
        check_user_id(user_id)
        with self._raise_service_error_on_db_failure():
            return self._list_memories(user_id, kinds)

    @contextmanager
    def _raise_service_error_on_db_failure(self) -> Generator[None, None, None]:
        """DB 라이브러리 예외를 팀 공통 ServiceError로 바꾼다(DB 종류와 무관하게)."""
        try:
            yield
        except self.db_error_types as error:
            raise ServiceError(
                "memory_store_error", "기억 저장소 요청 실패: " + type(error).__name__
            ) from None


def check_user_id(user_id: str) -> None:
    """빈 user_id로 검색하면 전체 사용자가 대상이 되므로 막는다."""
    if not user_id:
        raise ValueError("user_id is required")


def check_vector(vector: Sequence[float]) -> None:
    """길이가 틀린 벡터는 Redis가 오류 없이 인덱스에서만 빼 버리므로 미리 막는다."""
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"vector must have {VECTOR_SIZE} numbers")


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


def _to_source(is_user_saved: bool) -> str:
    """True/False만 받는다('auto_extracted' 같은 글자도 참이라 user_saved가 되므로)."""
    if not isinstance(is_user_saved, bool):
        raise TypeError("is_user_saved must be True or False")
    return "user_saved" if is_user_saved else "auto_extracted"
