"""사용자 기억 저장소 테스트.

- 규칙 테스트: DB 없이 값 검사, 오류 변환, id 규칙을 확인한다.
- 계약 테스트: 같은 시나리오를 로컬 Redis(6379)·Qdrant(6333)에 각각 돌린다. 로컬 DB가 없으면 건너뛴다(.env·Cloud는 쓰지 않음).
"""

import uuid

import pytest

from backend.app.config import Settings
from backend.app.schemas import ServiceError
from backend.app.storage import MemoryCategory, get_memory_store
from backend.app.storage.schema import MemoryStore, build_memory_id

LOCAL_REDIS_URL = "redis://127.0.0.1:6379"
LOCAL_QDRANT_URL = "http://127.0.0.1:6333"
LATER_TIME = "2030-01-01T00:00:00+00:00"


def _unit_vector(axis: int) -> list[float]:
    """axis 번째만 1인 1024차원 벡터. 같은 axis끼리 유사도 1, 다른 axis끼리 0."""
    vector = [0.0] * 1024
    vector[axis] = 1.0
    return vector


def _new_user_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"  # 하이픈이 들어간 id도 함께 확인됨


# ---- 규칙 테스트 (DB 없음) ----


class _FakeDbError(Exception):
    pass


class _RecordingStore(MemoryStore):
    """DB 대신 저장 요청만 기록하는 저장소. 값 검사가 DB에 닿기 전에 끝나는지 본다."""

    db_error_types = (_FakeDbError,)

    def __init__(self, error: Exception | None = None):
        self.saved = []
        self.error = error

    def _create_index_if_missing(self):
        return False

    def _save_memory(self, user_id, text, vector, memory_type, category, source):
        if self.error:
            raise self.error
        self.saved.append((memory_type, category, source))
        return "saved-id"

    def _list_memories(self, user_id, kinds):
        return []

    def _find_similar_notes(self, user_id, question_vector, top_k):
        return []

    def _update_memory_fields(self, user_id, memory_id, fields, vector):
        return False

    def _mark_memories_used(self, user_id, memory_ids, used_at_seconds):
        return 0

    def _delete_memory(self, user_id, memory_id):
        return False

    def _delete_all_memories_of_user(self, user_id):
        return 0


def test_each_save_function_stores_its_own_kind_only():
    store = _RecordingStore()

    store.save_preference("u", "3줄 이내로", _unit_vector(0), is_user_saved=True)
    store.save_category("u", MemoryCategory.ROUTE, "부산–LA", _unit_vector(0))
    store.save_category(
        "u", "carrier", "HMM", _unit_vector(0)
    )  # 글자도 목록에 있으면 허용
    store.save_note("u", "화요일 출항을 선호한다", _unit_vector(0))

    assert store.saved == [
        ("preference", None, "user_saved"),
        ("context", "route", "auto_extracted"),
        ("context", "carrier", "auto_extracted"),
        ("context", None, "auto_extracted"),
    ]


@pytest.mark.parametrize("bad_category", ["preefrence", "routee", "price", ""])
def test_save_category_rejects_unknown_category_before_saving(bad_category):
    store = _RecordingStore()

    with pytest.raises(ValueError):
        store.save_category("u", bad_category, "부산–LA", _unit_vector(0))

    assert store.saved == []


@pytest.mark.parametrize("bad_flag", ["user_saved", "auto_extracted", 1, None])
def test_save_rejects_is_user_saved_that_is_not_bool(bad_flag):
    store = _RecordingStore()

    with pytest.raises(TypeError):
        store.save_note("u", "note", _unit_vector(0), is_user_saved=bad_flag)

    assert store.saved == []


@pytest.mark.parametrize(
    "user_id, text, vector",
    [
        ("", "note", _unit_vector(0)),  # 빈 user_id는 전체 사용자 대상이 됨
        ("u", "", _unit_vector(0)),
        ("u", "note", [0.1] * 3),
    ],
)
def test_save_rejects_invalid_values_before_saving(user_id, text, vector):
    store = _RecordingStore()

    with pytest.raises(ValueError):
        store.save_preference(user_id, text, vector)

    assert store.saved == []


def test_db_error_becomes_service_error():
    store = _RecordingStore(error=_FakeDbError("connection refused"))

    with pytest.raises(ServiceError) as error_info:
        store.save_note("u", "note", _unit_vector(0))

    assert error_info.value.code == "memory_store_error"


def test_category_memory_id_is_fixed_per_user_and_category():
    assert build_memory_id("u", "route") == build_memory_id("u", "route")
    assert build_memory_id("u", "route") != build_memory_id("v", "route")
    assert build_memory_id("u", None) != build_memory_id("u", None)


# ---- 계약 테스트 (로컬 Redis·Qdrant) ----


@pytest.fixture(params=["redis", "qdrant"])
def store(request):
    settings = Settings(
        _env_file=None,
        redis_enabled=request.param == "redis",
        connection_timeout_seconds=2,
        redis_vector_url=LOCAL_REDIS_URL,
        qdrant_url=LOCAL_QDRANT_URL,
    )
    try:
        return get_memory_store(settings)
    except Exception as error:
        pytest.skip(f"로컬 {request.param} 없음: {type(error).__name__}")


@pytest.fixture
def seeded(store):
    """사용자 1명에게 preference 1, category 1, note 2(질문과 가까운 것·먼 것)를 저장하고, 끝나면 지운다."""
    user_id = _new_user_id("test")
    memory_ids = {
        "preference": store.save_preference(
            user_id, "3줄 이내로 답해주세요", _unit_vector(0), is_user_saved=True
        ),
        "category": store.save_category(
            user_id, MemoryCategory.ROUTE, "부산–LA", _unit_vector(0)
        ),
        "near_note": store.save_note(
            user_id, "화요일 출항을 선호한다", _unit_vector(0)
        ),
        "far_note": store.save_note(user_id, "보관료는 USD로 받는다", _unit_vector(1)),
    }
    yield store, user_id, memory_ids
    store.delete_all_memories_of_user(user_id)


@pytest.fixture
def other_user(store):
    """다른 사용자 1명(한글·점·@가 들어간 id)에게 note 1개를 저장하고, 끝나면 지운다."""
    user_id = _new_user_id("김대리.kim@corp")
    memory_id = store.save_note(user_id, "다른 사람 note", _unit_vector(0))
    yield user_id, memory_id
    store.delete_all_memories_of_user(user_id)


def test_list_preferences_and_categories_returns_only_those(seeded):
    store, user_id, memory_ids = seeded

    memories_by_id = _memories_by_id(store.list_preferences_and_categories(user_id))

    assert set(memories_by_id) == {memory_ids["preference"], memory_ids["category"]}
    assert memories_by_id[memory_ids["preference"]]["source"] == "user_saved"
    assert memories_by_id[memory_ids["category"]]["category"] == "route"


def test_list_notes_returns_only_notes(seeded):
    store, user_id, memory_ids = seeded

    notes_by_id = _memories_by_id(store.list_notes(user_id))

    assert set(notes_by_id) == {memory_ids["near_note"], memory_ids["far_note"]}
    near_note = notes_by_id[memory_ids["near_note"]]
    assert near_note["category"] is None
    assert near_note["source"] == "auto_extracted"
    assert (
        near_note["last_used_at"] == near_note["created_at"]
    )  # 사용 시각 시작값 = 저장 시각
    assert near_note["updated_at"] is None


def test_find_similar_notes_returns_only_similar_notes(seeded):
    store, user_id, memory_ids = seeded

    default_hits = store.find_similar_notes(user_id, _unit_vector(0))
    all_hits = store.find_similar_notes(
        user_id, _unit_vector(0), top_k=10, min_similarity=-1
    )

    # preference·category는 벡터가 같아도 빠지고, 유사도 0인 먼 note는 하한 0.5에서 빠짐
    assert [hit["id"] for hit in default_hits] == [memory_ids["near_note"]]
    assert [hit["id"] for hit in all_hits] == [
        memory_ids["near_note"],
        memory_ids["far_note"],
    ]
    assert all_hits[0]["similarity"] == pytest.approx(1, abs=1e-4)


def test_saving_same_category_overwrites_in_place(seeded):
    store, user_id, memory_ids = seeded
    before = _memories_by_id(store.list_preferences_and_categories(user_id))[
        memory_ids["category"]
    ]

    saved_id = store.save_category(user_id, "route", "부산–SEA", _unit_vector(2))
    memories_by_id = _memories_by_id(store.list_preferences_and_categories(user_id))

    assert saved_id == memory_ids["category"]
    assert len(memories_by_id) == 2
    after = memories_by_id[saved_id]
    assert after["memory"] == "부산–SEA"
    assert after["created_at"] == before["created_at"]
    assert after["last_used_at"] == before["last_used_at"]
    assert after["updated_at"] is not None


def test_mark_memories_used_updates_only_own_existing_memories(seeded, other_user):
    store, user_id, memory_ids = seeded
    _, other_memory_id = other_user
    missing_id = str(uuid.uuid4())

    used_count = store.mark_memories_used(
        user_id,
        [memory_ids["far_note"], missing_id, other_memory_id],
        used_at=LATER_TIME,
    )

    assert used_count == 1
    far_note = _memories_by_id(store.list_notes(user_id))[memory_ids["far_note"]]
    assert far_note["last_used_at"] == LATER_TIME


def test_mark_memory_user_saved_changes_only_source(seeded):
    store, user_id, memory_ids = seeded
    far_note_id = memory_ids["far_note"]

    is_marked = store.mark_memory_user_saved(user_id, far_note_id)
    far_note = _memories_by_id(store.list_notes(user_id))[far_note_id]

    assert is_marked is True
    assert (far_note["source"], far_note["type"], far_note["memory"]) == (
        "user_saved",
        "context",
        "보관료는 USD로 받는다",
    )
    assert far_note["updated_at"] is not None
    assert store.mark_memory_user_saved(user_id, str(uuid.uuid4())) is False


def test_update_memory_text_changes_search_result(seeded):
    store, user_id, memory_ids = seeded

    is_updated = store.update_memory_text(
        user_id, memory_ids["near_note"], "목요일 출항을 선호한다", _unit_vector(3)
    )

    assert is_updated is True
    hits = store.find_similar_notes(user_id, _unit_vector(3))
    assert [hit["memory"] for hit in hits] == ["목요일 출항을 선호한다"]
    assert store.find_similar_notes(user_id, _unit_vector(0)) == []


def test_other_users_memory_cannot_be_changed_or_deleted(seeded, other_user):
    store, user_id, _ = seeded
    other_user_id, other_memory_id = other_user

    is_updated = store.update_memory_text(
        user_id, other_memory_id, "바꿔치기", _unit_vector(5)
    )
    is_marked = store.mark_memory_user_saved(user_id, other_memory_id)
    is_deleted = store.delete_memory(user_id, other_memory_id)

    assert (is_updated, is_marked, is_deleted) == (False, False, False)
    other_note = _memories_by_id(store.list_notes(other_user_id))[other_memory_id]
    assert (other_note["memory"], other_note["source"]) == (
        "다른 사람 note",
        "auto_extracted",
    )


def test_delete_memory_deletes_only_once(seeded):
    store, user_id, memory_ids = seeded

    assert store.delete_memory(user_id, memory_ids["far_note"]) is True
    assert store.delete_memory(user_id, memory_ids["far_note"]) is False


def test_other_users_memories_are_not_mixed(seeded, other_user):
    store, _, _ = seeded
    other_user_id, _ = other_user

    other_hits = store.find_similar_notes(other_user_id, _unit_vector(0))

    assert [hit["memory"] for hit in other_hits] == ["다른 사람 note"]
    with pytest.raises(ValueError):
        store.list_notes("")  # 빈 user_id는 전체 사용자 검색이 되므로 거부


def test_delete_all_memories_of_user_deletes_every_memory(seeded):
    store, user_id, _ = seeded

    deleted_count = store.delete_all_memories_of_user(user_id)

    assert deleted_count == 4
    assert store.list_preferences_and_categories(user_id) == []
    assert store.list_notes(user_id) == []


@pytest.mark.parametrize("redis_enabled", [True, False])
def test_get_memory_store_raises_service_error_when_db_is_down(redis_enabled):
    """DB가 꺼져 있으면 멈춰 있지 않고 바로 ServiceError → 호출 쪽(memory)이 잡아서 경고만 남기면 된다(기획서 F2-3)."""
    pytest.importorskip("redis")
    settings = Settings(
        _env_file=None,
        redis_enabled=redis_enabled,
        connection_timeout_seconds=1,
        redis_vector_url="redis://127.0.0.1:1",  # 1번 포트: 아무것도 없음
        qdrant_url="http://127.0.0.1:1",
    )

    with pytest.raises(ServiceError) as error_info:
        get_memory_store(settings)

    assert error_info.value.code == "memory_store_error"


def _memories_by_id(memories: list[dict]) -> dict[str, dict]:
    return {memory["id"]: memory for memory in memories}
