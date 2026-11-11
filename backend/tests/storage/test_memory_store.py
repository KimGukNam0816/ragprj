"""사용자 기억 저장소 테스트.

- 규칙 테스트: DB 없이 값 검사와 id 규칙을 확인한다.
- 계약 테스트: 같은 시나리오를 로컬 Redis(6379)·Qdrant(6333)에 각각 돌린다. 로컬 DB가 없으면 건너뛴다(.env·Cloud는 쓰지 않음).
"""

import uuid

import pytest

from backend.app.config import Settings
from backend.app.storage import get_memory_store
from backend.app.storage.schema import build_memory_id, check_record

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


VALID_RECORD = {
    "user_id": "u",
    "memory": "m",
    "vector": _unit_vector(0),
    "memory_type": "context",
    "category": None,
    "source": "auto_extracted",
}


# ---- 규칙 테스트 (DB 없음) ----


def test_check_record_accepts_valid_record():
    check_record(**VALID_RECORD)


@pytest.mark.parametrize(
    "bad_values",
    [
        {"user_id": ""},
        {"vector": [0.1] * 3},
        {"memory": ""},
        {"memory_type": "note"},
        {"category": "route", "memory_type": "preference"},
        {"category": "price"},
        {"source": "user"},
        {"source": "explicit"},  # 예전 이름은 거부
        {"source": "auto"},
    ],
)
def test_check_record_rejects_invalid_values(bad_values):
    with pytest.raises(ValueError):
        check_record(**{**VALID_RECORD, **bad_values})


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
        "preference": store.save_memory(
            user_id,
            "3줄 이내로 답해주세요",
            _unit_vector(0),
            "preference",
            source="user_saved",
        ),
        "category": store.save_memory(
            user_id, "부산–LA", _unit_vector(0), "context", category="route"
        ),
        "near_note": store.save_memory(
            user_id, "화요일 출항을 선호한다", _unit_vector(0), "context"
        ),
        "far_note": store.save_memory(
            user_id, "보관료는 USD로 받는다", _unit_vector(1), "context"
        ),
    }
    yield store, user_id, memory_ids
    store.purge_user(user_id)


@pytest.fixture
def other_user(store):
    """다른 사용자 1명(한글·점·@가 들어간 id)에게 note 1개를 저장하고, 끝나면 지운다."""
    user_id = _new_user_id("김대리.kim@corp")
    memory_id = store.save_memory(user_id, "다른 사람 note", _unit_vector(0), "context")
    yield user_id, memory_id
    store.purge_user(user_id)


def test_list_memories_returns_every_saved_memory(seeded):
    store, user_id, memory_ids = seeded

    rows_by_id = _rows_by_id(store.list_memories(user_id))

    assert set(rows_by_id) == set(memory_ids.values())
    assert rows_by_id[memory_ids["category"]]["category"] == "route"
    near_note = rows_by_id[memory_ids["near_note"]]
    assert near_note["category"] is None
    assert (
        near_note["last_used_at"] == near_note["created_at"]
    )  # 사용 시각 시작값 = 저장 시각
    assert near_note["updated_at"] is None


def test_list_memories_filters_by_kind(seeded):
    store, user_id, memory_ids = seeded

    preference_and_category = store.list_memories(
        user_id, kinds=("preference", "category")
    )
    notes = store.list_memories(user_id, kinds=["note"])

    assert _ids(preference_and_category) == {
        memory_ids["preference"],
        memory_ids["category"],
    }
    assert _ids(notes) == {memory_ids["near_note"], memory_ids["far_note"]}
    with pytest.raises(ValueError):
        store.list_memories(user_id, kinds=["slot"])  # 예전 이름은 거부


def test_search_notes_returns_only_similar_notes(seeded):
    store, user_id, memory_ids = seeded

    default_hits = store.search_notes(user_id, _unit_vector(0))
    all_hits = store.search_notes(user_id, _unit_vector(0), top_k=10, min_similarity=-1)

    # preference·category는 벡터가 같아도 빠지고, 유사도 0인 먼 note는 하한 0.5에서 빠짐
    assert [hit["id"] for hit in default_hits] == [memory_ids["near_note"]]
    assert [hit["id"] for hit in all_hits] == [
        memory_ids["near_note"],
        memory_ids["far_note"],
    ]
    assert all_hits[0]["similarity"] == pytest.approx(1, abs=1e-4)


def test_saving_same_category_overwrites_in_place(seeded):
    store, user_id, memory_ids = seeded
    before = _rows_by_id(store.list_memories(user_id))[memory_ids["category"]]

    saved_id = store.save_memory(
        user_id, "부산–SEA", _unit_vector(2), "context", category="route"
    )
    rows_by_id = _rows_by_id(store.list_memories(user_id))

    assert saved_id == memory_ids["category"]
    assert len(rows_by_id) == 4
    after = rows_by_id[saved_id]
    assert after["memory"] == "부산–SEA"
    assert after["created_at"] == before["created_at"]
    assert after["last_used_at"] == before["last_used_at"]
    assert after["updated_at"] is not None


def test_touch_updates_only_own_existing_memories(seeded, other_user):
    store, user_id, memory_ids = seeded
    _, other_memory_id = other_user
    missing_id = str(uuid.uuid4())

    touched_count = store.touch(
        user_id,
        [memory_ids["far_note"], missing_id, other_memory_id],
        used_at=LATER_TIME,
    )

    assert touched_count == 1
    assert (
        _rows_by_id(store.list_memories(user_id))[memory_ids["far_note"]][
            "last_used_at"
        ]
        == LATER_TIME
    )


def test_update_memory_changes_only_given_fields(seeded):
    store, user_id, memory_ids = seeded
    far_note_id = memory_ids["far_note"]

    is_updated = store.update_memory(far_note_id, source="user_saved")
    row = _rows_by_id(store.list_memories(user_id))[far_note_id]

    assert is_updated is True
    assert (row["source"], row["type"], row["memory"]) == (
        "user_saved",
        "context",
        "보관료는 USD로 받는다",
    )
    assert row["updated_at"] is not None
    assert store.update_memory(str(uuid.uuid4()), source="auto_extracted") is False
    with pytest.raises(ValueError):
        store.update_memory(
            far_note_id, memory="새 본문"
        )  # 벡터 없이 본문만 바꾸면 거부


def test_update_memory_with_new_vector_changes_search_result(seeded):
    store, user_id, memory_ids = seeded

    store.update_memory(
        memory_ids["near_note"], memory="목요일 출항을 선호한다", vector=_unit_vector(3)
    )

    assert [hit["memory"] for hit in store.search_notes(user_id, _unit_vector(3))] == [
        "목요일 출항을 선호한다"
    ]
    assert store.search_notes(user_id, _unit_vector(0)) == []


def test_delete_memory_deletes_only_once(seeded):
    store, _, memory_ids = seeded

    assert store.delete_memory(memory_ids["far_note"]) is True
    assert store.delete_memory(memory_ids["far_note"]) is False


def test_other_users_memories_are_not_mixed(seeded, other_user):
    store, _, _ = seeded
    other_user_id, _ = other_user

    other_hits = store.search_notes(other_user_id, _unit_vector(0))

    assert [hit["memory"] for hit in other_hits] == ["다른 사람 note"]
    with pytest.raises(ValueError):
        store.list_memories("")  # 빈 user_id는 전체 사용자 검색이 되므로 거부


def test_purge_user_deletes_every_memory(seeded):
    store, user_id, _ = seeded

    deleted_count = store.purge_user(user_id)

    assert deleted_count == 4
    assert store.list_memories(user_id) == []


@pytest.mark.parametrize("redis_enabled", [True, False])
def test_get_memory_store_fails_fast_when_db_is_down(redis_enabled):
    """DB가 꺼져 있으면 멈춰 있지 않고 바로 예외 → 호출 쪽(memory)이 잡아서 경고만 남기면 된다(기획서 F2-3)."""
    pytest.importorskip("redis")
    settings = Settings(
        _env_file=None,
        redis_enabled=redis_enabled,
        connection_timeout_seconds=1,
        redis_vector_url="redis://127.0.0.1:1",  # 1번 포트: 아무것도 없음
        qdrant_url="http://127.0.0.1:1",
    )

    with pytest.raises(Exception):
        get_memory_store(settings)


def _rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def _ids(rows: list[dict]) -> set[str]:
    return {row["id"] for row in rows}
