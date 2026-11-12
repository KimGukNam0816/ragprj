"""시맨틱 캐시: 로컬 Redis Stack이 있을 때만 돈다(없으면 건너뜀). Ollama 없이 가짜 임베딩을 쓴다."""

import uuid

import pytest

from backend.app.storage.answer_cache import (
    _build_cache,
    _find_in_cache,
    _store_in_cache,
    build_cache_scope,
)

LOCAL_REDIS_URL = "redis://127.0.0.1:6379"
VECTOR_SIZE = 1024


def _fake_embed(text: str) -> list[float]:
    return [1.0] + [0.0] * (VECTOR_SIZE - 1)


def _unit_vector(axis: int) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[axis] = 1.0
    return vector


def _local_cache_or_skip():
    pytest.importorskip("redisvl")
    import redis

    try:
        redis.Redis.from_url(LOCAL_REDIS_URL, socket_connect_timeout=1).execute_command(
            "FT._LIST"
        )
    except redis.RedisError:
        pytest.skip("로컬 Redis Stack 없음")
    return _build_cache(
        LOCAL_REDIS_URL, 0.1, 60, _fake_embed, name="test_cache_" + uuid.uuid4().hex[:8]
    )


def test_similar_question_in_same_scope_returns_cached_response():
    cache = _local_cache_or_skip()
    scope = build_cache_scope("bge", "hybrid", "abcdef0123456789")
    response = {
        "answer": "면제 조건은 ...",
        "status": "answered",
        "request_id": "old",
        "latency_ms": 9000,
    }
    try:
        _store_in_cache(cache, "보관료 면제 조건은?", _unit_vector(0), scope, response)

        hit = _find_in_cache(cache, _unit_vector(0), scope)

        assert hit["is_hit"] is True
        assert hit["question"] == "보관료 면제 조건은?"
        assert hit["distance"] < 0.1
        assert hit["response"] == {
            "answer": "면제 조건은 ...",
            "status": "answered",
        }  # 요청별 값은 저장 안 함
    finally:
        cache.delete()


def test_far_question_or_other_scope_is_a_miss():
    cache = _local_cache_or_skip()
    scope = build_cache_scope("bge", "hybrid", "abcdef0123456789")
    try:
        _store_in_cache(
            cache, "보관료 면제 조건은?", _unit_vector(0), scope, {"answer": "a"}
        )

        far_question = _find_in_cache(cache, _unit_vector(1), scope)
        other_index = _find_in_cache(
            cache, _unit_vector(0), build_cache_scope("bge", "hybrid", "ffffffffffff")
        )
        other_mode = _find_in_cache(
            cache,
            _unit_vector(0),
            build_cache_scope("bge", "naive", "abcdef0123456789"),
        )

        assert (
            far_question["is_hit"] is False
        )  # 직각 벡터: 거리 1.0, 임계값 밖이지만 거리는 기록됨
        assert far_question["distance"] == pytest.approx(1.0)
        assert other_index is None and other_mode is None
    finally:
        cache.delete()
