"""시맨틱 캐시 hit 경로: 검색·LLM 없이 저장된 응답으로 답한다(Redis·Ollama 없이 가짜 조회로)."""

import json

import pytest
from pydantic import SecretStr

from backend.app.schemas import QueryRequest
from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401  팀 공용 오프라인 파이프라인 fixture

# 캐시를 켠 설정. 주소가 비어 있으면 파이프라인이 캐시를 건너뛰므로 가짜 주소를 넣는다.
# model_copy는 값을 검증하지 않아 SecretStr로 직접 감싼다.
CACHE_ON = {
    "semantic_cache_enabled": True,
    "redis_vector_url": SecretStr("redis://test"),
}
CACHED_RESPONSE = {
    "answer": "캐시된 답",
    "status": "answered",
    "reason": "원문 확인",
    "citations": [],
    "mode": "hybrid",
    "model": "gpt-6-luna",
    "contexts": [
        {
            "book_id": "AS001",
            "page": 1,
            "chunk_id": "c1",
            "text": "면제는 별도 승인이 필요합니다.",
        }
    ],
    "input_tokens": 10,
    "output_tokens": 5,
}


def _chat_that_only_judges_same_question(self, messages, schema=None):
    """같은 질문 판정에는 true로 답하고, 답변 생성용 호출이 오면 실패시킨다."""
    if schema and "same_question" in schema.get("properties", {}):
        return {"content": '{"same_question": true}', "model": "judge"}
    pytest.fail("hit이면 답변 생성 LLM을 부르지 않아야 한다")


def test_cache_hit_answers_without_retrieval_or_llm(
    offline_pipeline, monkeypatch, tmp_path
):
    pipeline, settings = offline_pipeline
    settings = settings.model_copy(update=CACHE_ON)
    (tmp_path / "indexes").mkdir(exist_ok=True)
    (tmp_path / "indexes" / "active_bge.json").write_text(
        json.dumps({"fingerprint": "offline-v1"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        pipeline.Providers,
        "embed",
        lambda self, texts, profile, query=False: [[0.0] * 4],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "find_cached_answer",
        lambda *a: {
            "is_hit": True,
            "response": CACHED_RESPONSE,
            "question": "면제 조건?",
            "distance": 0.01,
            "miss_reason": None,
        },
    )
    monkeypatch.setattr(
        pipeline, "retrieve", lambda *a: pytest.fail("hit이면 검색을 건너뛰어야 한다")
    )
    monkeypatch.setattr(
        pipeline.Providers, "chat", _chat_that_only_judges_same_question
    )

    response, output = pipeline.run_query(
        settings,
        QueryRequest(question="면제 조건은?", mode="hybrid"),
        run_id="cache-hit",
    )

    assert response.cache_hit is True
    assert response.answer == "캐시된 답"
    assert any("semantic cache" in warning for warning in response.warnings)
    record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert record["semantic_cache"]["hit"] is True
    assert record["semantic_cache"]["scope"] == "bge:hybrid:offline-v1"


def test_cache_is_skipped_when_request_turns_it_off(
    offline_pipeline, monkeypatch, tmp_path
):
    pipeline, settings = offline_pipeline
    settings = settings.model_copy(update=CACHE_ON)
    (tmp_path / "indexes").mkdir(exist_ok=True)
    (tmp_path / "indexes" / "active_bge.json").write_text(
        json.dumps({"fingerprint": "offline-v1"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        pipeline,
        "find_cached_answer",
        lambda *a: pytest.fail("토글을 끄면 캐시를 조회하지 않아야 한다"),
    )

    response, output = pipeline.run_query(
        settings,
        QueryRequest(question="면제 조건은?", mode="hybrid", use_semantic_cache=False),
        run_id="cache-off",
    )

    assert response.cache_hit is False
    record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert "semantic_cache" not in record


def test_cache_is_skipped_when_redis_url_is_empty(offline_pipeline, monkeypatch):
    """캐시를 켰어도 REDIS_VECTOR_URL이 비면 조회 없이 평소대로 답한다(Redis 없는 팀원 보호)."""
    pipeline, settings = offline_pipeline
    settings = settings.model_copy(update={"semantic_cache_enabled": True})
    monkeypatch.setattr(
        pipeline,
        "find_cached_answer",
        lambda *a: pytest.fail("주소가 비면 캐시를 조회하지 않아야 한다"),
    )

    response, _ = pipeline.run_query(
        settings, QueryRequest(question="면제 조건은?", mode="hybrid"), run_id="no-url"
    )

    assert response.cache_hit is False
    assert not any("Semantic cache" in warning for warning in response.warnings)


def test_cache_candidate_is_a_miss_when_llm_says_different_question(
    offline_pipeline, monkeypatch, tmp_path
):
    """벡터·규칙을 통과해도 LLM이 다른 질문이라 하면 평소대로 검색·생성한다(품목만 바뀐 경우)."""
    pipeline, settings = offline_pipeline
    settings = settings.model_copy(update=CACHE_ON)
    (tmp_path / "indexes").mkdir(exist_ok=True)
    (tmp_path / "indexes" / "active_bge.json").write_text(
        json.dumps({"fingerprint": "offline-v1"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        pipeline.Providers,
        "embed",
        lambda self, texts, profile, query=False: [[0.0] * 4],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "find_cached_answer",
        lambda *a: {
            "is_hit": True,
            "response": CACHED_RESPONSE,
            "question": "케이블 드럼 50개 보관료는?",
            "distance": 0.09,
            "miss_reason": None,
        },
    )
    monkeypatch.setattr(pipeline, "save_answer", lambda *a: None)
    # 오프라인 fixture의 chat은 답변 JSON만 돌려주므로 판정 응답으로는 읽히지 않아 "다른 질문"이 된다.

    response, output = pipeline.run_query(
        settings,
        QueryRequest(question="철근 코일 50개 보관료는?", mode="hybrid"),
        run_id="cache-llm-miss",
    )

    assert response.cache_hit is False
    assert response.answer == "별도 승인이 필요합니다."  # 평소 생성 경로의 답
    record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert record["semantic_cache"]["miss_reason"] == "llm_says_different"
