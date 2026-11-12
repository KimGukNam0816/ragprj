"""시맨틱 캐시 hit 경로: 검색·LLM 없이 저장된 응답으로 답한다(Redis·Ollama 없이 가짜 조회로)."""

import json

import pytest

from backend.app.schemas import QueryRequest
from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401  팀 공용 오프라인 파이프라인 fixture

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


def test_cache_hit_answers_without_retrieval_or_llm(
    offline_pipeline, monkeypatch, tmp_path
):
    pipeline, settings = offline_pipeline
    settings = settings.model_copy(update={"semantic_cache_enabled": True})
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
        pipeline.Providers,
        "chat",
        lambda *a, **k: pytest.fail("hit이면 LLM을 부르지 않아야 한다"),
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
    settings = settings.model_copy(update={"semantic_cache_enabled": True})
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
