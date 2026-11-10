"""공통 계약 v1. 변경 시 통합 담당자가 소비 모듈과 함께 검증한다."""
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, NotRequired, Protocol, TypedDict


class Chunk(TypedDict):
    chunk_id: str
    book_id: str
    page: int  # PDF 첫 페이지는 1
    text: str
    score: NotRequired[float]
    rerank_score: NotRequired[float]


class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str


class GenerationResult(TypedDict):
    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    attempts: list[dict[str, Any]]
    finish_reason: NotRequired[str | None]
    latency_ms: NotRequired[int]
    usage: NotRequired[dict[str, Any]]
    provider_request_id: NotRequired[str | None]
    response_id: NotRequired[str | None]


# Mem0 버전별 응답을 보존한다. pipeline이 표현 선호용 문자열로만 변환한다.
MemoryResult = dict[str, Any] | list[dict[str, Any]]


class RetrievalResult(NamedTuple):
    # 기존 selected, candidates, index = retrieve(...) 언패킹도 지원한다.
    selected: list[Chunk]
    candidates: list[Chunk]
    index: dict[str, Any]  # fingerprint, scope 필수. 기존 manifest 필드 보존.


@dataclass(frozen=True)
class PreparedContext:
    contexts: list[Chunk]
    messages: list[Message]
    estimated_input_tokens: int
    context_hash: str
    dropped_chunk_ids: list[str]
    warnings: list[str]


class InferenceProvider(Protocol):
    def embed(self, texts: list[str], profile: str, *, query: bool = False) -> list[list[float]]: ...
    def chat(self, messages: list[Message], schema: dict | None = None) -> GenerationResult: ...


class UserStore(Protocol):
    """미구현 Redis 연동의 계약. 현재 요청 경로에 자동 연결하지 않는다."""
    def get_user(self, user_id: str) -> dict[str, Any] | None: ...
    def save_user(self, user_id: str, data: dict[str, Any]) -> None: ...


class CacheStore(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...
    def delete(self, key: str) -> None: ...
