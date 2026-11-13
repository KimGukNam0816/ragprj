from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    mode: Literal['direct', 'naive', 'hybrid', 'hybrid_rerank'] = 'naive'
    embedding_profile: Literal['bge', 'qwen'] = 'bge'
    memory_enabled: bool = False
    # 시연용 스위치. 서버의 SEMANTIC_CACHE_ENABLED가 꺼져 있으면 요청으로 켤 수 없다.
    use_semantic_cache: bool = True
    # Local development identity only. Replace with verified auth identity next week.
    user_id: str = Field(default='local-demo-user', pattern=r'^[A-Za-z0-9_-]{1,80}$')
    session_id: str = Field(default='local-demo', pattern=r'^[A-Za-z0-9_-]{1,80}$')

    @field_validator('question')
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('질문을 입력해 주세요.')
        return value


class Citation(BaseModel):
    book_id: str
    page: int = Field(ge=1)
    chunk_id: str


class ModelAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    status: Literal['answered', 'needs_clarification', 'abstained']
    reason: str = Field(default='',max_length=500)
    citations: list[Citation] = Field(default_factory=list,max_length=4)


class QueryResponse(ModelAnswer):
    request_id: str
    trace_id: str | None = None
    mode: str
    model: str
    latency_ms: int
    cache_hit: bool = False  # 시맨틱 캐시에서 돌려준 답이면 True(검색·LLM 생략)
    input_tokens: int | None = None
    output_tokens: int | None = None
    contexts: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    user_id: str = Field(default='local-demo-user', pattern=r'^[A-Za-z0-9_-]{1,80}$')


class ServiceError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 503):
        self.code, self.message, self.http_status = code, message, http_status
        super().__init__(message)
