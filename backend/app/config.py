from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore', case_sensitive=False)

    app_env: str = 'development'
    backend_host: str = '127.0.0.1'
    backend_port: int = 8000
    frontend_origin: str = 'http://127.0.0.1:3000'
    connection_timeout_seconds: float = 15
    data_dir: Path = ROOT / 'data'
    runtime_dir: Path = ROOT / 'runtime'
    openai_base_url: str = ''
    openai_api_key: SecretStr = SecretStr('')
    openai_model: str = 'gpt-6-luna'
    openai_org_id: str = ''
    openai_project_id: str = ''
    llm_timeout_seconds: float = 90
    llm_max_retries: int = Field(1, ge=0, le=1)
    ollama_base_url: str = 'http://127.0.0.1:11434'
    embedding_model: str = 'bge-m3'
    embedding_dimensions: int = 1024
    comparison_embedding_model: str = 'qwen3-embedding:0.6b'
    comparison_embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(4, ge=1, le=64)
    qdrant_url: str = 'http://127.0.0.1:6333'
    qdrant_api_key: SecretStr = SecretStr('')
    qdrant_docs_bge_collection: str = 'docs_bge_v1'
    qdrant_docs_qwen_collection: str = 'docs_qwen06_v1'
    qdrant_memory_collection: str = 'mem_users_bge_v1'
    memory_enabled: bool = True
    memory_backend: Literal['mem0', 'dummy'] = 'mem0'
    mem0_embedding_model: str = 'bge-m3'
    mem0_embedding_dimensions: int = 1024
    mem0_history_db_path: Path = ROOT / 'runtime/mem0/history.db'
    memory_top_k: int = 3
    memory_context_max_tokens: int = 500
    # Mem0가 점수를 돌려줄 때만 적용한다. 점수 없는 응답은 그대로 통과시킨다.
    memory_min_relevance_score: float = 0.3
    memory_preference_max_chars: int = 200
    reranker_model: str = 'Qwen/Qwen3-Reranker-0.6B'
    reranker_device: str = 'auto'
    reranker_batch_size: int = 2
    reranker_max_length: int = 2048
    reranker_warmup_on_startup: bool = True
    langfuse_enabled: bool = True
    langfuse_base_url: str = ''
    langfuse_public_key: str = ''
    langfuse_secret_key: SecretStr = SecretStr('')
    prompt_source: Literal['local', 'langfuse'] = 'local'
    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    rrf_k: int = 60
    rerank_candidate_k: int = 20
    final_top_k: int = 4
    context_max_tokens: int = 8000
    chunk_tokens: int = 600
    chunk_overlap_tokens: int = 100
    connection_test_user_id: str = 'connection-test-user'
    execution_stage: str = 'local_smoke'
    generation_llm_provider: str = 'ollama'
    evaluation_llm_provider: str = 'ollama'
    mem0_llm_provider: str = 'ollama'
    local_llm_model: str = 'qwen3:4b'
    local_evaluation_model: str = 'qwen3:4b'
    local_mem0_model: str = 'qwen3:4b'
    allow_external_llm: bool = False
    # 관측 담당(2026-11-12): 실서비스 답변을 응답 직후 OpenAI judge로 채점해 같은 Langfuse trace에 점수를 붙인다.
    # 기본은 꺼짐. 이 스위치 하나와 OpenAI 주소·키로만 켠다. FORCE_LOCAL_ONLY·ALLOW_EXTERNAL_LLM은 RAG 파이프라인
    # (생성, 배치 평가, Mem0)의 규칙이라 건드리지 않는다. 주소·키가 없으면 판정만 건너뛰고 경고를 남긴다(질의는 그대로).
    # 실제로 판정할 때만 /health의 live_judge가 true다.
    live_judge_enabled: bool = False
    live_judge_model: str = 'gpt-6-luna'
    llm_max_output_tokens: int = Field(2048, ge=32, le=8192)
    local_context_tokens: int = 16384
    publish_metrics: bool = False
    metrics_device: str = 'rtx5060'
    force_local_only: bool = True
    evaluation_dir: Path = ROOT / 'data/advanced/Advanced_shipping_part01of09/evaluation'
    eval_question_limit: int = 1
    eval_question_ids_file: str = ''
    smoke_run: bool = True
    eval_selection: str = 'core10'
    request_timeout_seconds: float = 120
    low_vram_mode: bool = False
    embedding_context_tokens: int | None = None

    def check_inference_policy(self):
        providers = [self.generation_llm_provider, self.evaluation_llm_provider, self.mem0_llm_provider]
        if any(p not in {'ollama', 'openai'} for p in providers):
            raise ValueError('LLM provider must be ollama or openai')
        if self.execution_stage not in {'local_smoke', 'api_core10', 'api_final60'}:
            raise ValueError('Invalid execution stage')
        # 외부 호출은 사용자가 두 설정을 명시적으로 변경한 경우만 허용한다.
        if 'openai' in providers and (self.force_local_only or not self.allow_external_llm):
            raise ValueError('External inference is disabled; no fallback is permitted')
        if self.evaluation_llm_provider == 'openai' or self.mem0_llm_provider == 'openai':
            raise ValueError('External evaluation and memory inference are disabled')
        if self.generation_llm_provider == 'openai' and not self.llm_ready():
            raise ValueError('OpenAI is disabled until base URL and API key are configured')
        if self.generation_llm_provider == 'ollama' and not self.local_llm_model:
            raise ValueError('Set LOCAL_LLM_MODEL before running')
        if self.publish_metrics:
            import subprocess
            if self.smoke_run:
                raise ValueError('Smoke runs cannot publish formal metrics')
            try:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                raise ValueError('Cannot verify the RTX5060 metric device') from None
            if gpu.returncode or '5060' not in gpu.stdout or '4090' in gpu.stdout:
                raise ValueError('Formal metrics require the agreed RTX5060 environment')

    def profile(self, name: str) -> tuple[str, int, str]:
        if name == 'bge':
            return self.embedding_model, self.embedding_dimensions, self.qdrant_docs_bge_collection
        if name == 'qwen':
            return self.comparison_embedding_model, self.comparison_embedding_dimensions, self.qdrant_docs_qwen_collection
        raise ValueError('Unknown embedding profile')

    def llm_ready(self) -> bool:
        return bool(self.openai_base_url and self.openai_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
