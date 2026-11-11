"""입력 토큰 예산 계산 (문맥 길이 맞추기의 기준).

모델 창(LOCAL_CONTEXT_TOKENS 8,192) = 답변 자리(LLM_MAX_OUTPUT_TOKENS 2,048) + 입력 한도.
입력 한도 안에 지시문 + 질문 + 청크가 들어가야 하며, 넘으면 builder.py가 낮은 순위 청크부터 뺀다.
"""
from functools import lru_cache
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ── 토큰을 셀 토크나이저: 생성 모델과 같은 것을 써야 실제 길이와 맞음 ──
@lru_cache(maxsize=1)
def generation_tokenizer(provider='ollama', model=''):
    if provider == 'openai':
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # 미지원 모델은 근사값을 사용하고 실제 사용량은 API usage로 기록한다.
            return tiktoken.get_encoding('o200k_base')
    tokenizer = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3-4B', 'tokenizer.json'))
    tokenizer.no_truncation()
    return tokenizer


def estimate_input_tokens(tokenizer, messages, provider):
    """지시문 + 질문 + 청크 전체를 실제 토크나이저로 세고, 여유분 256토큰을 더한다."""
    tokenized = tokenizer.encode('\n'.join(m['content'] for m in messages))
    return len(tokenized.ids if provider == 'ollama' else tokenized) + 256


def input_budget(s):
    """입력 한도. 로컬은 모델 창(8,192)에서 답변 자리(2,048)를 먼저 떼고, CONTEXT_MAX_TOKENS(6,000)와 비교해 작은 값."""
    return min(s.context_max_tokens, s.local_context_tokens-s.llm_max_output_tokens) if s.generation_llm_provider == 'ollama' else s.context_max_tokens
