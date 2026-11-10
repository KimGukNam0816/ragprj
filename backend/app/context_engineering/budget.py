from functools import lru_cache
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


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
    tokenized = tokenizer.encode('\n'.join(m['content'] for m in messages))
    return len(tokenized.ids if provider == 'ollama' else tokenized) + 256


def input_budget(s):
    return min(s.context_max_tokens, s.local_context_tokens-s.llm_max_output_tokens) if s.generation_llm_provider == 'ollama' else s.context_max_tokens
