from functools import lru_cache
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import json

from ..schemas import ServiceError


class _TokenizerUnmapped(ServiceError, ValueError):
    """미지원 로컬 생성 모델. ValueError도 상속해 기존 except와 호환된다."""


@lru_cache(maxsize=1)
def generation_tokenizer(provider='ollama', model=''):
    if provider == 'openai':
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # 미지원 모델은 근사값을 사용하고 실제 사용량은 API usage로 기록한다.
            return tiktoken.get_encoding('o200k_base')
    if model and not model.lower().startswith('qwen3:4b'):
        raise _TokenizerUnmapped('tokenizer_unmapped', 'Local tokenizer mapping is not configured for this model; inject its tokenizer', 502)
    tokenizer = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3-4B', 'tokenizer.json'))
    tokenizer.no_truncation()
    return tokenizer


TOKEN_MARGIN = 256


def estimate_input_tokens(tokenizer, messages, provider):
    from ..schemas import ModelAnswer
    # 역할·메시지 경계·출력 스키마도 센다. 서버 chat template와의 차이는 여유분으로 둔다.
    envelope = json.dumps({'messages': messages, 'schema': ModelAnswer.model_json_schema()}, ensure_ascii=False)
    tokenized = tokenizer.encode(envelope)
    ids = tokenized.ids if hasattr(tokenized, 'ids') else tokenized
    return len(ids) + TOKEN_MARGIN


def input_budget(s):
    # OpenAI context_max_tokens는 기존 '입력 한도' 계약을 유지한다.
    # 모델의 전체 context 한도로 가장하지 않는다. 실제 한도는 API/모델 설정에서 확인한다.
    return min(s.context_max_tokens, s.local_context_tokens-s.llm_max_output_tokens) if s.generation_llm_provider == 'ollama' else s.context_max_tokens
