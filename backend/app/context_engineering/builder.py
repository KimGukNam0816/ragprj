"""검색/저장소/모델 호출 없이 최종 입력만 조립한다."""
import json

from ..artifacts import digest
from ..config import Settings
from ..contracts import Chunk, PreparedContext
from ..schemas import ServiceError
from .budget import estimate_input_tokens, generation_tokenizer, input_budget
from .prompts import RAG_PROMPT


def prepare_context(s: Settings, question: str, chunks: list[Chunk], *,
                    memory_text: str = '사용하지 않음', defense_addition: str | None = None,
                    tokenizer=None) -> PreparedContext:
    # 호출자의 후보/고정 Context를 수정하지 않는다. 기존 상위 순서를 유지한다.
    contexts = [dict(c) for c in chunks]
    warnings, dropped = [], []
    if tokenizer is None:
        tokenizer = generation_tokenizer(s.generation_llm_provider, s.openai_model)
    while True:
        context = json.dumps([{k: c[k] for k in ['book_id', 'page', 'chunk_id', 'text']}
                              for c in contexts], ensure_ascii=False)
        messages = [{'role': 'system' if m.type == 'system' else 'user', 'content': m.content}
                    for m in RAG_PROMPT.format_messages(question=question, memory=memory_text, context=context)]
        if defense_addition:
            messages[0]['content'] += '\n' + defense_addition
        tokens = estimate_input_tokens(tokenizer, messages, s.generation_llm_provider)
        if tokens <= input_budget(s):
            break
        if not contexts:
            raise ServiceError('context_budget', 'Question/prompt exceeds token budget', 422)
        dropped.append(contexts.pop()['chunk_id'])
        warnings.append('A lower-ranked chunk was dropped to satisfy the input budget')
    return PreparedContext(contexts, messages, tokens, digest(contexts), dropped, warnings)
