"""검색/저장소/모델 호출 없이 최종 입력만 조립한다.

검색이 고른 상위 청크(k=4)를 프롬프트에 넣고, 입력 한도를 넘으면 순위가 낮은 청크부터 통째로 뺀다.
여기서 남은 청크가 final_context로 저장되고, Hit@4·Recall@4는 이 문맥(모델이 실제로 본 것)으로 계산한다.
"""
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
        # ── 1. 프롬프트 조립: 지시문 + 질문 + 청크(book_id·page·chunk_id·text만, 점수·정답은 넣지 않음) ──
        context = json.dumps([{k: c[k] for k in ['book_id', 'page', 'chunk_id', 'text']}
                              for c in contexts], ensure_ascii=False)
        messages = [{'role': 'system' if m.type == 'system' else 'user', 'content': m.content}
                    for m in RAG_PROMPT.format_messages(question=question, memory=memory_text, context=context)]
        if defense_addition:
            messages[0]['content'] += '\n' + defense_addition  # 보안 비교의 방어 문장

        # ── 2. 입력 전체 길이 확인: 한도 = min(6,000, 8,192 − 출력 2,048) ──
        tokens = estimate_input_tokens(tokenizer, messages, s.generation_llm_provider)
        if tokens <= input_budget(s):
            break

        # ── 3. 넘으면 가장 낮은 순위 청크를 통째로 빼고 다시 계산 (청크 중간을 자르지 않음) ──
        if not contexts:
            # 청크를 다 빼도 넘으면 질문 자체가 너무 김 → 422
            raise ServiceError('context_budget', 'Question/prompt exceeds token budget', 422)
        dropped.append(contexts.pop()['chunk_id'])
        warnings.append('A lower-ranked chunk was dropped to satisfy the input budget')
    return PreparedContext(contexts, messages, tokens, digest(contexts), dropped, warnings)
