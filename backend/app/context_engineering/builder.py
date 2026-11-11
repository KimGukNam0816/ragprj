"""검색/저장소/모델 호출 없이 최종 입력만 조립한다."""
import json

from ..artifacts import digest
from ..config import Settings
from ..contracts import Chunk, PreparedContext
from ..schemas import ServiceError
from .budget import estimate_input_tokens, generation_tokenizer, input_budget
from .prompts import RAG_PROMPT
from .prompts import PROMPT_VERSION
from .telemetry import context_stage


def _deduplicate(chunks):
    contexts, duplicate_ids, by_id, seen = [], [], {}, set()
    for c in chunks:
        if (not isinstance(c, dict) or any(not isinstance(c.get(k), str) or not c[k].strip()
                for k in ('chunk_id', 'book_id', 'text')) or
                type(c.get('page')) is not int or c['page'] < 1):
            raise ServiceError('context_input', 'Invalid chunk source/body fields', 422)
        # 같은 ID의 원문/출처/버전 충돌을 최신 값으로 덮어쓰지 않는다.
        identity = (c['book_id'], c['page'], c['text'],
                    json.dumps({k:c.get(k) for k in ('version','effective_date')}, sort_keys=True, default=str))
        if c['chunk_id'] in by_id and by_id[c['chunk_id']] != identity:
            raise ServiceError('context_conflict', 'Conflicting source/body for the same chunk ID', 422)
        by_id[c['chunk_id']] = identity
        if identity in seen:
            duplicate_ids.append(c['chunk_id'])
            continue
        seen.add(identity)
        contexts.append(dict(c))
    return contexts, duplicate_ids


def prepare_context(s: Settings, question: str, chunks: list[Chunk], *,
                    memory_text: str = '사용하지 않음', defense_addition: str | None = None,
                    tokenizer=None) -> PreparedContext:
    # 호출자의 후보/고정 Context를 수정하지 않는다. 기존 상위 순서를 유지한다.
    with context_stage('context_deduplication', metadata={'candidate_count':len(chunks)}) as span:
        contexts, duplicate_ids = _deduplicate(chunks)
        if span:
            span.update(output={'kept_chunk_ids':[c['chunk_id'] for c in contexts], 'duplicate_chunk_ids':duplicate_ids})
    warnings = [f'Duplicate chunk removed: {cid}' for cid in duplicate_ids]
    dropped = []
    with context_stage('context_budget', metadata={'provider':s.generation_llm_provider,
                       'measurement':'estimate_with_roles_schema_and_margin'}) as span:
        if tokenizer is None:
            model = s.local_llm_model if s.generation_llm_provider == 'ollama' else s.openai_model
            tokenizer = generation_tokenizer(s.generation_llm_provider, model)
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
        if span:
            span.update(output={'estimated_input_tokens':tokens, 'input_budget':input_budget(s),
                'output_reserve_tokens':s.llm_max_output_tokens, 'dropped_chunk_ids':dropped,
                'selected_chunk_ids':[c['chunk_id'] for c in contexts]})
    with context_stage('context_prompt', metadata={'prompt_version':PROMPT_VERSION}) as span:
        if span:
            span.update(output={'context_hash':digest(contexts), 'messages_hash':digest(messages),
                                'source_count':len(contexts), 'memory_used':memory_text != '사용하지 않음'})
    return PreparedContext(contexts, messages, tokens, digest(contexts), dropped, warnings)
