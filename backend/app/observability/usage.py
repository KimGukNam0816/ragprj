"""관측 값 변환. 미확인 usage는 0으로 만들지 않는다."""


def usage_details(generated):
    """Langfuse usage_details from a generation result. Unknown token counts are left out, never 0."""
    return {key: value for key, value in {
        'input': generated.get('input_tokens'), 'output': generated.get('output_tokens')
    }.items() if value is not None}


def generation_metadata(generated, *, provider, request_id):
    """Generation span metadata: the result without its content, plus provider and request_id."""
    # 원문은 output으로 분리하며 API 인증 헤더를 받지 않는다.
    return {'provider': provider, 'request_id': request_id,
            **{k: v for k, v in generated.items() if k != 'content'}}


def generation_parameters(s) -> dict:
    """model_parameters for the generation span, matching what Providers.chat sends (contract test guards drift)."""
    if s.generation_llm_provider == 'openai':
        return {'max_completion_tokens': s.llm_max_output_tokens}
    # temperature 0 is hardcoded in providers.py, not a setting.
    return {'temperature': 0, 'num_ctx': s.local_context_tokens, 'num_predict': s.llm_max_output_tokens}


def rank_list(rows: list[dict]) -> list[dict]:
    """Compact ranks for retrieval spans: rank, chunk_id, book_id, page, score. No text."""
    return [{'rank': i, 'chunk_id': r['chunk_id'], 'book_id': r['book_id'], 'page': r['page'],
             'score': r.get('rerank_score', r.get('score'))} for i, r in enumerate(rows, 1)]
