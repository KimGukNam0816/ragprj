from ..schemas import ModelAnswer


def validate_answer(content: str, contexts):
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    answer = ModelAnswer.model_validate_json(text)
    valid = {(c['book_id'], c['page'], c['chunk_id']) for c in contexts}
    if any((c.book_id, c.page, c.chunk_id) not in valid for c in answer.citations):
        raise ValueError('Citation does not exist in actual model context')
    if answer.status == 'answered' and not answer.citations:
        raise ValueError('Answered result requires an actual source citation')
    return answer


