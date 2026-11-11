"""구조·상태·이번 요청의 인용 범위를 검증한다. 사실성 평가는 별도이다."""
import json
import re

from pydantic import ConfigDict, Field
from ..schemas import Citation, ModelAnswer
from .telemetry import context_stage


class _Citation(Citation):
    model_config = ConfigDict(extra='forbid', strict=True)


class _Answer(ModelAnswer):
    model_config = ConfigDict(extra='forbid', strict=True)
    citations: list[_Citation] = Field(default_factory=list, max_length=4)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def validate_answer(content: str, contexts):
    with context_stage('context_output_validation', metadata={'source_count':len(contexts)}) as span:
        text = content.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        data = json.loads(text, object_pairs_hook=_unique_object)
        answer = _Answer.model_validate(data)
        if not answer.answer.strip():
            raise ValueError('Answer must not be blank')
        valid = {(c['book_id'], c['page'], c['chunk_id']) for c in contexts}
        cited = [(c.book_id, c.page, c.chunk_id) for c in answer.citations]
        if any(c not in valid for c in cited):
            raise ValueError('Citation does not exist in actual model context')
        if len(set(cited)) != len(cited):
            raise ValueError('Duplicate citation')
        if answer.status == 'answered' and not answer.citations:
            raise ValueError('Answered result requires an actual source citation')
        # 본문의 표준 출처 표기가 있으면 citations와도 일치해야 한다.
        inline = re.findall(r'\[([A-Za-z0-9_-]+)\s+p\.\s*(\d+)\]', answer.answer)
        pages = {(c.book_id, c.page) for c in answer.citations}
        if any((book, int(page)) not in pages for book, page in inline):
            raise ValueError('Inline citation does not match declared citations')
        if span:
            span.update(output={'status':answer.status, 'citation_count':len(cited),
                                'reason_present':bool(answer.reason.strip()),
                                'validation':'passed', 'semantic_correctness':'not_evaluated'})
        return ModelAnswer.model_validate(answer.model_dump())
