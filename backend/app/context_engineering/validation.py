"""구조·상태·이번 요청의 인용 범위를 검증한다. 사실성 평가는 별도이다."""
import json
import re

from pydantic import ConfigDict, Field, ValidationError
from ..schemas import Citation, ModelAnswer, ServiceError
from .telemetry import context_stage


class _Citation(Citation):
    model_config = ConfigDict(extra='forbid', strict=True)


class _Answer(ModelAnswer):
    model_config = ConfigDict(extra='forbid', strict=True)
    citations: list[_Citation] = Field(default_factory=list, max_length=4)


USER_MESSAGE = '답변 처리 중 오류가 발생했습니다. 저장된 실행 기록을 확인하세요.'


class OutputInvalid(ServiceError, ValueError):
    """모델 출력이 검증에 실패했다. reason은 검사 범주 하나다. ValueError도 상속해 기존 except와 호환된다.

    .message(사용자 응답)는 예전과 같은 일반 문구다. str(exc)는 검사 내용만 담고 모델 출력 값은 담지 않는다."""
    def __init__(self, reason, detail):
        super().__init__('output_invalid', USER_MESSAGE, 502)
        self.args = (detail,)
        self.reason = reason


_KNOWN_FIELDS = frozenset(_Answer.model_fields) | frozenset(_Citation.model_fields)


def _known(part):
    return str(part) if isinstance(part, int) or part in _KNOWN_FIELDS else '<extra>'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputInvalid('duplicate_field', 'Duplicate JSON field')   # the key is model output
        result[key] = value
    return result


def _check(content, contexts):
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        if '\n' not in text:   # 한 줄 펜스는 예전에도 거부(IndexError)했다. 거부는 같고 분류만 json으로 한다.
            raise OutputInvalid('json', 'Code fence without a body line')
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except OutputInvalid:
        raise
    except RecursionError:
        raise OutputInvalid('json', 'JSON nested too deeply') from None
    except ValueError as exc:   # JSONDecodeError(위치만 담음)와 너무 긴 정수 변환 오류
        raise OutputInvalid('json', str(exc)) from None
    try:
        answer = _Answer.model_validate(data)
    except ValidationError as exc:
        # 위치와 오류 종류만 남긴다. pydantic 기본 문구에는 모델 출력 값(input_value)이 들어가고,
        # 추가 필드의 loc는 모델이 정한 키 이름이라 스키마에 있는 이름만 남긴다.
        detail = '; '.join(f"{'.'.join(_known(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors(include_input=False))
        raise OutputInvalid('schema', detail) from None
    if not answer.answer.strip():
        raise OutputInvalid('blank', 'Answer must not be blank')
    valid = {(c['book_id'], c['page'], c['chunk_id']) for c in contexts}
    cited = [(c.book_id, c.page, c.chunk_id) for c in answer.citations]
    if any(c not in valid for c in cited):
        raise OutputInvalid('citation_out_of_context', 'Citation does not exist in actual model context')
    if len(set(cited)) != len(cited):
        raise OutputInvalid('duplicate_citation', 'Duplicate citation')
    if answer.status == 'answered' and not answer.citations:
        raise OutputInvalid('missing_citation', 'Answered result requires an actual source citation')
    # 본문의 표준 출처 표기가 있으면 citations와도 일치해야 한다.
    inline = re.findall(r'\[([A-Za-z0-9_-]+)\s+p\.\s*(\d+)\]', answer.answer)
    pages = {(c.book_id, c.page) for c in answer.citations}
    if any((book, int(page)) not in pages for book, page in inline):
        raise OutputInvalid('inline_mismatch', 'Inline citation does not match declared citations')
    return answer, cited


def validate_answer(content: str, contexts):
    with context_stage('context_output_validation', metadata={'source_count':len(contexts)}) as span:
        try:
            answer, cited = _check(content, contexts)
        except OutputInvalid as exc:
            if span:
                span.update(output={'validation':'rejected', 'reason':exc.reason})
            raise
        if span:
            span.update(output={'status':answer.status, 'citation_count':len(cited),
                                'reason_present':bool(answer.reason.strip()),
                                'validation':'passed', 'semantic_correctness':'not_evaluated'})
        return ModelAnswer.model_validate(answer.model_dump())
