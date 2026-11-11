"""생성 직후 출력 검증 (답변 단계의 안전장치, 채점 아님).

모델 출력이 정해진 JSON 형식인지, 인용한 출처가 실제로 넣어 준 문맥에 있는지 코드로 확인한다.
없는 출처를 지어내면 답변을 거부하므로, 채점(Judge 근거 항목)은 "실재하는 인용이 주장을 뒷받침하나"만 보면 된다.
"""
from ..schemas import ModelAnswer


def validate_answer(content: str, contexts):
    # 1. 코드 블록(```)으로 감싼 출력이면 벗긴다
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]

    # 2. 형식 검사: answer · status(answered/needs_clarification/abstained) · reason · citations
    answer = ModelAnswer.model_validate_json(text)

    # 3. 인용 검사: (문서 ID, 쪽, 청크 ID)가 실제 최종 문맥에 있어야 함 → 가짜 출처 차단
    valid = {(c['book_id'], c['page'], c['chunk_id']) for c in contexts}
    if any((c.book_id, c.page, c.chunk_id) not in valid for c in answer.citations):
        raise ValueError('Citation does not exist in actual model context')

    # 4. "답했다"면 출처가 최소 1개 있어야 함 (발제 필수 과제 1: 문서명·페이지 표시)
    if answer.status == 'answered' and not answer.citations:
        raise ValueError('Answered result requires an actual source citation')
    return answer


