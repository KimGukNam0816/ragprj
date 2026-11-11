"""Mem0를 호출하지 않는 연결 테스트용 고정 응답. 실제 기억 저장소가 아니다."""
from ..contracts import MemoryResult
from .telemetry import context_stage


def search_dummy_memory(question: str, user_id: str) -> MemoryResult:
    # 선호의 모양만 재현한다. 질문 임베딩·기억 검색·저장·갱신은 하지 않는다.
    result = {'results': [
        {'id': 'dummy-language', 'user_id': user_id,
         'memory': '한국어로 간결하게 답변해 주세요.',
         'metadata': {'source': 'dummy', 'scope': 'expression_preference'}},
        {'id': 'dummy-format', 'user_id': user_id,
         'memory': '여러 비용을 비교할 때 표 형식을 선호합니다.',
         'metadata': {'source': 'dummy', 'scope': 'expression_preference'}},
    ]}
    with context_stage('memory_preference_fixture', metadata={'source':'dummy', 'user_id':user_id}) as span:
        if span:
            span.update(output=result)
    return result
