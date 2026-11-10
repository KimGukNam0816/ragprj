"""노영우: 평가 점수 기록용 진입점. 점수 계산과 전송을 분리한다."""
from .tracing import langfuse_client


def record_score(trace_id: str, name: str, value: float, *, comment: str | None = None):
    client = langfuse_client()
    if client is None:
        return False
    client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    return True
