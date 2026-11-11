"""기존 요청 trace가 있을 때만 자식 단계를 기록한다. 단위 실행은 외부 전송 없음."""
from contextlib import contextmanager


@contextmanager
def context_stage(name, *, input=None, metadata=None):
    from opentelemetry import trace
    if not trace.get_current_span().is_recording():
        yield None
        return
    from ..observability import observation
    with observation(name, input=input, metadata=metadata) as span:
        yield span
