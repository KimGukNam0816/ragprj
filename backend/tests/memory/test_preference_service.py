from contextlib import nullcontext

import pytest

from backend.app.config import Settings
from backend.app.schemas import ServiceError


def test_preference_service_keeps_user_scope_and_existing_response(monkeypatch):
    from backend.app.memory import service
    calls = []
    result = {'results': [{'memory': '한국어로 짧게 답변'}]}
    class Memory:
        def search(self, question, **kwargs):
            calls.append(kwargs)
            return result
        def add(self, messages, **kwargs):
            calls.append(kwargs)
            return result
    monkeypatch.setattr(service, 'get_settings', lambda: Settings(_env_file=None, memory_top_k=2))
    monkeypatch.setattr(service, 'memory_client', lambda: Memory())
    monkeypatch.setattr(service, 'observation', lambda *a, **k: nullcontext(None))
    assert service.search_memory('면제 조건?', 'user-a') is result
    assert calls[0] == {'user_id': 'user-a', 'limit': 2}
    assert service.add_memory('짧게 답변해 주세요.', 'user-a') is result
    assert calls[1]['user_id'] == 'user-a'
    assert calls[1]['metadata']['explicit'] is True


def test_dummy_backend_refuses_writes_instead_of_reaching_mem0(monkeypatch):
    from backend.app.memory import service
    monkeypatch.setattr(service, 'get_settings', lambda: Settings(_env_file=None, memory_backend='dummy'))
    monkeypatch.setattr(service, 'memory_client', lambda: pytest.fail('Mem0 must not be called'))
    with pytest.raises(ServiceError) as caught:
        service.add_memory('짧게 답변해 주세요.', 'user-a')
    assert caught.value.code == 'memory_backend_readonly'
    assert caught.value.http_status == 422
