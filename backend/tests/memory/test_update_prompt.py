from types import SimpleNamespace

from backend.app.config import Settings
from backend.app.memory import mem0_client
from backend.app.memory.update_prompt import PREFERENCE_UPDATE_PROMPT


def test_mem0_update_prompt_is_wired_without_skipping_inference(monkeypatch, tmp_path):
    import mem0
    from backend.app import providers
    from mem0.configs.prompts import get_update_memory_messages

    captured = {}
    def from_config(config):
        captured.update(config)
        return SimpleNamespace()
    monkeypatch.setattr(mem0.Memory, 'from_config', from_config)
    monkeypatch.setattr(mem0_client, 'get_settings', lambda: Settings(
        _env_file=None, mem0_history_db_path=tmp_path / 'history.db'))
    calls = []
    class Provider:
        def __init__(self, settings):
            pass
        def chat(self, messages, schema):
            calls.append((messages, schema))
            return {'content': '{"memory":[]}', 'input_tokens': 1, 'output_tokens': 1}
    monkeypatch.setattr(providers, 'Providers', Provider)
    mem0_client.memory_client.cache_clear()
    try:
        client = mem0_client.memory_client()
        prompt = get_update_memory_messages(
            [{'id': '0', 'text': '한국어로 짧게 답변해 주세요'}],
            ['여러 항목을 비교할 때는 표로 정리해 주세요'],
            captured['custom_update_memory_prompt'])
        assert captured['custom_update_memory_prompt'] == PREFERENCE_UPDATE_PROMPT
        assert 'custom_fact_extraction_prompt' in captured
        assert '한국어로 짧게 답변해 주세요' in prompt
        assert '여러 항목을 비교할 때는 표로 정리해 주세요' in prompt
        client.llm.generate_response([{'role': 'user', 'content': prompt}])
        schema = calls[0][1]
        assert 'old_memory' in schema['$defs']['Action']['properties']
        assert calls[0][0][0]['content'] == prompt
    finally:
        mem0_client.memory_client.cache_clear()
