import json
from contextlib import contextmanager

import pytest

from backend.app.config import Settings
from backend.app.context_engineering.diagnostics.runner import diagnose_case, load_questions, run_diagnostics
from backend.app.context_engineering.validation import validate_answer


@pytest.mark.parametrize('case', load_questions(), ids=lambda c:c['id'])
def test_synthetic_context_cases_without_network(case, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, 'post', lambda *a, **k: pytest.fail('No network in offline diagnostics'))
    monkeypatch.setattr(httpx, 'get', lambda *a, **k: pytest.fail('No network in offline diagnostics'))
    s = Settings(_env_file=None)
    result = diagnose_case(s, case)
    assert result['checks'] and all(result['checks'].values()), result
    assert result['semantic_review_required'] is True


def test_html_report_escapes_untrusted_model_text_and_keeps_artifacts(tmp_path, monkeypatch):
    from backend.app.context_engineering.diagnostics import runner
    case = load_questions()[0]
    case['question'] = '<script>alert(1)</script>'
    monkeypatch.setattr(runner, 'load_questions', lambda:[case])
    folder, summary = run_diagnostics(Settings(_env_file=None, runtime_dir=tmp_path), run_name='offline-report')
    text = (folder/'report.html').read_text(encoding='utf-8')
    assert '<script>alert(1)</script>' not in text
    assert '&lt;script&gt;' in text
    assert (folder/'results.jsonl').exists()
    assert summary['cloud_verified'] is False
    assert summary['structural_passes'] == 1
    with pytest.raises(ValueError, match='already exists'):
        run_diagnostics(Settings(_env_file=None, runtime_dir=tmp_path), run_name='offline-report')


def test_model_diagnostics_require_explicit_limit(tmp_path):
    with pytest.raises(ValueError, match='explicit limit'):
        run_diagnostics(Settings(_env_file=None, runtime_dir=tmp_path), mode='model')


def test_missing_trace_configuration_is_not_reported_as_success(tmp_path, monkeypatch):
    from backend.app.context_engineering.diagnostics import runner
    monkeypatch.setattr(runner, 'langfuse_client', lambda:None)
    with pytest.raises(ValueError, match='not configured'):
        run_diagnostics(Settings(_env_file=None, runtime_dir=tmp_path), trace=True)


def test_model_output_is_separate_from_sample_validation(monkeypatch):
    from backend.app.context_engineering.diagnostics import runner
    class Provider:
        def __init__(self, s): pass
        def chat(self, messages, schema):
            # 기대값·샘플 출력은 생성 입력에 전달하지 않는다.
            assert 'expected_validation' not in json.dumps(messages)
            return {'content':json.dumps(load_questions()[0]['sample_output']), 'input_tokens':10,
                    'output_tokens':5, 'attempts':[], 'model':'test'}
    monkeypatch.setattr(runner, 'Providers', Provider)
    monkeypatch.setattr('backend.app.observability.observation', lambda *a, **k: __import__('contextlib').nullcontext(None))
    result = diagnose_case(Settings(_env_file=None), load_questions()[0], mode='model', tokenizer=runner.FixtureTokenizer())
    assert result['sample_validation']['status'] == 'passed'
    assert result['model_result']['validation'] == 'passed'
    assert result['semantic_review_required']


@pytest.mark.parametrize('body', [
    '{"answer":"x","answer":"y","status":"abstained"}',
    '{"answer":"x","status":"abstained","extra":1}',
    '{"answer":"x","status":"answered","citations":[{"book_id":"A","page":true,"chunk_id":"c"}]}',
])
def test_strict_output_rules(body):
    with pytest.raises(ValueError):
        validate_answer(body, [{'book_id':'A','page':1,'chunk_id':'c'}])


def test_existing_trace_gets_context_stage_records(monkeypatch):
    from backend.app.context_engineering import builder
    from backend.app.context_engineering.diagnostics.runner import FixtureTokenizer
    events = []
    @contextmanager
    def stage(name, **kwargs):
        class Span:
            def update(self, **data): events.append((name, data))
        yield Span()
    monkeypatch.setattr(builder, 'context_stage', stage)
    builder.prepare_context(Settings(_env_file=None), '질문', load_questions()[0]['chunks'], tokenizer=FixtureTokenizer())
    assert {name for name, _ in events} == {'context_deduplication','context_budget','context_prompt'}
