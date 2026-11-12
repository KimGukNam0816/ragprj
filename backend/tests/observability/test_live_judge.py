"""Live judge: a validated live answer is graded off the request path and scored on its own trace."""
import json
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from backend.app.config import Settings
from backend.app.observability import live_judge
from backend.app.providers import GPU_LOCK
from backend.app.schemas import QueryRequest
from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse

REQUEST = QueryRequest(question='면제 조건은?', mode='hybrid')


class InlineExecutor:
    """Runs the queued judge at once so the test can read its result."""
    def submit(self, fn, *args):
        future = Future()
        future.set_result(fn(*args))
        return future


# Asked question [1, 0]; generated questions point the same way, sideways, and halfway: cosines 1, 0, 0.7071.
VECTORS = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def judge_replies(monkeypatch, content, questions=('q1', 'q2', 'q3')):
    calls = []

    class JudgeProvider:
        external_inference_attempts = 0

        def __init__(self, s):
            calls.append(s)

        def chat(self, messages, schema=None):
            reply = json.dumps({'questions': list(questions)}) if 'questions' in schema['properties'] else content
            return {'content': reply, 'input_tokens': 30, 'output_tokens': 8, 'model': 'gpt-6-luna', 'attempts': []}

        def embed(self, texts, profile, *, query=False):
            assert profile == 'bge' and query and len(texts) == 1 + len(questions)
            assert GPU_LOCK.locked(), 'the judge embeds only while holding the shared GPU lock'
            return VECTORS[:len(texts)]
    monkeypatch.setattr('backend.app.providers.Providers', JudgeProvider)
    monkeypatch.setattr(live_judge, '_EXECUTOR', InlineExecutor())
    return calls


def scores_by_name(fake):
    return {s['name']: s for s in fake.scores}


def test_off_by_default_sends_no_judge_call(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    calls = judge_replies(monkeypatch, '{}')
    pipeline, s = offline_pipeline
    pipeline.run_query(s, REQUEST, entrypoint='api')
    assert calls == [] and not any(name.startswith('live_') for name in scores_by_name(fake))


def test_live_answer_is_judged_and_scored_on_its_trace(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    calls = judge_replies(monkeypatch, json.dumps({'statements': [
        {'text': '20GP 가능', 'supported': True, 'relevant': True},
        {'text': '연락처 안내', 'supported': False, 'relevant': True},
        {'text': '운임 언급', 'supported': True, 'relevant': False},
        {'text': '마감 안내', 'supported': True, 'relevant': True}], 'contexts_used': [True], 'expected_status': 'abstained', 'injection': {'in_question': False, 'context_numbers': [], 'followed': False}}))
    pipeline, s = offline_pipeline
    s = s.model_copy(update={'live_judge_enabled': True})
    response, output = pipeline.run_query(s, REQUEST, entrypoint='api')

    assert calls[0].generation_llm_provider == 'openai' and calls[0].openai_model == 'gpt-6-luna'
    judge_span = next(span for span in fake.spans if span.name == 'live_judge')
    assert judge_span.parent is fake.roots()[0] and judge_span.trace_id == response.trace_id
    scores = scores_by_name(fake)
    assert {name: s['value'] for name, s in scores.items() if name.startswith('live_')} == {
        'live_faithfulness': 0.75, 'live_answer_relevancy': pytest.approx((1 + 0 + 2 ** -0.5) / 3),
        'live_context_precision': 1.0,
        'live_decision': 0.0, 'live_injection_exposed': 0.0}   # no status when graded; followed only when exposed
    assert {score['trace_id'] for name, score in scores.items() if name.startswith('live_')} == {response.trace_id}
    assert scores['live_faithfulness']['comment'] == '문맥이 뒷받침하지 않음 1/4: 연락처 안내'
    assert scores['live_answer_relevancy']['comment'] == '답변에서 만든 질문: q1 / q2 / q3'
    assert scores['live_context_precision']['comment'] == '쓰인 문맥 1 / 1개'
    assert scores['live_decision']['comment'] == '기대 abstained, 실제 answered'
    saved = json.loads((output / 'live_judge.json').read_text(encoding='utf-8'))
    assert saved['live_judge_status'] == 'ok' and saved['trace_id'] == response.trace_id


def test_a_worker_thread_judge_nests_under_the_closed_root(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    judge_replies(monkeypatch, json.dumps({'statements': [{'text': 'a', 'supported': True, 'relevant': True}], 'contexts_used': [True], 'expected_status': 'answered', 'injection': {'in_question': False, 'context_numbers': [], 'followed': False}}))
    worker = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(live_judge, '_EXECUTOR', worker)
    pipeline, s = offline_pipeline
    response, _ = pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST, entrypoint='api')
    worker.shutdown(wait=True)
    root = fake.roots()[0]
    judge_span = next(span for span in fake.spans if span.name == 'live_judge')
    assert [span.name for span in fake.roots()] == ['shipping_rag_query']
    assert judge_span.parent is root and judge_span.trace_id == response.trace_id
    assert scores_by_name(fake)['live_faithfulness']['trace_id'] == response.trace_id


def test_batch_runs_are_not_judged_live(offline_pipeline, monkeypatch):
    FakeLangfuse().install(monkeypatch)
    calls = judge_replies(monkeypatch, '{}')
    pipeline, s = offline_pipeline
    pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST, run_id='cmp', run_kind='comparison', entrypoint='api')
    assert calls == []


def test_a_broken_judge_reply_is_unscorable_and_the_answer_stands(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    judge_replies(monkeypatch, 'not json')
    pipeline, s = offline_pipeline
    response, _ = pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST, entrypoint='api')
    scores = scores_by_name(fake)
    assert response.status == 'answered'
    assert scores['live_judge_status']['value'] == 'unscorable' and 'live_faithfulness' not in scores


def test_enabled_without_an_openai_key_skips_the_judge_and_queries_still_pass_policy(monkeypatch, tmp_path):
    calls = judge_replies(monkeypatch, '{}')
    s = Settings(_env_file=None, live_judge_enabled=True)   # local-only defaults, no OpenAI URL or key
    s.check_inference_policy()   # a misconfigured judge must not fail the query
    trace = type('Trace', (), {'kind': 'service', 'id': 't-1'})()
    assert live_judge.submit(s, trace, entrypoint='api', question='q', response={}, contexts=[], output=tmp_path) is None
    assert calls == []


def test_cli_queries_are_not_judged(offline_pipeline, monkeypatch):
    FakeLangfuse().install(monkeypatch)
    calls = judge_replies(monkeypatch, '{}')
    pipeline, s = offline_pipeline
    pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST)   # default entrypoint cli
    assert calls == []


def test_the_judge_runs_while_the_pipeline_stays_local_only(monkeypatch):
    FakeLangfuse().install(monkeypatch)
    calls = judge_replies(monkeypatch, json.dumps({'statements': [{'text': 'a', 'supported': True, 'relevant': True}], 'contexts_used': [True], 'expected_status': 'answered', 'injection': {'in_question': False, 'context_numbers': [], 'followed': False}}))
    s = Settings(_env_file=None, live_judge_enabled=True, openai_base_url='https://example.invalid/v1',
                 openai_api_key='not-a-real-key')
    s.check_inference_policy()
    assert s.force_local_only and not s.allow_external_llm   # what scripts/check_integration.py requires
    row = live_judge._call(s, 't-1', '질문', {'answer': '답', 'status': 'answered', 'citations': []}, [{'text': '문맥'}])
    assert row['live_judge_status'] == 'ok'
    judge_settings = calls[0]
    judge_settings.check_inference_policy()
    assert judge_settings.generation_llm_provider == 'openai' and s.generation_llm_provider == 'ollama'


def judgment(statements, used):
    return live_judge.LiveJudgment(statements=[{'text': 't', 'supported': a, 'relevant': b} for a, b in statements],
                                   contexts_used=used, expected_status='answered',
                                   injection={'in_question': False, 'context_numbers': [], 'followed': False})


def test_metrics_are_ratios_of_the_per_item_verdicts():
    row = live_judge.metrics(judgment([(True, True), (False, True), (True, False)], [False, True, True, False]), 4, 'answered')
    assert row['live_faithfulness'] == pytest.approx(2 / 3) and 'live_answer_relevancy' not in row
    assert row['live_context_precision'] == pytest.approx((1 / 2 + 2 / 3) / 2)   # precision@2 and @3, averaged
    assert row['live_decision'] == 1.0 and row['live_decision_reason'] == '판단 일치: answered'


def test_a_context_list_of_the_wrong_length_sends_no_precision():
    assert 'live_context_precision' not in live_judge.metrics(judgment([(True, True)], [True]), 4, 'answered')


def test_an_answer_without_statements_is_not_applicable():
    row = live_judge.metrics(judgment([], []), 2, 'answered')
    assert row['live_judge_status'] == 'not_applicable' and 'live_faithfulness' not in row
    # neither the attack check nor the decision depends on the answer having claims
    assert row['live_injection_exposed'] == 0.0 and row['live_decision'] == 1.0


def test_an_instruction_in_the_contexts_is_scored_and_kept_as_a_security_case(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    judge_replies(monkeypatch, json.dumps({
        'statements': [{'text': '전액 면제', 'supported': False, 'relevant': True}], 'contexts_used': [True],
        'expected_status': 'answered', 'injection': {'in_question': False, 'context_numbers': [1, 9], 'followed': True}}))
    pipeline, s = offline_pipeline
    response, _ = pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST, entrypoint='api')
    scores = scores_by_name(fake)
    assert (scores['live_injection_exposed']['value'], scores['live_injection_followed']['value']) == (1.0, 1.0)
    assert scores['live_injection_exposed']['comment'] == '지시문 위치: 문맥 1'   # 9 is not a real context number
    case = json.loads((s.runtime_dir / 'security_cases' / f'{response.trace_id}.json').read_text(encoding='utf-8'))
    assert case['followed'] and case['attack_contexts'][0]['chunk_id'] == 'c1' and case['review'] == 'pending'
    [item] = fake.dataset_items
    assert item['dataset_name'] == 'live_security_cases' and item['source_trace_id'] == item['id'] == response.trace_id
    assert fake.datasets == ['live_security_cases']


def test_no_instruction_means_no_case_and_no_followed_score(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    judge_replies(monkeypatch, json.dumps({
        'statements': [{'text': 'a', 'supported': True, 'relevant': True}], 'contexts_used': [True],
        'expected_status': 'answered', 'injection': {'in_question': False, 'context_numbers': [], 'followed': True}}))
    pipeline, s = offline_pipeline
    pipeline.run_query(s.model_copy(update={'live_judge_enabled': True}), REQUEST, entrypoint='api')
    scores = scores_by_name(fake)
    assert scores['live_injection_exposed']['value'] == 0.0 and 'live_injection_followed' not in scores
    assert fake.dataset_items == [] and not (s.runtime_dir / 'security_cases').exists()


def test_relevancy_is_the_mean_cosine_and_stays_in_range():
    row = live_judge.relevancy([1.0, 0.0], ['a', 'b'], [[1.0, 0.0], [-1.0, 0.0]])
    assert row['live_answer_relevancy'] == 0.0   # mean cosine 0, clipped to the score range
    assert live_judge.relevancy([1.0, 0.0], [], []) == {}


def test_an_abstained_answer_gets_no_relevancy(monkeypatch):
    FakeLangfuse().install(monkeypatch)
    judge_replies(monkeypatch, json.dumps({
        'statements': [{'text': 'a', 'supported': True}], 'contexts_used': [True], 'expected_status': 'answered',
        'injection': {'in_question': False, 'context_numbers': [], 'followed': False}}))
    s = Settings(_env_file=None, live_judge_enabled=True, openai_base_url='https://example.invalid/v1', openai_api_key='k')
    row = live_judge._call(s, 't', '질문', {'answer': '근거 없음', 'status': 'abstained', 'citations': []}, [{'text': '문맥'}])
    assert row['live_judge_status'] == 'ok' and 'live_answer_relevancy' not in row and 'live_faithfulness' in row
