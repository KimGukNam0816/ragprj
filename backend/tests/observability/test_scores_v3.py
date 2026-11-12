"""Adversarial review fixes (interrogate): status overwrite after retry, failure cause only for answered items,
a dedicated probe score name, and score-config category updates."""
from types import SimpleNamespace

from backend.app.observability import publish_scores, setup
from backend.app.observability.scores import SCORES
from backend.tests.observability.fake_langfuse import FakeLangfuse


def latest(fake):
    """What Langfuse keeps: the last write per score_id."""
    return {(s['trace_id'], s['name']): s['value'] for s in {s['score_id']: s for s in fake.scores}.values()}


def test_a_successful_rejudge_overwrites_the_earlier_unscorable_status(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    base = {'trace_id': 't1', 'evaluator_sha256': 'e', 'faithfulness_status': 'not_applicable'}
    publish_scores({**base, 'judge': None, 'judge_status': 'unscorable'}, run_id='r')
    judge = {k: {'score': 1, 'reason': 'ok'} for k in ('facts', 'conditions', 'evidence', 'decision')}
    publish_scores({**base, 'judge': judge, 'judge_status': 'ok'}, run_id='r')
    assert latest(fake)[('t1', 'judge_status')] == 'ok' and latest(fake)[('t1', 'judge_decision')] == 1


def test_failure_cause_is_not_sent_for_failed_requests(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    row = {'trace_id': 't2', 'retrieval_status': 'ok', 'hit_at_4': 0, 'recall_at_4': 0.0, 'first_gold_rank': 2,
           'retrieval_failure_cause': 'context_dropped', 'judge_status': 'output_invalid',
           'judge': {'decision': {'score': 0, 'reason': 'output_invalid'}}}
    publish_scores(row, run_id='r')
    names = {name for _, name in latest(fake)}
    assert 'retrieval_failure_cause' not in names and {'hit_at_k', 'first_gold_rank', 'judge_status'} <= names
    publish_scores({**row, 'trace_id': 't3', 'judge_status': 'ok'}, run_id='r')
    assert latest(fake)[('t3', 'retrieval_failure_cause')] == 'context_dropped'


def test_the_probe_score_is_never_published_from_a_row(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    publish_scores({'trace_id': 't', 'observability_check': 1, 'output_valid': None}, run_id='r')
    assert all(s['name'] != 'observability_check' for s in fake.scores)
    from backend.app.observability import cloud_check
    assert cloud_check.CHECK_SCORE == 'observability_check' and 'observability_check' in {s.name for s in SCORES}


def test_setup_updates_a_config_whose_categories_grew(monkeypatch):
    fake = FakeLangfuse(prompts={'rag_answer': SimpleNamespace(is_fallback=False)}).install(monkeypatch)
    old = SimpleNamespace(name='judge_status', id='cfg-1',
                          categories=[SimpleNamespace(label=c) for c in ('unscorable', 'service_error', 'output_invalid')])
    others = [SimpleNamespace(name=s.name, id=s.name, categories=None) for s in SCORES if s.name != 'judge_status']
    updated = []
    fake.api.score_configs = SimpleNamespace(
        get=lambda page=1, limit=100: SimpleNamespace(data=[old, *others], meta=SimpleNamespace(total_pages=1)),
        update=lambda config_id, **fields: updated.append((config_id, [c.label for c in fields['categories']])),
        create=lambda **c: None)
    assert setup.plan(fake) == [('update_score_config', 'judge_status')]
    setup.apply(fake, setup.plan(fake))
    assert updated == [('cfg-1', ['unscorable', 'service_error', 'output_invalid', 'ok'])]
