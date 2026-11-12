import copy

import pytest

from backend.app.observability import SCORE_NAMES, flush_scores, publish_scores, record_score
from backend.app.observability import scores
from backend.tests.observability.fake_langfuse import FakeLangfuse

JUDGE_OK = {k: {'score': 1, 'reason': f'{k} ok'} for k in ('facts', 'conditions', 'evidence', 'decision')}


def row(**fields):
    base = {'trace_id': 't1', 'evaluator_sha256': 'e1', 'mode': 'hybrid', 'profile': 'bge',
            'retrieval_status': 'ok', 'hit_at_4': 1, 'recall_at_4': 0.5, 'gold_doc_count': 2, 'first_gold_rank': 3,
            'retrieval_failure_cause': 'hit', 'faithfulness': 0.8, 'faithfulness_status': 'ok',
            'judge': JUDGE_OK, 'judge_status': 'ok'}
    return {**base, **fields}


def sent(fake):
    return {s['name']: s for s in fake.scores}


def test_ok_row_sends_numbers_and_no_status_scores(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    assert publish_scores(row(), run_id='run-1') == 12
    by_name = sent(fake)
    assert set(by_name) == {'hit_at_k', 'recall_at_k', 'first_gold_rank', 'retrieval_failure_cause', 'faithfulness',
                            'judge_facts', 'judge_conditions', 'judge_evidence', 'judge_decision',
                            'retrieval_status', 'faithfulness_status', 'judge_status'}   # status 'ok' overwrites a stale one
    assert by_name['recall_at_k']['metadata'] == {'gold_doc_count': 2, 'mode': 'hybrid', 'profile': 'bge', 'run_id': 'run-1'}
    assert by_name['judge_facts']['comment'] == 'facts ok' and by_name['judge_facts']['data_type'] == 'BOOLEAN'
    assert by_name['faithfulness']['metadata'] is None


def test_not_applicable_and_unscorable_send_only_status_categories(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    publish_scores(row(retrieval_status='not_applicable', hit_at_4=None, recall_at_4=None, first_gold_rank=None,
                       retrieval_failure_cause='no_reference', faithfulness=None, faithfulness_status='unscorable'),
                   run_id='run-1')
    by_name = sent(fake)
    assert by_name['retrieval_status']['value'] == 'not_applicable'
    assert by_name['faithfulness_status']['value'] == 'unscorable'
    assert by_name['retrieval_failure_cause']['value'] == 'no_reference'
    assert not {'hit_at_k', 'recall_at_k', 'first_gold_rank', 'faithfulness'} & set(by_name)


@pytest.mark.parametrize('status', ['service_error', 'output_invalid'])
def test_failed_service_scores_decision_zero_with_status(monkeypatch, status):
    fake = FakeLangfuse().install(monkeypatch)
    judge = {k: {'score': None, 'reason': status} for k in ('facts', 'conditions', 'evidence')}
    publish_scores(row(retrieval_status='unscorable', judge={**judge, 'decision': {'score': 0, 'reason': status}},
                       judge_status=status, faithfulness=None, faithfulness_status='unscorable'), run_id='run-1')
    by_name = sent(fake)
    assert by_name['judge_decision']['value'] == 0 and by_name['judge_status']['value'] == status
    assert 'judge_facts' not in by_name and 'retrieval_failure_cause' not in by_name


def test_resend_uses_same_score_ids_and_does_not_change_the_row(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    original = row()
    frozen = copy.deepcopy(original)
    publish_scores(original, run_id='run-1')
    publish_scores(original, run_id='run-1')
    ids = [s['score_id'] for s in fake.scores]
    assert ids[:len(ids) // 2] == ids[len(ids) // 2:] and len(set(ids)) == len(ids) // 2
    assert original == frozen
    other_evaluator = FakeLangfuse().install(monkeypatch)
    publish_scores(row(evaluator_sha256='e2'), run_id='run-1')
    assert {s['score_id'] for s in other_evaluator.scores}.isdisjoint(ids)


def test_no_trace_id_or_langfuse_off_sends_nothing(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    assert publish_scores(row(trace_id=None), run_id='run-x') == 0 and fake.scores == []
    monkeypatch.setattr(scores.tracing, 'langfuse_client', lambda: None)
    assert publish_scores(row(), run_id='run-x') == 0
    assert record_score('t1', 'hit_at_k', 1) is False


def test_record_score_checks_name_and_type(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    assert record_score('t1', 'made_up', 1) is False
    assert record_score('t1', 'judge_status', 1) is False            # number under a CATEGORICAL name
    assert record_score('t1', 'hit_at_k', 0.4) is False              # BOOLEAN must be 0/1
    assert record_score('t1', 'security_case_status', 'defended', comment='c') is True
    assert fake.scores[-1]['data_type'] == 'CATEGORICAL' and 'made_up' not in SCORE_NAMES


def test_security_record_maps_to_security_scores(monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    judgment = {'exposed': True, 'attack_followed': False, 'business_correct': True, 'unnecessary_refusal': False}
    publish_scores({'trace_id': 't2', 'status': 'ok', 'judgment': judgment, 'judgment_status': 'ok',
                    'defense_success': True}, run_id='sec-1')
    by_name = sent(fake)
    assert by_name['security_case_status']['value'] == 'defended' and by_name['security_case_pass']['value'] == 1
    assert by_name['security_exposed']['value'] == 1 and 'judge_status' not in by_name
    fake.scores.clear()
    publish_scores({'trace_id': 't3', 'status': 'error', 'judgment': None, 'judgment_status': 'unscorable',
                    'defense_success': False}, run_id='sec-1')
    assert [(s['name'], s['value']) for s in fake.scores] == [('security_case_status', 'generation_failed')]


def test_send_and_flush_failures_only_warn(monkeypatch, caplog):
    fake = FakeLangfuse(flush_error=RuntimeError('down')).install(monkeypatch)

    def broken(**score):
        raise RuntimeError('queue full')

    monkeypatch.setattr(fake, 'create_score', broken)
    assert publish_scores(row(), run_id='run-1') == 0
    flush_scores()
    failures = {(r.getMessage(), r.fields.get('error_type')) for r in caplog.records}
    assert {('score_send_failed', 'RuntimeError'), ('score_flush_failed', 'RuntimeError')} <= failures
