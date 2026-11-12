"""Review fix (thermos, quality M5): the score table is the single source for ranges and run metadata."""
import pytest

from backend.app.observability import record_score
from backend.app.observability.scores import SCORES
from backend.tests.observability.fake_langfuse import FakeLangfuse


@pytest.mark.parametrize('name,value,ok', [('recall_at_k', 1.2, False), ('recall_at_k', 1.0, True),
                                           ('faithfulness', -0.1, False), ('first_gold_rank', 0, False),
                                           ('first_gold_rank', 12, True), ('hit_at_k', 2, False)])
def test_numeric_ranges_and_boolean_values_are_enforced(monkeypatch, name, value, ok):
    fake = FakeLangfuse().install(monkeypatch)
    assert record_score('t1', name, value) is ok and len(fake.scores) == int(ok)


def test_only_retrieval_scores_carry_run_metadata():
    assert {s.name for s in SCORES if s.run_metadata} == {'hit_at_k', 'recall_at_k', 'retrieval_status',
                                                          'first_gold_rank', 'retrieval_failure_cause'}
