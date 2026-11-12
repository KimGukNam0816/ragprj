"""Adversarial review fix (interrogate): a run name the SDK would drop still groups its traces."""
import hashlib

from backend.app.schemas import QueryRequest
from backend.tests.integration.test_team_pipeline import offline_pipeline  # noqa: F401 (pytest fixture)
from backend.tests.observability.fake_langfuse import FakeLangfuse


def test_non_ascii_run_name_gets_a_stable_ascii_session(offline_pipeline, monkeypatch):
    fake = FakeLangfuse().install(monkeypatch)
    pipeline, s = offline_pipeline
    for _ in range(2):
        pipeline.run_query(s, QueryRequest(question='면제 조건은?', mode='hybrid'), run_id='비교실행1', run_kind='comparison')
    roots = fake.roots()
    expected = 'run-' + hashlib.sha256('비교실행1'.encode()).hexdigest()[:8]
    assert [r.session_id for r in roots] == [expected, expected]
    assert all(r.metadata['run_id'] == '비교실행1' for r in roots)
