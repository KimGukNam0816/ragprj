"""Review fix (thermos, security M2): a `filter` overrides query parameters, so every condition must be inside it."""
import json

from backend.app.observability import query
from backend.tests.observability.fake_langfuse import FakeLangfuse


def conditions(fake):
    return {c['column']: c for c in json.loads(fake.api.observations.calls[-1]['filter'])}


def test_session_name_root_and_tags_are_all_in_the_filter(monkeypatch):
    fake = FakeLangfuse(observations=[]).install(monkeypatch)
    query.list_traces(run_id='run-1', tags=['comparison'])
    found = conditions(fake)
    assert found['sessionId']['value'] == 'run-1' and found['name']['value'] == 'shipping_rag_query'
    assert found['isRootObservation']['value'] is True and found['tags']['value'] == ['comparison']


def test_where_conditions_are_added_not_replacing(monkeypatch):
    fake = FakeLangfuse(observations=[]).install(monkeypatch)
    where = [{'type': 'stringObject', 'column': 'metadata', 'key': 'context_source', 'operator': '=', 'value': 'fixed'}]
    query.list_traces(where=where)
    found = conditions(fake)
    assert found['metadata']['key'] == 'context_source' and 'isRootObservation' in found and 'sessionId' not in found
