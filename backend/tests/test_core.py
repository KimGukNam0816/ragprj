import json

import pytest

from backend.app.config import Settings
from backend.app.indexing import keyword_tokens, rrf
from backend.app.pipeline import validate_answer
from backend.app.providers import Providers


def test_local_policy_blocks_external_before_request(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('External HTTP request attempted')
    monkeypatch.setattr('backend.app.providers.httpx.post', forbidden)
    s = Settings(_env_file=None, execution_stage='local_smoke', generation_llm_provider='openai', allow_external_llm=True)
    with pytest.raises(ValueError, match='disabled'):
        Providers(s).chat([{'role': 'user', 'content': 'hello'}])


def test_local_policy_checks_memory_and_evaluator():
    s = Settings(_env_file=None, execution_stage='local_smoke', evaluation_llm_provider='openai')
    with pytest.raises(ValueError, match='disabled'):
        s.check_inference_policy()


@pytest.mark.parametrize('field', ['generation_llm_provider', 'evaluation_llm_provider', 'mem0_llm_provider'])
def test_external_inference_cannot_be_enabled_by_legacy_flags(field):
    s = Settings(_env_file=None, execution_stage='api_final60',
                 force_local_only=False, allow_external_llm=True, **{field: 'openai'})
    with pytest.raises(ValueError, match='disabled'):
        s.check_inference_policy()


def test_rrf_shared_hit_ranks_first_and_duplicate_not_counted_twice():
    a, b, c = [{'chunk_id': x} for x in ['a', 'b', 'c']]
    rows = rrf([[a, b, a], [c, b]])
    assert rows[0]['chunk_id'] == 'b'
    assert len(rows) == 3
    assert next(r['score'] for r in rows if r['chunk_id'] == 'a') == 1 / 61


def test_citation_must_match_exact_context_page_and_chunk():
    source = {'book_id': 'AS007', 'page': 1, 'chunk_id': 'actual'}
    obj = {'answer': '답변', 'status': 'answered', 'citations': [dict(source)]}
    assert validate_answer(json.dumps(obj), [source]).status == 'answered'
    obj['citations'][0]['page'] = 2
    with pytest.raises(ValueError, match='Citation'):
        validate_answer(json.dumps(obj), [source])


def test_answered_requires_citation_but_clarification_does_not():
    with pytest.raises(ValueError, match='requires'):
        validate_answer('{"answer":"내용", "status":"answered"}', [])
    assert validate_answer('{"answer":"어떤 예약인가요?", "status":"needs_clarification"}', []).citations == []


def test_keyword_codes_are_preserved():
    tokens = keyword_tokens('승인번호 APR-DL28-0510 보관료')
    assert 'apr-dl28-0510' in tokens
    assert '보관료' in tokens and '보관' in tokens


def test_index_resumes_failed_batch_without_duplicates(tmp_path, monkeypatch):
    from backend.app.artifacts import write_json, write_jsonl
    from backend.app.indexing import build_index
    corpus = tmp_path / 'corpus'
    write_json(corpus / 'manifest.json', {'failure_count': 0, 'corpus_hash': 'source', 'scope': 'full'})
    write_jsonl(corpus / 'chunks.jsonl', [{'chunk_id': str(i), 'book_id': 'AS007', 'page': 1, 'text': '문서'} for i in range(5)])
    class Tags:
        def raise_for_status(self): pass
        def json(self):
            return {'models': [{'name': 'bge-m3:latest', 'digest': 'frozen-model'}]}
    monkeypatch.setattr('backend.app.indexing.httpx.get', lambda *a, **kw: Tags())
    points = {}
    class Store:
        def __init__(self, s): pass
        def ensure(self, *a): pass
        def count(self, *a): return len(points)
        def upsert(self, collection, batch):
            points.update({p['id']: p for p in batch})
        def close(self): pass
    monkeypatch.setattr('backend.app.indexing.QdrantStore', Store)
    calls = []
    def embed(self, texts, profile):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError('simulated interrupted batch')
        return [[.1]*1024 for _ in texts]
    monkeypatch.setattr('backend.app.indexing.Providers.embed', embed)
    s = Settings(_env_file=None, runtime_dir=tmp_path/'runtime', embedding_batch_size=2)
    with pytest.raises(RuntimeError):
        build_index(s, corpus, 'bge')
    assert len(points) == 2
    index = build_index(s, corpus, 'bge')
    assert len(points) == 5
    assert json.loads((index/'manifest.json').read_text())['complete']
    count = len(calls)
    build_index(s, corpus, 'bge')
    assert len(points) == 5 and len(calls) == count


def test_formal_metrics_never_allowed_in_smoke():
    with pytest.raises(ValueError,match='Smoke'):
        Settings(_env_file=None,publish_metrics=True,smoke_run=True).check_inference_policy()


def test_retrieval_metric_matches_project_document_definition():
    from evaluation.quality import retrieval_scores
    ref={'evidence_groups':[[{'book_id':'A'},{'book_id':'B'}],[{'book_id':'A'}]]}
    score=retrieval_scores(ref,[{'book_id':'A'},{'book_id':'A'}])
    assert score['hit_at_4']==1 and score['recall_at_4']==.5
    assert retrieval_scores(ref,[])['recall_at_4']==0
    assert retrieval_scores({'evidence_groups':[]},[])['retrieval_status']=='not_applicable'


def test_ollama_down_returns_actionable_message(monkeypatch):
    """Ollama가 꺼져 있을 때 ConnectError가 아니라 조치 가능한 한국어 메시지를 돌려준다."""
    import httpx
    from backend.app.config import Settings
    from backend.app.providers import ollama_models
    from backend.app.schemas import ServiceError

    def refuse(*args, **kwargs):
        raise httpx.ConnectError('connection refused')

    monkeypatch.setattr('backend.app.providers.httpx.get', refuse)
    with pytest.raises(ServiceError) as caught:
        ollama_models(Settings(_env_file=None))
    assert caught.value.code == 'ollama_unavailable'
    assert 'Ollama' in caught.value.message and '11434' in caught.value.message
