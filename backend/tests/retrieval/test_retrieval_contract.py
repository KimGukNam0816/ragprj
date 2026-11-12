from backend.app.contracts import RetrievalResult
import pytest


def test_old_indexing_and_reranker_imports_share_actual_implementation():
    from backend.app import indexing, reranker
    from backend.app.retrieval import service, reranker as actual_reranker
    assert indexing is service
    assert reranker is actual_reranker


def test_result_supports_named_fields_and_legacy_unpacking():
    result = RetrievalResult([], [], {'fingerprint':'v1', 'scope':'full'})
    selected, candidates, index = result
    assert selected is result.selected
    assert candidates is result.candidates
    assert index['scope'] == result.index['scope']


@pytest.mark.parametrize('mode', ['naive', 'hybrid', 'hybrid_rerank'])
def test_retrieve_modes_keep_contract_without_external_services(tmp_path, monkeypatch, mode):
    from backend.app.retrieval import service, reranker
    from backend.app.artifacts import write_json, write_jsonl
    from backend.app.config import Settings
    from contextlib import nullcontext
    s = Settings(_env_file=None, runtime_dir=tmp_path, final_top_k=1)
    directory = tmp_path/'indexes'/'fixture'
    write_json(tmp_path/'indexes'/'active_bge.json', {'index_dir':str(directory)})
    write_json(directory/'manifest.json', {'complete':True, 'fingerprint':'test', 'scope':'full',
        'collection':'test-only', 'config':{'model':'bge-m3', 'dimensions':1024, 'model_digest':'digest'}})
    rows = [{'book_id':'AS001', 'page':1, 'chunk_id':str(i), 'text':text}
            for i, text in enumerate(['shipping fee invoice', 'shipping booking receipt'])]
    write_jsonl(directory/'chunks.jsonl', rows)
    class Tags:
        def raise_for_status(self): pass
        def json(self): return {'models':[{'name':'bge-m3', 'digest':'digest'}]}
    monkeypatch.setattr(service.httpx, 'get', lambda *a, **k: Tags())
    monkeypatch.setattr(service, 'observation', lambda *a, **kw: nullcontext(None))
    class Store:
        def __init__(self, s): pass
        def search(self, *args): return [{**r, 'score':0.8} for r in rows]
        def close(self): pass
    monkeypatch.setattr(service, 'QdrantStore', Store)
    class Provider:
        def embed(self, *args, **kwargs): return [[0.0]*1024]
    class Ranker:
        def rank(self, question, candidates):
            return [{**c, 'rerank_score':1.0} for c in reversed(candidates)]
    monkeypatch.setattr(reranker, 'get_reranker', lambda s: Ranker())
    result = service.retrieve(s, Provider(), 'invoice', 'bge', mode)
    assert isinstance(result, RetrievalResult)
    assert len(result.selected) == 1 and len(result.candidates) == 2
    assert result.index['fingerprint'] == 'test'
    assert ('rerank_score' in result.selected[0]) == (mode == 'hybrid_rerank')
