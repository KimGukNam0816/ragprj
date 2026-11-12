"""S1-S5 (plan 4.3): retrieval failure fields and compact rank lists on retrieval spans."""
import pytest

from evaluation.retrieval.metrics import failure_fields, retrieval_scores

REF = {'evidence_groups': [[{'book_id': 'G1'}], [{'book_id': 'G2'}]]}


def chunks(*books):
    return [{'book_id': b, 'chunk_id': f'{b}-{i}'} for i, b in enumerate(books)]


@pytest.mark.parametrize('contexts,candidates,rank,cause', [
    (chunks('G1', 'X'), chunks('G1', 'X'), 1, 'hit'),
    (chunks('X'), chunks('X', 'G2', 'Y'), 2, 'context_dropped'),
    (chunks('X'), chunks('X', 'Y', 'Z', 'W', 'G1'), 5, 'rank_pushed'),
    (chunks('X'), chunks('X', 'Y'), None, 'not_in_candidates'),
    (chunks('X'), None, None, 'not_in_candidates'),
])
def test_failure_cause_and_first_gold_rank(contexts, candidates, rank, cause):
    score = failure_fields(REF, contexts, candidates)
    assert (score['first_gold_rank'], score['retrieval_failure_cause'], score['gold_doc_count']) == (rank, cause, 2)


def test_existing_keys_keep_their_values():
    old = {'hit_at_4': 1, 'recall_at_4': 0.5, 'retrieval_status': 'ok'}
    assert {k: retrieval_scores(REF, chunks('G1'))[k] for k in old} == old
    none = {**retrieval_scores({'evidence_groups': []}, []), **failure_fields({'evidence_groups': []}, [])}
    assert (none['retrieval_status'], none['retrieval_failure_cause'], none['first_gold_rank']) == \
        ('not_applicable', 'no_reference', None)


@pytest.mark.parametrize('mode', ['naive', 'hybrid', 'hybrid_rerank'])
def test_retrieval_spans_carry_ranks_without_text(tmp_path, monkeypatch, mode):
    from backend.app.artifacts import write_json, write_jsonl
    from backend.app.config import Settings
    from backend.app.retrieval import reranker, service
    from backend.tests.observability.fake_langfuse import FakeLangfuse
    fake = FakeLangfuse().install(monkeypatch)
    s = Settings(_env_file=None, runtime_dir=tmp_path, final_top_k=1)
    directory = tmp_path/'indexes'/'fixture'
    write_json(tmp_path/'indexes'/'active_bge.json', {'index_dir': str(directory)})
    write_json(directory/'manifest.json', {'complete': True, 'fingerprint': 'test', 'scope': 'full',
        'collection': 'test-only', 'config': {'model': 'bge-m3', 'dimensions': 1024, 'model_digest': 'digest'}})
    rows = [{'book_id': 'AS001', 'page': 1, 'chunk_id': str(i), 'text': text}
            for i, text in enumerate(['shipping fee invoice', 'shipping booking receipt'])]
    write_jsonl(directory/'chunks.jsonl', rows)

    class Tags:
        def json(self): return {'models': [{'name': 'bge-m3', 'digest': 'digest'}]}

    class Store:
        def __init__(self, s): pass
        def search(self, *args): return [{**r, 'score': 0.8} for r in rows]
        def close(self): pass

    class Provider:
        def embed(self, *args, **kwargs): return [[0.0]*1024]

    class Ranker:
        def rank(self, question, candidates):
            return [{**c, 'rerank_score': 1.0} for c in reversed(candidates)]

    monkeypatch.setattr(service.httpx, 'get', lambda *a, **k: Tags())
    monkeypatch.setattr(service, 'QdrantStore', Store)
    monkeypatch.setattr(reranker, 'get_reranker', lambda s: Ranker())
    result = service.retrieve(s, Provider(), 'invoice', 'bge', mode)
    dense = fake.named('dense_retrieval').metadata['ranks']
    assert dense == [{'rank': 1, 'chunk_id': '0', 'book_id': 'AS001', 'page': 1, 'score': 0.8},
                     {'rank': 2, 'chunk_id': '1', 'book_id': 'AS001', 'page': 1, 'score': 0.8}]
    if mode != 'naive':
        bm25 = fake.named('bm25_rrf').metadata
        assert isinstance(bm25['bm25_ranks'], list) and bm25['index_fingerprint'] == 'test' and 'text' not in str(bm25['ranks'])
    if mode == 'hybrid_rerank':
        assert fake.named('reranking').metadata['ranks'][0]['score'] == 1.0
    assert len(result.selected) == 1 and len(result.candidates) == 2   # search results unchanged
