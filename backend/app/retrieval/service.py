import json
import uuid
from pathlib import Path
import httpx
from ..artifacts import digest, file_hash, read_jsonl, write_json
from ..config import Settings
from ..contracts import RetrievalResult
from ..observability import observation, rank_list
from ..providers import Providers
from ..schemas import ServiceError
from .bm25 import keyword_tokens, load_bm25, _cached_bm25
from .fusion import rrf
from .dense import QdrantStore


def build_index(s: Settings, corpus: Path, profile: str, books=None, max_page=None):
    source = json.loads((corpus / 'manifest.json').read_text(encoding='utf-8'))
    if source['failure_count']:
        raise ValueError('Resolve extraction failures before indexing')
    model, dim, base_name = s.profile(profile)
    tags = httpx.get(s.ollama_base_url.rstrip('/') + '/api/tags', timeout=15).json()['models']
    matches = [t for t in tags if t['name'] in {model, model + ':latest'}]
    if not matches:
        raise ValueError('Requested embedding model is not installed')
    config = {'corpus_hash': source['corpus_hash'], 'chunk_file_sha256': file_hash(corpus / 'chunks.jsonl'), 'model': model, 'model_digest': matches[0]['digest'], 'dimensions': dim, 'books': sorted(books) if books else None, 'keyword_tokenizer': 'codes-ko-bigram-v1'}
    if max_page:
        config['max_page'] = max_page
    if s.embedding_context_tokens:
        config['embedding_context_tokens']=s.embedding_context_tokens
    fingerprint = digest(config)
    collection = ('smoke_' if max_page else '') + base_name + '_' + fingerprint[:12]
    index_dir = s.runtime_dir / 'indexes' / collection
    manifest_path = index_dir / 'manifest.json'
    chunks = [c for c in read_jsonl(corpus / 'chunks.jsonl') if (not books or c['book_id'] in books) and (not max_page or c['page'] <= max_page)]
    if not chunks or (books and set(books) != {c['book_id'] for c in chunks}):
        raise ValueError('Index selection contains unknown/empty documents')
    from ..artifacts import write_jsonl
    if not manifest_path.exists():
        write_jsonl(index_dir / 'chunks.jsonl', chunks)
        write_json(manifest_path, {'config': config, 'fingerprint': fingerprint, 'collection': collection, 'profile': profile, 'scope': 'full' if source['scope'] == 'full' and not books and not max_page else 'subset', 'expected_count': len(chunks), 'indexed_count': 0, 'complete': False, 'corpus_dir': str(corpus.resolve())})
    state = json.loads(manifest_path.read_text(encoding='utf-8'))
    if state['fingerprint'] != fingerprint:
        raise ValueError('Index manifest mismatch')
    store, provider = QdrantStore(s), Providers(s)
    try:
        store.ensure(collection, dim)
        count = store.count(collection)
        offset = state['indexed_count']
        if count < offset or count > len(chunks):
            raise ValueError('Cloud count and local checkpoint disagree; inspect before resuming')
        while offset < len(chunks):
            batch = chunks[offset:offset+s.embedding_batch_size]
            vectors = provider.embed([c['text'] for c in batch], profile)
            points = [{'id': str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint + c['chunk_id'])), 'vector': v, 'payload': c} for c, v in zip(batch, vectors)]
            store.upsert(collection, points)
            offset += len(batch)
            state['indexed_count'] = offset
            write_json(manifest_path, state)
            if offset % (s.embedding_batch_size*10) == 0 or offset == len(chunks):
                print(f'{collection}: {offset}/{len(chunks)}', flush=True)
        count = store.count(collection)
        if count != len(chunks):
            raise ValueError('Final point count mismatch')
        # BM25 is reconstructed from the same immutable chunks, never from evaluation JSON.
        state.update(complete=True, cloud_count=count, document_count=len({c['book_id'] for c in chunks}), bm25_config={'tokenizer': 'codes-ko-bigram-v1', 'k1': 1.5, 'b': .75})
        write_json(manifest_path, state)
        write_json(s.runtime_dir / 'indexes' / f'active_{profile}.json', {'index_dir': str(index_dir.resolve()), 'collection': collection, 'fingerprint': fingerprint, 'scope': state['scope']})
        return index_dir
    finally:
        store.close()


def retrieve(s: Settings, provider: Providers, question: str, profile: str, mode: str, query_vector=None) -> RetrievalResult:
    pointer = s.runtime_dir / 'indexes' / f'active_{profile}.json'
    if not pointer.exists():
        raise ServiceError('index_missing', 'Run 02_build_indexes for this embedding profile first')
    active = json.loads(pointer.read_text(encoding='utf-8'))
    index_dir = Path(active['index_dir'])
    state = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not state['complete']:
        raise ServiceError('index_incomplete', 'Selected index is incomplete')
    model, dimensions, _ = s.profile(profile)
    if state['config']['model'] != model or state['config']['dimensions'] != dimensions:
        raise ServiceError('index_mismatch', 'Embedding profile differs from index manifest')
    if state['config'].get('embedding_context_tokens') != s.embedding_context_tokens:
        raise ServiceError('index_context_mismatch','Embedding context setting changed; rebuild the versioned index')
    tags = httpx.get(s.ollama_base_url.rstrip('/') + '/api/tags', timeout=15).json()['models']
    match = [t for t in tags if t['name'] in {model, model + ':latest'}]
    if not match or match[0]['digest'] != state['config']['model_digest']:
        raise ServiceError('model_version_mismatch', 'Embedding digest changed; build a new versioned index')
    vector = query_vector  # 시맨틱 캐시 조회에서 이미 만든 벡터가 있으면 다시 임베딩하지 않는다
    if vector is None:
        with observation('query_embedding', kind='embedding', input=question, model=model) as span:
            vector = provider.embed([question], profile, query=True)[0]
            if span:
                span.update(output={'dimensions': len(vector)})
    store = QdrantStore(s)
    try:
        with observation('dense_retrieval', kind='retriever', input={'question': question, 'k': s.dense_candidate_k}) as span:
            rows = store.search(state['collection'], vector, s.dense_candidate_k)
            if span:
                span.update(output=rows, metadata={'ranks': rank_list(rows)})
    finally:
        store.close()
    if mode != 'naive':
        with observation('bm25_rrf', kind='retriever', input={'question': question, 'rrf_k': s.rrf_k}) as span:
            chunks, bm25 = load_bm25(index_dir / 'chunks.jsonl', state['fingerprint'])
            scores = bm25.get_scores(keyword_tokens(question))
            order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]['chunk_id']))[:s.bm25_candidate_k]
            keyword_rows = [{**chunks[i], 'score': float(scores[i])} for i in order if scores[i] > 0]
            rows = rrf([rows, keyword_rows], s.rrf_k)
            if span:
                span.update(output=rows, metadata={'index_fingerprint': state['fingerprint'], 'bm25_cache': 'versioned_process_cache',
                                                    'ranks': rank_list(rows), 'bm25_ranks': rank_list(keyword_rows)})
    if mode == 'hybrid_rerank':
        from .reranker import get_reranker
        with observation('reranking', kind='retriever', input={'question': question, 'candidates': rows[:s.rerank_candidate_k]}) as span:
            rows = get_reranker(s).rank(question, rows[:s.rerank_candidate_k])
            if span:
                span.update(output=rows, metadata={'ranks': rank_list(rows)})
    return RetrievalResult(rows[:s.final_top_k], rows, state)
