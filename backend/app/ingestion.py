import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from pypdf import PdfReader
from tokenizers import Tokenizer

from .artifacts import digest, file_hash, read_jsonl, write_json, write_jsonl
from .config import Settings


class TokenBudgets:
    """Exact embedding tokenizer counts; identical boundaries for both profiles."""
    def __init__(self):
        self.tokenizers = []
        self.revisions = {}
        for repo in ['BAAI/bge-m3', 'Qwen/Qwen3-Embedding-0.6B']:
            path = hf_hub_download(repo_id=repo, filename='tokenizer.json')
            tokenizer = Tokenizer.from_file(path)
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self.tokenizers.append(tokenizer)
            self.revisions[repo] = {'tokenizer_sha256': file_hash(Path(path)), 'snapshot': Path(path).parent.name}

    def counts(self, text):
        return [len(t.encode(text, add_special_tokens=False).ids) for t in self.tokenizers]

    def split(self, text, target, overlap):
        if not 0 <= overlap < target:
            raise ValueError('Chunk overlap must be smaller than target')
        start = 0
        while start < len(text):
            remaining = text[start:]
            encs = [t.encode(remaining, add_special_tokens=False) for t in self.tokenizers]
            ends = [e.offsets[target - 1][1] if len(e.ids) > target else len(remaining) for e in encs]
            length = min(ends)
            boundary = remaining.rfind('\n', 0, length)
            if boundary > length * .65:
                length = boundary
            piece = remaining[:max(length, 1)]
            if piece.strip():
                yield piece.strip()
            end = start + max(length, 1)
            if end >= len(text):
                break
            offsets = [t.encode(text[start:end], add_special_tokens=False).offsets for t in self.tokenizers]
            # The smallest overlap start preserves at least the requested overlap for both.
            overlap_start = min((o[-overlap][0] if overlap and len(o) > overlap else len(piece)) for o in offsets)
            start = max(start + 1, start + overlap_start)


def discover_documents(s: Settings):
    dataset = s.data_dir / 'advanced'
    files = sorted(dataset.glob('*/documents/*.pdf'))
    if len(files) != 120 or len({p.stem for p in files}) != 120:
        raise ValueError('Advanced shipping requires exactly 120 unique PDFs in data/advanced/*/documents/')
    expected = {f'AS{i:03d}' for i in range(1, 121)}
    if {p.stem for p in files} != expected:
        raise ValueError('Document IDs must be AS001..AS120')
    return files


def build_chunks(s: Settings, books: list[str] | None = None):
    files = discover_documents(s)
    if books:
        if not set(books) <= {p.stem for p in files}:
            raise ValueError('Unknown book ID')
        files = [p for p in files if p.stem in books]
    budgets = TokenBudgets()
    config = {'version': 'page-text-v1', 'target': s.chunk_tokens, 'overlap': s.chunk_overlap_tokens, 'max': 1000, 'tokenizers': budgets.revisions}
    config_hash = digest(config)
    scope = 'all' if not books else digest(sorted(books))[:8]
    output = s.runtime_dir / 'corpora' / f'advanced-{config_hash[:10]}-{scope}'
    documents = []
    for path in files:
        source_hash = file_hash(path)
        cache = output / 'documents' / f'{path.stem}.json'
        if cache.exists():
            cached = json.loads(cache.read_text(encoding='utf-8'))
            if cached['source_sha256'] == source_hash and cached['status'] == 'ok':
                documents.append(cached)
                print(f'{path.stem}: resumed {cached["chunks"]} chunks', flush=True)
                continue
        pages, chunks, failures = [], [], []
        try:
            reader = PdfReader(path)
            for number, page in enumerate(reader.pages, 1):
                try:
                    raw = (page.extract_text() or '').replace('\x00', '')
                    raw = re.sub(r'(?m)^\s*\d+\s*/\s*150\s*$', '', raw).strip()
                    if not raw:
                        raise ValueError('Empty page text')
                    pages.append({'book_id': path.stem, 'page': number, 'text': raw})
                    heading = '\n'.join(raw.splitlines()[:3])
                    for idx, part in enumerate(budgets.split(raw, s.chunk_tokens, s.chunk_overlap_tokens)):
                        text = heading + '\n\n' + part if not part.startswith(heading) else part
                        counts = budgets.counts(text)
                        if max(counts) > 1000:
                            raise ValueError('Chunk exceeds hard embedding token budget')
                        cid = digest([path.stem, number, idx, text, source_hash, config_hash])
                        chunks.append({'chunk_id': cid, 'book_id': path.stem, 'page': number, 'section_title': heading, 'text': text, 'source_path': str(path.relative_to(s.data_dir)), 'source_sha256': source_hash, 'chunker_hash': config_hash, 'embedding_token_counts': counts})
                except Exception as exc:
                    failures.append({'book_id': path.stem, 'page': number, 'error_type': type(exc).__name__})
            status = 'ok' if not failures else 'partial'
            total_pages = len(reader.pages)
        except Exception as exc:
            failures.append({'book_id': path.stem, 'page': None, 'error_type': type(exc).__name__})
            status, total_pages = 'error', 0
        write_jsonl(output / 'pages' / f'{path.stem}.jsonl', pages)
        write_jsonl(output / 'chunks' / f'{path.stem}.jsonl', chunks)
        record = {'book_id': path.stem, 'source_sha256': source_hash, 'source_path': str(path.relative_to(s.data_dir)), 'pages': total_pages, 'extracted_pages': len(pages), 'chunks': len(chunks), 'status': status, 'failures': failures}
        write_json(cache, record)
        documents.append(record)
        print(f'{path.stem}: pages={total_pages}, chunks={len(chunks)}, failures={len(failures)}', flush=True)
    write_jsonl(output / 'chunks.jsonl', (c for doc in documents for c in read_jsonl(output / 'chunks' / (doc['book_id'] + '.jsonl'))))
    manifest = {'dataset': 'advanced_shipping', 'scope': 'full' if not books else 'subset', 'config': config, 'config_hash': config_hash, 'documents': documents, 'document_count': len(documents), 'page_count': sum(d['pages'] for d in documents), 'chunk_count': sum(d['chunks'] for d in documents), 'failure_count': sum(len(d['failures']) for d in documents)}
    manifest['corpus_hash'] = digest([config_hash, [(d['book_id'], d['source_sha256']) for d in documents]])
    write_json(output / 'manifest.json', manifest)
    return output
