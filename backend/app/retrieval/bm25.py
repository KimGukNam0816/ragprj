import re
import threading
from functools import lru_cache
from pathlib import Path
from rank_bm25 import BM25Okapi
from ..artifacts import read_jsonl
from ..schemas import ServiceError


def keyword_tokens(text):
    # Codes stay intact; Korean character bigrams improve inflection matching without another service.
    result = re.findall(r'[a-z0-9]+(?:[-_:./][a-z0-9]+)*|[가-힣]+', text.lower())
    return result + [word[i:i+2] for word in result if re.fullmatch('[가-힣]+', word) for i in range(len(word)-1)]


_bm25_lock = threading.Lock()


@lru_cache(maxsize=2)
def _cached_bm25(path, fingerprint, size, modified_ns):
    # 활성 버전 또는 파일이 바뀌면 새 객체를 만들고, 같은 버전은 재사용한다.
    chunks = list(read_jsonl(Path(path)))
    if not chunks:
        raise ServiceError('index_empty', 'BM25 chunks are empty')
    return chunks, BM25Okapi([keyword_tokens(c['text']) for c in chunks])


def load_bm25(path, fingerprint):
    path = Path(path).resolve()
    stat = path.stat()
    with _bm25_lock:
        return _cached_bm25(str(path), fingerprint, stat.st_size, stat.st_mtime_ns)


