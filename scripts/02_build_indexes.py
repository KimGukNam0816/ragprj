import argparse
from pathlib import Path
from backend.app.config import get_settings
from backend.app.indexing import build_index

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Create/resume a versioned Qdrant index; never deletes collections')
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--embedding', choices=['bge', 'qwen'], required=True)
    p.add_argument('--books', nargs='+', help='Explicit document subset for a development smoke test')
    p.add_argument('--max-page', type=int, help='Fixed first-N-page smoke sample, independent of evaluation answers')
    args = p.parse_args()
    print('INDEX_DIR=' + str(build_index(get_settings(), args.corpus, args.embedding, args.books, args.max_page)))
