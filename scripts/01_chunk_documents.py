"""Run with uv run python -m scripts.01_chunk_documents [--books AS007]."""
import argparse
from backend.app.config import get_settings
from backend.app.ingestion import build_chunks

def main():
    parser = argparse.ArgumentParser(description='Extract/chunk only Advanced shipping PDFs')
    parser.add_argument('--books', nargs='+', help='Optional explicit smoke-test subset; default all 120')
    args = parser.parse_args()
    output = build_chunks(get_settings(), args.books)
    print('CORPUS_DIR=' + str(output))

if __name__ == '__main__':
    main()
