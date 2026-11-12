"""Point this PC's active index pointers at the index files committed in shared-index/.

The shared Qdrant collections are built once by the index builder. Every other PC needs only each
collection's manifest.json and chunks.jsonl (BM25), plus active_<profile>.json pointers that hold this
PC's path. A pointer to an index built on this PC is left alone. Writes nothing to Qdrant.
Run: uv run python -m scripts.use_shared_index
"""
import json
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / 'shared-index'


def link(shared: Path, indexes: Path) -> dict[str, str]:
    """Write active_<profile>.json in indexes for each shared index. Returns profile -> what happened."""
    found: dict[str, Path] = {}
    for manifest in sorted(shared.glob('*/manifest.json')):
        profile = json.loads(manifest.read_text(encoding='utf-8'))['profile']
        if profile in found:
            raise SystemExit(f'[FAIL] two shared indexes for profile {profile}: keep one folder in {shared}')
        found[profile] = manifest.parent
    indexes.mkdir(parents=True, exist_ok=True)
    result = {}
    for profile, index_dir in found.items():
        pointer = indexes / f'active_{profile}.json'
        if pointer.exists():
            current = Path(json.loads(pointer.read_text(encoding='utf-8'))['index_dir'])
            if current.exists() and current.resolve().parent != shared.resolve():
                result[profile] = f'kept local index {current.name}'
                continue
        state = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
        pointer.write_text(json.dumps({'index_dir': str(index_dir.resolve()), 'collection': state['collection'],
                                       'fingerprint': state['fingerprint'], 'scope': state['scope']}, indent=2),
                           encoding='utf-8')
        result[profile] = f'shared index {index_dir.name}'
    return result


if __name__ == '__main__':
    from backend.app.config import get_settings
    for profile, what in link(SHARED, get_settings().runtime_dir / 'indexes').items():
        print(f'{profile}: {what}')
