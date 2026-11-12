import json

import pytest

from scripts.use_shared_index import link


def _index(root, name, profile):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps({'profile': profile, 'collection': name,
                                                      'fingerprint': 'f-' + name, 'scope': 'subset'}), encoding='utf-8')
    return folder


def _pointer(indexes, profile):
    return json.loads((indexes / f'active_{profile}.json').read_text(encoding='utf-8'))


def test_writes_pointers_with_this_pc_path(tmp_path):
    shared, indexes = tmp_path / 'shared', tmp_path / 'rt' / 'indexes'
    bge = _index(shared, 'c_bge', 'bge')
    _index(shared, 'c_qwen', 'qwen')
    assert set(link(shared, indexes)) == {'bge', 'qwen'}
    assert _pointer(indexes, 'bge') == {'index_dir': str(bge.resolve()), 'collection': 'c_bge',
                                        'fingerprint': 'f-c_bge', 'scope': 'subset'}


def test_keeps_local_index_but_replaces_dead_or_shared_pointer(tmp_path):
    shared, indexes = tmp_path / 'shared', tmp_path / 'indexes'
    _index(shared, 'c_bge', 'bge')
    _index(shared, 'c_qwen', 'qwen')
    local = _index(indexes, 'local_bge', 'bge')
    indexes.mkdir(exist_ok=True)
    (indexes / 'active_bge.json').write_text(json.dumps({'index_dir': str(local)}), encoding='utf-8')
    (indexes / 'active_qwen.json').write_text(json.dumps({'index_dir': '/builder/pc/gone'}), encoding='utf-8')
    result = link(shared, indexes)
    assert result['bge'].startswith('kept local')
    assert _pointer(indexes, 'qwen')['collection'] == 'c_qwen'
    assert link(shared, indexes)['qwen'].startswith('shared')   # rerun refreshes its own pointer


def test_two_shared_indexes_for_one_profile_fail(tmp_path):
    _index(tmp_path / 'shared', 'a', 'bge')
    _index(tmp_path / 'shared', 'b', 'bge')
    with pytest.raises(SystemExit):
        link(tmp_path / 'shared', tmp_path / 'indexes')
