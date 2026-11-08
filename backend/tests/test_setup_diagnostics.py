import json
from types import SimpleNamespace

from scripts import check_integration


def test_setup_stops_at_failure_and_redacts_exception(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_integration, 'settings', lambda: SimpleNamespace(runtime_dir=tmp_path))
    calls = []

    def fake_check(stage, args):
        calls.append(stage)
        if stage == 'env':
            raise RuntimeError('private-token-must-not-be-printed')
        return {'ok': True}

    monkeypatch.setattr(check_integration, 'check', fake_check)
    assert check_integration.main(['--stage', 'all']) == 1
    report = json.loads(next((tmp_path / 'checks').glob('setup-*.json')).read_text(encoding='utf-8'))
    assert calls == ['tools', 'env']
    assert report['success'] is False
    assert [r['status'] for r in report['checks']][:3] == ['PASS', 'FAIL', 'SKIP']
    assert all(r['status'] == 'SKIP' for r in report['checks'][2:])
    assert 'private-token' not in capsys.readouterr().out
    assert 'private-token' not in json.dumps(report)
    assert report['checks'][1]['error'] == 'RuntimeError'


def test_single_stage_success_has_report_and_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(check_integration, 'settings', lambda: SimpleNamespace(runtime_dir=tmp_path))
    monkeypatch.setattr(check_integration, 'check', lambda stage, args: {'stage': stage})
    assert check_integration.main(['--stage', 'api']) == 0
    report = json.loads(next((tmp_path / 'checks').glob('setup-*.json')).read_text(encoding='utf-8'))
    assert report['checks'][0]['stage'] == 'api'
    assert report['checks'][0]['status'] == 'PASS'
