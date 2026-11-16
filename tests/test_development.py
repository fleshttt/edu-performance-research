import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from edurisk import api, runs
from edurisk.oulad_fixture import make_fixture
from edurisk.oulad_external import run_external, validate_files, source_provenance

client = TestClient(api.app)


def test_generated_fixture_cannot_claim_real_origin(tmp_path):
    directory = make_fixture(tmp_path / 'renamed', n_per_group=6)
    with pytest.raises(ValueError, match='Synthetic fixture'):
        run_external(directory, tmp_path / 'out', fixture=False, bootstrap_reps=2)
    assert not (tmp_path / 'out').exists()


def test_bundled_fixture_detected_even_without_marker(tmp_path):
    import shutil
    fixture = api.ROOT / 'data' / 'oulad' / 'fixture'
    for file in fixture.glob('*.csv'):
        shutil.copy(file, tmp_path / file.name)
    with pytest.raises(ValueError, match='Synthetic fixture'):
        source_provenance(tmp_path, validate_files(tmp_path), False)


def test_required_withdrawal_column_validated(tmp_path):
    import pandas as pd
    directory = make_fixture(tmp_path, n_per_group=6)
    path = directory / 'studentRegistration.csv'
    pd.read_csv(path).drop(columns='date_unregistration').to_csv(path, index=False)
    with pytest.raises(ValueError, match='date_unregistration'):
        validate_files(directory)


def test_run_rejects_unknown_profile_and_cross_origin():
    assert client.post('/api/runs', json={'profile':'arbitrary_shell'}).status_code == 400
    assert client.post('/api/runs', json={'profile':'fixture_lr'}, headers={'Origin':'https://outside.example'}).status_code == 403
    assert client.post('/api/runs', json={'profile':'fixture_lr'}, headers={'Sec-Fetch-Site':'cross-site'}).status_code == 403


def test_one_run_at_a_time_and_restart_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, 'RUNS', tmp_path)
    monkeypatch.setattr(runs, '_active', None)
    monkeypatch.setattr(runs, '_pool', Mock())
    first = runs.start('fixture_lr')
    with pytest.raises(RuntimeError):
        runs.start('fixture_rf')
    assert runs.history()[0]['status'] == 'running'
    monkeypatch.setattr(runs, '_active', None)
    assert runs.history()[0]['status'] == 'interrupted'
    assert json.loads((tmp_path / first['id'] / 'run.json').read_text())['profile'] == 'fixture_lr'


def test_failed_run_is_recorded_and_slot_released(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, '_active', 'test')
    monkeypatch.setattr(runs.subprocess, 'run', Mock(side_effect=OSError('test failure')))
    record = {'id':'test','profile':'fixture_lr','status':'running','started_at':'2026-11-03'}
    runs._execute(tmp_path, record)
    result=json.loads((tmp_path / 'run.json').read_text())
    assert result['status']=='failed' and result['finished_at']
    assert runs._active is None


def test_run_artifact_boundary():
    for url in ['/api/runs/not-an-id/log','/api/runs/'+'a'*32+'/private','/api/runs/'+'a'*32+'/summary']:
        assert client.get(url).status_code == 404


def test_missing_raw_files_are_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(api,'ROOT',tmp_path)
    response=client.get('/api/development')
    assert response.status_code==200
    assert not response.json()['schema_ready']
    assert len(response.json()['files'])==7
