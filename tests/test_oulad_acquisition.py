import io
import json
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from edurisk import api, runs, oulad_download
from edurisk.oulad_external import source_provenance, validate_files, REQUIRED_FILES
from edurisk.oulad_fixture import make_fixture


def test_download_receipt_and_non_overwrite(tmp_path, monkeypatch):
    original = make_fixture(tmp_path/'generator', n_per_group=5)
    binary = io.BytesIO()
    with zipfile.ZipFile(binary, 'w') as archive:
        for name in REQUIRED_FILES:
            archive.write(original/name, name)
    def response(*args, **kwargs):
        stream=io.BytesIO(binary.getvalue())
        stream.url=oulad_download.URL
        return stream
    monkeypatch.setattr(oulad_download.urllib.request,'urlopen',response)
    destination=tmp_path/'acquired'
    receipt=oulad_download.download(destination)
    assert receipt['files']==validate_files(destination)
    provenance=source_provenance(destination,validate_files(destination),False)
    assert provenance['matches_download_receipt']
    assert not provenance['verified_against_official_source']
    before=(destination/'studentInfo.csv').read_bytes()
    with pytest.raises(FileExistsError):oulad_download.download(destination)
    assert (destination/'studentInfo.csv').read_bytes()==before
    with (destination/'studentInfo.csv').open('ab') as file:file.write(b'\n')
    with pytest.raises(ValueError,match='receipt'):
        source_provenance(destination,validate_files(destination),False)


def test_unexpected_zip_never_extracts_paths(tmp_path, monkeypatch):
    binary=io.BytesIO()
    with zipfile.ZipFile(binary,'w') as archive:
        archive.writestr('../outside.txt','untrusted')
    stream=io.BytesIO(binary.getvalue());stream.url=oulad_download.URL
    monkeypatch.setattr(oulad_download.urllib.request,'urlopen',lambda *a,**k:stream)
    with pytest.raises(ValueError,match='entry'):oulad_download.download(tmp_path/'raw')
    assert not (tmp_path/'outside.txt').exists()
    assert not (tmp_path/'raw').exists()


def test_real_endpoint_does_not_fall_back_to_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(runs,'RUNS',tmp_path)
    client=TestClient(api.app)
    assert client.get('/api/external-validation?dataset=oulad').json()['mode']=='NOT_RUN'
    assert client.get('/api/external-validation?dataset=fixture').json()['mode']=='SYNTHETIC_SCHEMA_FIXTURE'
    assert client.get('/api/external-validation?dataset=unknown').status_code==422


def test_family_and_source_mismatch_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(runs,'RUNS',tmp_path)
    directory=tmp_path/('a'*32);directory.mkdir()
    (directory/'run.json').write_text(json.dumps({'id':'a'*32,'status':'completed','profile':'oulad_lr','started_at':'2026-11-03'}))
    report={'mode':'SYNTHETIC_SCHEMA_FIXTURE','design':{'family':'logistic_regression'}}
    path=directory/'external_oulad_summary.json';path.write_text(json.dumps(report))
    client=TestClient(api.app)
    assert client.get('/api/external-validation?dataset=oulad').status_code==403
    report['mode']='REAL_OULAD_EXTERNAL';report['design']['family']='random_forest'
    path.write_text(json.dumps(report))
    assert client.get('/api/external-validation?dataset=oulad').status_code==403


def test_real_profile_requires_files(tmp_path,monkeypatch):
    monkeypatch.setattr(runs,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='семь'):runs.start('oulad_lr')
