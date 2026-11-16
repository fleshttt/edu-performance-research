"""Source schedule integrity and public-material boundary checks."""
from fastapi.testclient import TestClient

from edurisk import api

client = TestClient(api.app)


def test_plan_covers_all_months_and_retains_source_constraints():
    response = client.get('/api/research-plan')
    assert response.status_code == 200
    plan = response.json()
    expected = [f'{2026 + (9 + n) // 12}-{(9 + n) % 12 + 1:02}' for n in range(36)]
    assert [m['id'] for m in plan['months']] == expected
    stages = {s['id']: s for s in plan['stages']}
    assert len(stages) == 6
    for month in plan['months']:
        stage = stages[month['stage_id']]
        assert stage['start'] <= month['id'] <= stage['end']
        assert month['tasks'] and month['deliverables']
        assert 'status' not in month
    assert plan['period']['enrollment_end'] == '2029-10-01'
    assert any(r['id'] == 'dissertation-80' for r in plan['requirements'])
    assert any(n['id'] == 'eighty-percent' for n in plan['alignment_notes'])
    november = next(m for m in plan['months'] if m['id'] == '2026-11')
    assert november['deliverables'] == ['Реестр источников с аннотациями и критериями отбора.']


def test_material_links_are_complete_and_serve_expected_files():
    plan = client.get('/api/research-plan').json()
    materials = {m['id']: m for m in plan['materials']}
    assert set(materials) == set(api.MATERIAL_FILES)
    for entry in plan['months'] + plan['stages']:
        assert set(entry['material_ids']) <= set(materials)
    for entry in materials.values():
        result = client.get(entry['url']+'?format=source')
        assert result.status_code == 200
        assert result.content == (api.ROOT / api.MATERIAL_FILES[entry['id']]).read_bytes()
        assert result.headers['x-content-type-options'] == 'nosniff'


def test_arbitrary_files_and_paths_are_not_public():
    for path in ['.env', 'README.md', 'plan.json', '..', 'src%2Fedurisk%2Fapi.py',
                 '%2E%2E%2Frequirements.txt', 'selected_model_synthetic.joblib']:
        assert client.get('/api/materials/' + path).status_code == 404


def test_material_redirect_outside_allowed_directories_is_rejected(tmp_path, monkeypatch):
    secret = tmp_path / 'private.md'
    secret.write_text('private content', encoding='utf-8')
    monkeypatch.setitem(api.MATERIAL_FILES, 'problem', str(secret))
    result = client.get('/api/materials/problem')
    assert result.status_code == 404
    assert b'private content' not in result.content


def test_plan_unavailable_is_reported_without_server_error(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'ROOT', tmp_path)
    assert client.get('/api/research-plan').status_code == 503
