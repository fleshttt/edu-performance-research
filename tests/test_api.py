"""Smoke tests for read-only synthetic dashboard endpoints."""
from fastapi.testclient import TestClient
from edurisk.api import app

client = TestClient(app)


def test_summary_and_comparison():
    response = client.get('/api/summary')
    assert response.status_code == 200
    data = response.json()
    assert data['version'] == '0.2.0'
    assert len(data['metrics']) == 5
    assert 'history_vs_snapshot' in data


def test_single_trajectory_and_cohort():
    response = client.get('/api/students/SYN-0001/trajectory')
    assert response.status_code == 200
    data = response.json()
    assert data['student_id'] == 'SYN-0001'
    assert [r['term_index'] for r in data['observations']] == [1,2,3,4,5]
    assert data['forecast']['next_term'] == 6
    result = client.get('/api/term-trends')
    assert result.status_code == 200
    assert len(result.json()['rows']) == 5


def test_non_synthetic_id_rejected():
    assert client.get('/api/students/student-1/trajectory').status_code == 400
    assert client.get('/api/students/SYN-99999/trajectory').status_code == 404


def test_ablation_api_results_are_synthetic_and_serializable():
    response = client.get('/api/history-ablation')
    assert response.status_code == 200
    report = response.json()
    assert report['version'] == '0.3.0'
    assert report['mode'] == 'SYNTHETIC_DEMO_ONLY'
    assert len(report['design']['windows']) == 5
    assert len(report['rows']) > 0
    assert len(report['window_selections']) > 0
    assert client.get('/api/health').json()['version'] == '0.8.0'


def test_adaptive_api_aggregates_only_synthetic_results():
    response = client.get('/api/adaptive-history')
    assert response.status_code == 200
    result = response.json()
    assert result['mode'] == 'SYNTHETIC_DEMO_ONLY'
    assert result['version'] == '0.4.0'
    assert result['design']['n_model_fits'] == 190
    assert len(result['comparison']) == 14
    assert len(result['window_usage']) > 0


def test_robustness_api_synthetic_summary():
    response=client.get('/api/robustness')
    assert response.status_code==200
    payload=response.json()
    assert payload['version']=='0.5.0'
    assert payload['mode']=='SYNTHETIC_DEMO_ONLY'
    assert payload['n_folds']>=3
    assert len(payload['aggregate_results'])>0
    assert len(payload['comparative_deltas'])>0


def test_nested_selection_api():
    response = client.get('/api/nested-selection')
    assert response.status_code == 200
    data = response.json()
    assert data['mode'] == 'SYNTHETIC_DEMO_ONLY'
    assert data['version'] == '0.6.0'
    assert data['n_seed_replicates'] >= 3
    assert any(r['method']=='selected_nested' for r in data['summaries'])


def test_uncertainty_endpoint_synthetic_only():
    from edurisk.api import uncertainty_selection
    result = uncertainty_selection()
    assert result['mode'] == 'SYNTHETIC_DEMO_ONLY'
    assert result['version'] == '0.7.0'
    assert any(row['policy']=='selected_stability_1' for row in result['summaries'])


def test_external_api_never_exposes_raw_identifiers():
    response=client.get('/api/external-validation')
    assert response.status_code==200
    result=response.json()
    assert result['version']=='0.8.0'
    assert result['mode'] in ('SYNTHETIC_SCHEMA_FIXTURE','REAL_OULAD_EXTERNAL','NOT_RUN')
    assert 'source_file_hashes' not in result
    assert 'individual_predictions' not in result
    assert 'id_student' not in result
