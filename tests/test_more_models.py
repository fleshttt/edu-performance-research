import pytest
from edurisk.oulad_fixture import make_fixture
from edurisk.oulad_external import run_external
from edurisk import api, runs
from fastapi.testclient import TestClient

@pytest.mark.parametrize('family',['hist_gradient_boosting','extra_trees','knn','gaussian_nb'])
def test_new_family_end_to_end(tmp_path,monkeypatch,family):
    raw=make_fixture(tmp_path/'raw',n_per_group=12,seed=88)
    report=run_external(raw,tmp_path/'out',family=family,fixture=True,bootstrap_reps=3)
    assert report['design']['family']==family
    for metrics in report['metrics'].values():
        assert all(0<=v<=1 for v in (metrics[k] for k in ('ap','roc_auc','brier','ece10')))
    assert report['subgroup_audit']['rows']
    monkeypatch.setattr(runs,'RUNS',tmp_path/'empty')
    response=TestClient(api.app).get('/api/external-validation',params={'family':family,'dataset':'oulad'})
    assert response.status_code==200
    assert response.json()['mode']=='NOT_RUN'

