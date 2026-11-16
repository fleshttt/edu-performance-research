"""v0.7 non-leakage, deterministic fold-policy selection and reproducibility."""
import numpy as np
import pandas as pd
import pytest
from edurisk.ablation import TARGET
from edurisk.generate_data import generate_synthetic
from edurisk.nested_selection import nested_evaluate
from edurisk.selection_policy import estimate_fold_stability, choose_stability_policies, POLICIES
from edurisk.uncertainty_selection import run_stability_study
from edurisk.adaptive import GateParameters


def test_one_se_prefers_simpler_policy_inside_tolerance():
    y=np.array([0,1,0,1,1,0,1,0,1,0,1,0]);p=np.array([.2,.9,.4,.8,.6,.5,.7,.3,.8,.1,.9,.2]);
    preds={m:p.copy() for m in ('fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble')}
    metrics=estimate_fold_stability(y,preds,[4,4,4]);choices=choose_stability_policies(metrics)
    assert choices['selected_one_se']=='fixed_val'
    assert all(v=='fixed_val' for v in choices.values())


def test_invalid_fold_sizes_and_probabilities_rejected():
    y=np.array([0,1,0,1]);preds={m:np.array([.3,.7,.2,.8]) for m in ('fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble')}
    with pytest.raises(ValueError): estimate_fold_stability(y,preds,[3,2])
    preds['fixed_val']=np.array([.3,1.7,.2,.8])
    with pytest.raises(ValueError): estimate_fold_stability(y,preds,[2,2])


def test_selector_cannot_read_test_labels():
    original=generate_synthetic(n_students=64,n_terms=10,seed=2026)
    kw={'family':'logistic_regression','windows':(1,2,3),
        'gate_grid':(GateParameters(0,0,.25),GateParameters(.5,.3,.8)),
        'include_uncertainty':True}
    a=nested_evaluate(original,**kw)
    changed=original.copy()
    changed.loc[changed.term_index.eq(9),TARGET]=1-changed.loc[changed.term_index.eq(9),TARGET]
    b=nested_evaluate(changed,**kw)
    assert a.selection==b.selection
    assert set(POLICIES).issubset({r['method'] for r in a.rows})
    assert a.n_fits==12  # 3 inner folds + 1 final fit per window
    amap={r['method']:r['test_average_precision'] for r in a.rows}
    assert all(np.isclose(amap[p],amap[a.selection['stability_policies'][p]]) for p in POLICIES)
    assert any(x['test_average_precision']!=y['test_average_precision'] for x,y in zip(a.rows,b.rows))


def test_study_writes_csv_without_outer_selection_leak(tmp_path):
    res=run_stability_study(tmp_path,seeds=(22,33,44),n_students=45,
        scenarios=(('complete',0),),families=('logistic_regression',),
        windows=(1,2),gate_grid=(GateParameters(0,0,.25),))
    assert res['version']=='0.7.0'
    assert res['n_model_fits']==3*2*4
    assert res['n_scenario_families_seeds']==3
    sel=pd.read_csv(tmp_path/'uncertainty_selections_synthetic.csv')
    assert len(sel)==3
    for p in POLICIES: assert sel[p].isin(['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble']).all()
    assert len(pd.read_csv(tmp_path/'uncertainty_runs_synthetic.csv'))==3*(5+len(POLICIES))
