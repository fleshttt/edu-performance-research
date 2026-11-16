"""Nested temporal validation and synthetic reproducibility safeguards."""
import numpy as np
import pandas as pd
import pytest
from edurisk.generate_data import generate_synthetic
from edurisk.ablation import TARGET, MEASURES
from edurisk.adaptive import GateParameters
from edurisk.nested_selection import (
    temporal_masks, nested_evaluate, bootstrap_paired_delta, run_nested_study,
)
from edurisk.robustness import inject_structured_missingness


def frame(n=72, seed=99):
    return generate_synthetic(n_students=n, n_terms=10, seed=seed)


def kw():
    return dict(family='logistic_regression',windows=(1,2,3),
                gate_grid=(GateParameters(0,0,.25), GateParameters(.5,.3,.8)))


def test_nested_masks_temporal_and_disjoint():
    original=frame()
    inner,development,test=temporal_masks(original)
    assert [sorted(original.loc[v,'term_index'].unique().tolist()) for _,v in inner] == [[6],[7],[8]]
    assert set(original.loc[development,'term_index'].unique())=={4,5,6,7,8}
    assert set(original.loc[test,'term_index'].unique())=={9}
    for train,val in inner:
        assert original.loc[train,'term_index'].max()<original.loc[val,'term_index'].min()
        assert not np.logical_and(test,val).any()
        assert not np.logical_and(test,train).any()


def test_invalid_temporal_protocol_rejected():
    with pytest.raises(ValueError):
        temporal_masks(frame(),inner_terms=(8,9))
    with pytest.raises(ValueError):
        temporal_masks(frame(),inner_terms=(7,6))


def test_nested_method_matches_selected_prediction_and_never_uses_test_labels_to_select():
    original=frame()
    baseline=nested_evaluate(original,**kw())
    altered=original.copy()
    altered.loc[altered.term_index.eq(9),TARGET] = 1 - altered.loc[altered.term_index.eq(9),TARGET]
    compared=nested_evaluate(altered,**kw())
    assert baseline.selection==compared.selection
    # Only TEST performance changes. This is the sole place test labels appear.
    a={r['method']:r['test_average_precision'] for r in baseline.rows}
    b={r['method']:r['test_average_precision'] for r in compared.rows}
    assert any(a[k]!=b[k] for k in a)
    assert a['selected_nested']==a[baseline.selection['chosen_method']]


def test_nested_is_deterministic_and_later_observations_do_not_change_selections():
    original=frame()
    first=nested_evaluate(original,**kw())
    second=nested_evaluate(original,**kw())
    assert first.selection==second.selection
    assert first.rows==second.rows
    third=original.copy(deep=True)
    third.loc[third.term_index.eq(9), MEASURES]=0  # outer holdout features do not inform tuning
    fourth=nested_evaluate(third,**kw())
    assert first.selection==fourth.selection
    assert any(a['test_average_precision'] != b['test_average_precision'] for a,b in zip(first.rows,fourth.rows))


def test_stress_missingness_has_finite_methods():
    original=frame()
    masked=inject_structured_missingness(original,'mnar_value',0.25,2)
    result=nested_evaluate(masked,**kw())
    assert result.selection['chosen_method'] in {'fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble'}
    assert len(result.rows)==7
    assert all(np.isfinite(r['test_average_precision']) for r in result.rows)
    assert all(0 <= r['test_average_precision'] <= 1 for r in result.rows)
    assert result.n_fits==len((1,2,3))*4


def test_paired_seed_bootstrap_and_short_interval():
    values=np.array([.02,-.01,.03,.05,.0,-.02])
    a=bootstrap_paired_delta(values)
    b=bootstrap_paired_delta(values)
    assert a==b
    assert a['ci_low'] <= a['mean'] <= a['ci_high']
    assert a['positive_count']==3
    assert bootstrap_paired_delta([.1,.2])['ci_low'] is None
    with pytest.raises(ValueError):
        bootstrap_paired_delta([float('nan')])


def test_nested_outputs_files_and_fixed_anchor(tmp_path):
    result=run_nested_study(tmp_path,seeds=(12,24,36),n_students=65,
        scenarios=(('complete',0.0),), families=('logistic_regression',),
        windows=(1,2),gate_grid=(GateParameters(0,0,.25),))
    assert result['version']=='0.6.0'
    assert result['n_experiments']==3
    assert result['n_model_fits']==3*2*4
    assert len(result['summaries'])==7
    assert any(s['method']=='selected_nested' for s in result['summaries'])
    assert (tmp_path/'nested_summary_synthetic.json').exists()
    assert (tmp_path/'nested_selections_synthetic.csv').exists()
    assert pd.read_csv(tmp_path/'nested_runs_synthetic.csv').shape[0]==3*7
