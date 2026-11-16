"""v0.5 synthetic-only robustness protocol checks."""
import json
import numpy as np
import pandas as pd
import pytest

from edurisk.ablation import MEASURES, TARGET, window_features
from edurisk.adaptive import GateParameters
from edurisk.generate_data import generate_synthetic
from edurisk.robustness import (
    _term_split, build_cohort_dataset, inject_structured_missingness,
    evaluate_fold, validation_softmax_weights, run_robustness,
)


def sample(n=95):
    return generate_synthetic(n_students=n, n_terms=11, seed=2031)


def test_disjoint_cohort_holdout_and_chronology():
    df = build_cohort_dataset(n_per_cohort=90)
    assert df.student_id.nunique() == 360
    cohorts = {x:set(df.loc[df.cohort.eq(x), 'student_id']) for x in 'ABCD'}
    assert all(not cohorts[a].intersection(cohorts[b]) for a in 'ABCD' for b in 'ABCD' if a!=b)
    fold=_term_split(sample(),5,'temporal_forward','t','test')
    assert set(fold.data.loc[fold.train,'term_index'])=={5,6,7}
    assert set(fold.data.loc[fold.validation,'term_index'])=={8}
    assert set(fold.data.loc[fold.test,'term_index'])=={9}


@pytest.mark.parametrize('mechanism,rate', [
    ('complete',0.0),('mar_attendance',0.25),
    ('mnar_value',0.25),('contiguous_terms',0.25),
])
def test_missingness_preserves_keys_and_labels(mechanism, rate):
    raw=sample(30)
    changed=inject_structured_missingness(raw,mechanism,rate,seed=7)
    pd.testing.assert_frame_equal(raw[['student_id','term_index',TARGET]],changed[['student_id','term_index',TARGET]])
    assert len(raw)==len(changed)
    assert changed[MEASURES].isna().sum().sum() == 0 if rate==0 else changed[MEASURES].isna().sum().sum() > 0
    assert changed[MEASURES].notna().sum().sum() > 0
    assert not changed[MEASURES].where(changed[MEASURES].notna()).stack().empty
    assert changed[MEASURES].notna().to_numpy().sum() <= raw[MEASURES].notna().to_numpy().sum()
    if mechanism=='mar_attendance':
        assert changed.attendance_rate.notna().all()


def test_missingness_is_deterministic_and_future_does_not_change_past_features():
    raw=sample(30)
    first=inject_structured_missingness(raw,'contiguous_terms',0.25,22)
    second=inject_structured_missingness(raw,'contiguous_terms',0.25,22)
    pd.testing.assert_frame_equal(first,second)
    X1,_=window_features(first,5)
    changed=first.copy(deep=True)
    changed.loc[changed.term_index.ge(9),MEASURES]=np.nan
    changed[TARGET]=1-changed[TARGET]
    X2,_=window_features(changed,5)
    pd.testing.assert_frame_equal(X1.loc[first.term_index.le(8)],X2.loc[first.term_index.le(8)])


def test_weights_and_invalid_parameter():
    w=validation_softmax_weights({1:0.6,2:0.8,3:0.7})
    assert sum(w.values())==pytest.approx(1)
    assert w[2]>w[3]>w[1]
    with pytest.raises(ValueError):
        validation_softmax_weights({1:0.7},temperature=0)


def test_test_outcomes_cannot_change_hyperparameter_selection():
    df=sample()
    fold=_term_split(df,5,'temporal_forward','t1','example')
    params=(GateParameters(0,0,.25),GateParameters(.5,.3,.8))
    kwargs=dict(windows=(1,2,3), families=('logistic_regression',),
                scenarios=(('complete',0.0),), gate_grid=params)
    result_a,selection_a,_=evaluate_fold(fold,**kwargs)
    altered=df.copy(deep=True)
    altered.loc[fold.test,TARGET]=1-altered.loc[fold.test,TARGET]
    fold_b=_term_split(altered,5,'temporal_forward','t1','example')
    result_b,selection_b,_=evaluate_fold(fold_b,**kwargs)
    assert selection_a==selection_b
    a={r['method']:r['test_average_precision'] for r in result_a}
    b={r['method']:r['test_average_precision'] for r in result_b}
    assert any(a[k]!=b[k] for k in a)


def test_smoke_end_to_end_artifacts(tmp_path):
    csv=tmp_path/'legacy.csv'
    sample(95).to_csv(csv,index=False)
    result=run_robustness(csv,tmp_path/'out',protocols=('legacy_anchor',),
        windows=(1,2),families=('logistic_regression',),
        scenarios=(('complete',0.0),('mar_attendance',0.25)),
        gate_grid=(GateParameters(0,0,.25),GateParameters(.5,0,.8)))
    assert result['n_trained_models']==4
    assert result['mode']=='SYNTHETIC_DEMO_ONLY'
    assert len(result['aggregate_results']) == 12
    assert len(result['comparative_deltas']) == 6
    saved=json.loads((tmp_path/'out'/'robustness_summary_synthetic.json').read_text(encoding='utf-8'))
    assert saved['version']=='0.5.0'
    assert (tmp_path/'out'/'robustness_runs_synthetic.csv').exists()
