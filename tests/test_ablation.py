"""v0.3 guardrails and reproducibility tests."""
import json
import numpy as np
import pandas as pd
import pytest
from edurisk.ablation import (inject_missingness, window_features, run_ablation,
                              MEASURES, validate_experiment_data)
from edurisk.generate_data import generate_synthetic


def sample():
    return generate_synthetic(n_students=30, n_terms=10, seed=37)


def test_lookback_is_exact_and_past_only():
    data = sample()
    old, _ = window_features(data, 5)
    modified = data.copy()
    modified.loc[modified['term_index'] >= 8, MEASURES] = 2.0
    modified.loc[modified['term_index'] >= 8, 'next_term_low_performance'] = 1
    new, _ = window_features(modified, 5)
    pd.testing.assert_frame_equal(old.loc[data.term_index <= 7], new.loc[data.term_index <= 7])


def test_history_length_alters_window_not_future():
    data = sample()
    first, one = window_features(data, 1)
    later, five = window_features(data, 5)
    assert one == list(first.columns)
    assert 'grade_lag_4' in five and 'grade_lag_4' not in one
    pd.testing.assert_frame_equal(first, later[first.columns])
    t5 = data.term_index.eq(5)
    assert np.allclose(later.loc[t5, 'grade_lag_4'],
                       data.loc[data.term_index.eq(1), 'current_avg_grade'].to_numpy())


def test_missing_semester_does_not_shift_lag():
    df = sample()
    df = df[~((df.student_id == 'SYN-0001') & (df.term_index == 3))]
    f, _ = window_features(df, 3)
    row = f.loc[(df.student_id == 'SYN-0001') & (df.term_index == 4)].iloc[0]
    assert np.isnan(row.grade_lag_1)
    assert row.grade_lag_2 == pytest.approx(float(df.loc[(df.student_id == 'SYN-0001') &
                                                  (df.term_index == 2), 'current_avg_grade'].iloc[0]))


def test_monotone_masks_and_preserve_labels():
    df = sample()
    for pattern in ('mcar_cells', 'blocked_terms'):
        a = inject_missingness(df, .1, pattern, 7)
        b = inject_missingness(df, .4, pattern, 7)
        assert not (a[MEASURES].isna().to_numpy() & ~b[MEASURES].isna().to_numpy()).any()
        pd.testing.assert_series_equal(a['next_term_low_performance'], df['next_term_low_performance'])
        pd.testing.assert_series_equal(b['student_id'], df['student_id'])
        assert (b[MEASURES].isna().to_numpy().mean()) > (a[MEASURES].isna().to_numpy().mean())
        if pattern == 'blocked_terms':
            assert b[MEASURES].isna().all(axis=1).equals(b[MEASURES].isna().any(axis=1))


def test_reproducibility_and_invalid_input():
    df=sample()
    pd.testing.assert_frame_equal(inject_missingness(df, .25, 'mcar_cells', 33),
                                  inject_missingness(df, .25, 'mcar_cells', 33))
    with pytest.raises(ValueError):
        inject_missingness(df, 1.0, 'mcar_cells', 7)
    with pytest.raises(ValueError):
        window_features(pd.concat([df, df.iloc[:1]], ignore_index=True), 4)
    validate_experiment_data(df)


def test_end_to_end_ablation(tmp_path):
    dataset = tmp_path / 'data.csv'
    sample().to_csv(dataset, index=False)
    r = run_ablation(dataset, tmp_path / 'reports', windows=(1, 2, 4),
                     rates=(0, .2), patterns=('mcar_cells',), seeds=(8,),
                     families=('logistic_regression',))
    assert r['n_model_fits'] == 6
    assert len(r['window_selections']) == 2
    assert r['design']['split_rows'] == {'train': 90, 'validation': 30, 'test': 30}
    assert (tmp_path / 'reports' / 'ablation_aggregates_synthetic.csv').is_file()
    assert json.loads((tmp_path / 'reports' / 'ablation_summary_synthetic.json').read_text())['mode'] == 'SYNTHETIC_DEMO_ONLY'
