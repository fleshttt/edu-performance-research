"""Longitudinal features use only past/current semesters and respect missing semesters."""
import numpy as np
import pandas as pd
import pytest
from edurisk.history import add_history_features, HISTORY_COLUMNS, LONGITUDINAL_COLUMNS, BASE_COLUMNS
from edurisk.generate_data import generate_synthetic
from edurisk.config import TARGET_COLUMN


def _sample():
    return pd.DataFrame({
        "student_id": ["SYN-0001"] * 4,
        "term_index": [1, 2, 3, 4],
        "current_avg_grade": [3., 4., 5., 2.],
        "attendance_rate": [.5,.6,.8,.7],
        "assignment_completion_rate": [.4,.7,.8,.9],
        "credits_completion_rate": [.5,.7,.8,.9],
        "late_submission_rate": [.1,.1,.2,.3],
        TARGET_COLUMN: [0,0,1,0],
    })


def test_lags_rolling_trend_and_change():
    result = add_history_features(_sample())
    assert np.isnan(result.loc[0, 'grade_lag_1'])
    assert result.loc[0, 'history_terms_available'] == 1
    assert result.loc[2, 'grade_lag_1'] == 4
    assert result.loc[2, 'grade_lag_2'] == 3
    assert result.loc[2, 'grade_mean_3'] == 4
    assert result.loc[2, 'grade_slope_3'] == pytest.approx(1.0)
    assert result.loc[2, 'grade_std_3'] == pytest.approx(np.std([3,4,5]))
    assert result.loc[2, 'attendance_change_1'] == pytest.approx(.2)
    assert result.loc[3, 'grade_mean_3'] == pytest.approx((4+5+2)/3)


def test_no_future_or_target_leakage():
    a = _sample()
    original = add_history_features(a)
    a.loc[3,'current_avg_grade'] = 5.0
    a[TARGET_COLUMN] = 1 - a[TARGET_COLUMN]
    changed = add_history_features(a)
    pd.testing.assert_frame_equal(original.loc[:2,HISTORY_COLUMNS],changed.loc[:2,HISTORY_COLUMNS])
    assert not any(c in LONGITUDINAL_COLUMNS for c in (TARGET_COLUMN,'student_id'))


def test_gap_is_not_false_lag():
    frame = _sample().drop(index=1).reset_index(drop=True)
    result = add_history_features(frame)
    third = result.loc[result.term_index == 3].iloc[0]
    assert np.isnan(third.grade_lag_1)
    assert third.grade_lag_2 == 3
    assert third.grade_slope_3 == pytest.approx(1)
    assert third.history_terms_available == 2


def test_unsorted_rows_preserve_alignment_and_separate_students():
    raw = generate_synthetic(n_students=6)
    shuffled = raw.sample(frac=1, random_state=123).reset_index(drop=True)
    first = add_history_features(raw)
    second = add_history_features(shuffled)
    columns = ['student_id','term_index',*HISTORY_COLUMNS]
    sorter = lambda frame: frame[columns].sort_values(['student_id','term_index']).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorter(first),sorter(second))
    assert (first.loc[first.term_index==1,'history_terms_available']==1).all()


def test_reject_duplicate_student_term():
    data = _sample()
    with pytest.raises(ValueError,match='Duplicate'):
        add_history_features(pd.concat([data,data.iloc[[1]]],ignore_index=True))


def test_missing_features_not_fabricated():
    data = _sample()
    data.loc[1,'current_avg_grade']=np.nan
    features=add_history_features(data)
    assert np.isnan(features.loc[2,'grade_lag_1'])
    assert features.loc[2,'grade_mean_3']==4
    assert features.loc[2,'history_terms_available']==2
