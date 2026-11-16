from pathlib import Path
import pandas as pd
import pytest
from edurisk.config import FEATURE_COLUMNS, TARGET_COLUMN
from edurisk.generate_data import generate_synthetic
from edurisk.dataset import split_chronologically, validate_dataset
from edurisk.train import run_experiment


def test_reproducible_synthetic_data():
    first = generate_synthetic(n_students=25, seed=123)
    second = generate_synthetic(n_students=25, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 25 * 5
    assert set(first[TARGET_COLUMN].unique()).issubset({0, 1})
    assert first.student_id.str.startswith('SYN-').all()


def test_chronological_split_and_no_target_leakage():
    df = generate_synthetic(n_students=35)
    tr, val, te = split_chronologically(df)
    assert set(tr.term_index.unique()) == {1, 2, 3}
    assert set(val.term_index.unique()) == {4}
    assert set(te.term_index.unique()) == {5}
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert 'student_id' not in FEATURE_COLUMNS


def test_reject_duplicate_rows():
    df = generate_synthetic(n_students=4)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_reject_invalid_rates():
    df = generate_synthetic(n_students=3)
    df.loc[0, 'attendance_rate'] = 1.5
    with pytest.raises(ValueError, match="Out-of-range"):
        validate_dataset(df)


def test_end_to_end(tmp_path: Path):
    csv = tmp_path / 'synthetic.csv'
    generate_synthetic(n_students=90).to_csv(csv, index=False)
    result = run_experiment(csv, tmp_path / 'reports')
    assert result['mode'] == 'SYNTHETIC_DEMO_ONLY'
    assert (tmp_path / 'reports/summary.json').exists()
    assert (tmp_path / 'reports/predictions_synthetic.csv').exists()
    assert result['selected_model'] in {'dummy_prior', 'logistic_regression', 'random_forest', 'history_logistic_regression', 'history_random_forest'}
    assert (tmp_path / 'reports/history_features_synthetic.csv').exists()
    assert set(result['history_vs_snapshot']) == {'logistic_regression', 'random_forest'}
