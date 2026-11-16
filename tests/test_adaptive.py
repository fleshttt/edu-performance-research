"""Safety, reproducibility and validation/test-separation checks for v0.4."""
import json
import numpy as np
import pandas as pd
import pytest

from edurisk.adaptive import (
    GateParameters, gate_statistics, select_windows,
    combine_predictions, run_adaptive,
)
from edurisk.generate_data import generate_synthetic
from edurisk.ablation import MEASURES


def sample():
    return generate_synthetic(n_students=35, n_terms=10, seed=37)


def test_quality_gate_does_not_see_future_or_labels():
    original = sample()
    stats1 = gate_statistics(original, (1, 2, 3, 4, 5))
    changed = original.copy()
    changed.loc[changed.term_index >= 8, MEASURES] = 2.0
    changed['next_term_low_performance'] = 1 - changed['next_term_low_performance']
    stats2 = gate_statistics(changed, (1, 2, 3, 4, 5))
    for k in (1, 2, 3, 4, 5):
        pd.testing.assert_frame_equal(stats1[k].loc[original.term_index <= 7],
                                      stats2[k].loc[original.term_index <= 7])
    params = GateParameters(.5, .3, .8)
    mask = np.flatnonzero(original.term_index <= 7)
    np.testing.assert_array_equal(select_windows(stats1, params, mask),
                                  select_windows(stats2, params, mask))


def test_absent_semester_is_not_mistaken_for_previous():
    data = sample()
    mask = (data.student_id == 'SYN-0001') & data.term_index.eq(3)
    data.loc[mask, MEASURES] = np.nan
    stats = gate_statistics(data, (1, 2, 3))
    i = data.index[(data.student_id == 'SYN-0001') & data.term_index.eq(4)][0]
    assert stats[1].loc[i, 'grade_count'] == 1
    assert stats[2].loc[i, 'grade_count'] == 1
    assert stats[3].loc[i, 'grade_count'] == 2
    assert stats[2].loc[i, 'coverage'] == pytest.approx(.5)
    assert stats[3].loc[i, 'coverage'] == pytest.approx(2 / 3)


def test_prediction_gating_and_invalid_input():
    p = {1: np.array([.1, .2, .3]), 2: np.array([.4, .5, .6]),
         3: np.array([.7, .8, .9])}
    np.testing.assert_allclose(combine_predictions(p, np.array([1, 3, 2])), [.1, .8, .6])
    with pytest.raises(ValueError):
        combine_predictions(p, np.array([1, 4, 3]))
    with pytest.raises(ValueError):
        gate_statistics(sample(), (0, 3))


def test_end_to_end_fixed_and_adaptive_on_same_splits(tmp_path):
    f = tmp_path / 'observations.csv'
    sample().to_csv(f, index=False)
    grid = (GateParameters(0, 0, .25), GateParameters(.5, .3, 1.3))
    r = run_adaptive(f, tmp_path / 'reports', windows=(1, 2, 4), rates=(0, .25),
                     patterns=('mcar_cells',), seeds=(7,),
                     families=('logistic_regression',), gate_grid=grid)
    assert r['design']['n_model_fits'] == 6
    assert len(r['selections']) == 2
    d = pd.read_csv(tmp_path / 'reports' / 'adaptive_test_comparison_synthetic.csv')
    assert set(d.approach) == {'snapshot_k1', 'longest_window', 'validation_selected_fixed', 'adaptive'}
    assert all(d.test_ap.between(0, 1))
    assert json.loads((tmp_path / 'reports' / 'adaptive_summary_synthetic.json').read_text())['mode'] == 'SYNTHETIC_DEMO_ONLY'


def test_test_labels_do_not_drive_window_or_gate_selection(tmp_path):
    original = sample()
    left = tmp_path / 'left.csv'; right = tmp_path / 'right.csv'
    original.to_csv(left, index=False)
    modified = original.copy()
    test = modified.term_index == 9
    modified.loc[test, 'next_term_low_performance'] = 1 - modified.loc[test, 'next_term_low_performance']
    modified.to_csv(right, index=False)
    args = dict(windows=(1, 2), rates=(0,),
                patterns=('mcar_cells',), seeds=(7,),
                families=('logistic_regression',),
                gate_grid=(GateParameters(.5, .3, .8), GateParameters(0, 0, 1.3)))
    a = run_adaptive(left, tmp_path/'outa', **args)
    b = run_adaptive(right, tmp_path/'outb', **args)
    assert a['selections'] == b['selections']
    assert a['comparison'][0]['adaptive_test_ap_mean'] != b['comparison'][0]['adaptive_test_ap_mean']
