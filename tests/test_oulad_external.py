import numpy as np
import pandas as pd
import pytest
from edurisk.oulad_fixture import make_fixture
from edurisk.oulad_external import (validate_files, build_features, split_no_overlap,
        run_external, _adaptive_indices, _group_stability)

@pytest.fixture(scope='module')
def fixture_data(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp('oulad_fake'),n_per_group=12,seed=88)


def test_schema(fixture_data):
    assert len(validate_files(fixture_data))==7


def test_missing_csv_refused(fixture_data, tmp_path):
    (tmp_path/'studentInfo.csv').write_text('id_student\n1\n')
    with pytest.raises((FileNotFoundError,ValueError)):validate_files(tmp_path)


def test_history_never_sees_future_or_status(fixture_data):
    first=build_features(fixture_data,chunksize=91)
    assert 'final_result' not in first.columns
    assert 'date_unregistration' not in first.columns
    # Rewrite assessments after the prediction cutoff. Prediction features
    # must be invariant to these values, including their IDs.
    file=fixture_data/'studentAssessment.csv'
    original=file.read_bytes()
    try:
        sa=pd.read_csv(file)
        sa.loc[sa.date_submitted>84,'score']=0
        sa.to_csv(file,index=False)
        second=build_features(fixture_data,chunksize=132)
        pd.testing.assert_frame_equal(first,second)
    finally:
        file.write_bytes(original)


def test_chunk_invariance(fixture_data):
    a=build_features(fixture_data,chunksize=29)
    b=build_features(fixture_data,chunksize=900000)
    pd.testing.assert_frame_equal(a,b)


def test_splits_are_disjoint(fixture_data):
    parts=split_no_overlap(build_features(fixture_data))
    assert not(set(parts['train'].id_student)&set(parts['test'].id_student))
    assert not(set(parts['val'].id_student)&set(parts['test'].id_student))
    assert all(x.target_risk.nunique()==2 for x in parts.values())


def test_gate_is_label_independent(fixture_data):
    frame=build_features(fixture_data)
    a=_adaptive_indices(frame,(14,28,56,84))
    changed=frame.copy()
    changed['target_risk']=1-changed['target_risk']
    b=_adaptive_indices(changed,(14,28,56,84))
    np.testing.assert_equal(a,b)


def test_external_pipeline_aggregate_only(fixture_data,tmp_path):
    result=run_external(fixture_data,tmp_path,fixture=True,bootstrap_reps=40)
    assert result['mode']=='SYNTHETIC_SCHEMA_FIXTURE'
    assert result['n_rows']['test']>0
    assert result['selection']['selected_method'] in {'fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble'}
    assert all('risk_probability' not in str(x) for x in result)
    assert (tmp_path/'external_oulad_summary.json').exists()


def test_real_data_is_not_claimed_by_fixture(fixture_data,tmp_path):
    r=run_external(fixture_data,tmp_path,fixture=True,bootstrap_reps=20)
    assert r['mode']!='REAL_OULAD_EXTERNAL'


def test_holdout_labels_do_not_change_selection(fixture_data,tmp_path):
    """Changing future final outcomes may change test metrics but not selection."""
    original_path=fixture_data/'studentInfo.csv'
    content=original_path.read_bytes()
    try:
        a=run_external(fixture_data,tmp_path/'a',fixture=True,bootstrap_reps=20)
        df=pd.read_csv(original_path)
        mask=df.code_presentation.eq('2014J')
        swap={'Fail':'Pass','Withdrawn':'Distinction','Pass':'Fail','Distinction':'Withdrawn'}
        df.loc[mask,'final_result']=df.loc[mask,'final_result'].replace(swap)
        df.to_csv(original_path,index=False)
        b=run_external(fixture_data,tmp_path/'b',fixture=True,bootstrap_reps=20)
        assert a['selection']==b['selection']
        assert a['metrics']['fixed_val']['ap']!=b['metrics']['fixed_val']['ap']
    finally:
        original_path.write_bytes(content)


def test_prior_withdrawal_is_excluded(fixture_data):
    register_file=fixture_data/'studentRegistration.csv'
    original=register_file.read_bytes()
    try:
        reg=pd.read_csv(register_file)
        sid=reg.iloc[0].id_student
        reg.loc[0,'date_unregistration']=20
        reg.to_csv(register_file,index=False)
        x=build_features(fixture_data)
        assert sid not in set(x.id_student)
    finally:
        register_file.write_bytes(original)


def test_future_vle_events_excluded(fixture_data):
    vle_file=fixture_data/'studentVle.csv'
    original=vle_file.read_bytes()
    try:
        baseline=build_features(fixture_data)
        vle=pd.read_csv(vle_file)
        r=vle.iloc[0].copy()
        r['date']=190;r['sum_click']=9999
        vle=pd.concat([vle,pd.DataFrame([r])],ignore_index=True)
        vle.to_csv(vle_file,index=False)
        result=build_features(fixture_data)
        pd.testing.assert_frame_equal(baseline,result)
    finally:
        vle_file.write_bytes(original)
