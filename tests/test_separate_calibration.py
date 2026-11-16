import pytest
import pandas as pd
from edurisk.oulad_external import split_calibration,run_external
from edurisk.oulad_fixture import make_fixture

def test_split_is_disjoint_stable_and_label_independent():
 frame=pd.DataFrame({'id_student':list(range(40))*2,'target_risk':[0,1]*40})
 a,b=split_calibration(frame)
 assert set(a.id_student).isdisjoint(b.id_student)
 assert len(a)+len(b)==len(frame)
 changed=frame.sample(frac=1,random_state=7).copy();changed.target_risk=1-changed.target_risk
 aa,bb=split_calibration(changed)
 assert set(aa.id_student)==set(a.id_student)
 assert set(bb.id_student)==set(b.id_student)

def test_separate_end_to_end(tmp_path):
 raw=make_fixture(tmp_path/'raw',n_per_group=20,seed=88)
 r=run_external(raw,tmp_path/'out',fixture=True,separate_calibration=True,bootstrap_reps=3)
 a=r['calibration_audit']
 assert a['overlap_students']==0
 assert a['selection_rows']+a['calibration_rows']==r['n_rows']['val']
 assert r['design']['calibration_protocol']=='separate'

def test_single_class_fails():
 with pytest.raises(ValueError):split_calibration(pd.DataFrame({'id_student':[1,2,3,4],'target_risk':[0]*4}))
