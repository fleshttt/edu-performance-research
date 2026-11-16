import numpy as np
import pandas as pd
from edurisk.oulad_external import subgroup_audit

def test_group_audit_counts_boundaries_and_errors():
    train=pd.DataFrame({'clicks_84':[0,2,4,6]})
    test=pd.DataFrame({'clicks_84':[0]*40+[4]*40+[9]*40,'code_module':['A']*120,'target_risk':[0,1]*60})
    result=subgroup_audit(train,test,{'selected':np.ones(120)*.8},84)
    assert result['activity_boundary']==4
    rows=[r for r in result['rows'] if r['dimension']=='activity']
    assert sum(r['n'] for r in rows)==120
    assert all(r['false_positives']==20 and r['false_negatives']==0 for r in rows)
    assert all(r['error']==.5 for r in rows)

def test_small_and_single_class_groups_suppressed():
    train=pd.DataFrame({'clicks_84':[0,0]})
    test=pd.DataFrame({'clicks_84':[0]*40,'code_module':['A']*40,'target_risk':[0]*40})
    result=subgroup_audit(train,test,{'selected':np.zeros(40)},84)
    assert result['activity_boundary']==0
    assert all(r['limited'] and r['ap'] is None and r['false_positives'] is None for r in result['rows'])
