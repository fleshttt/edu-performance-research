"""Tiny GENERATED OULAD-like CSV set for schema/anti-leakage testing ONLY.

Not a replacement for the actual Open University anonymised records.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def make_fixture(destination: str|Path, *, n_per_group:int=48, seed:int=42):
    path=Path(destination);path.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    presentations=('2013B','2013J','2014B','2014J')
    modules=('AAA','BBB','CCC')
    students=[];reg=[];courses=[];assessments=[];student_assessment=[];vle=[]
    uid=10000
    aid=100
    for term in presentations:
        for module in modules:
            courses.append({'code_module':module,'code_presentation':term,'module_presentation_length':240})
            ids=[]
            for _ in range(n_per_group):
                uid+=1
                sid=uid
                ids.append(sid)
                latent=rng.normal() + (0.3 if term=='2014J' else 0)
                risky=(latent + rng.normal(scale=.85) < 0)
                result=(rng.choice(['Fail','Withdrawn']) if risky else rng.choice(['Pass','Distinction']))
                students.append(dict(code_module=module,code_presentation=term,id_student=sid,
                    final_result=result,gender='?',region='?',highest_education='?',imd_band='?',
                    age_band='?',num_of_prev_attempts=int(rng.random()<.12),
                    studied_credits=int(rng.choice([30,60,90])),disability='?'))
                reg.append(dict(code_module=module,code_presentation=term,id_student=sid,
                    date_registration=-int(rng.integers(10,90)),date_unregistration='?'))
                for day in range(0,85):
                    if rng.random() < (0.6 if latent>0 else .27):
                        vle.append(dict(code_module=module,code_presentation=term,
                                        id_student=sid,id_site=1,date=day,
                                        sum_click=int(rng.integers(1,5))))
            for due in (21,42,63,84,110):
                aid+=1
                assessments.append(dict(code_module=module,code_presentation=term,
                    id_assessment=aid,assessment_type='TMA',date=due,weight=20))
                for sid in ids:
                    if rng.random() < .86:
                        student_assessment.append(dict(id_assessment=aid,id_student=sid,
                            date_submitted=due-int(rng.choice([0,1,2])),is_banked=0,
                            score=int(np.clip(rng.normal(72,12),0,100))))
    files={'studentInfo.csv':students,'studentRegistration.csv':reg,'courses.csv':courses,
           'assessments.csv':assessments,'studentAssessment.csv':student_assessment,
           'studentVle.csv':vle,'vle.csv':[dict(id_site=1,code_module=m,
           code_presentation=p,activity_type='resource',week_from=0,week_to=35)
           for p in presentations for m in modules]}
    for file,rows in files.items():pd.DataFrame(rows).to_csv(path/file,index=False)
    (path / 'edurisk-source.json').write_text(json.dumps({'kind':'synthetic_fixture','generator':'edurisk.oulad_fixture','seed':seed}),encoding='utf-8')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/oulad/fixture');p.add_argument('--n',type=int,default=48)
    a=p.parse_args();print(make_fixture(a.out,n_per_group=a.n))
