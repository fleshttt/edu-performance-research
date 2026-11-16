"""EduRisk v0.8: an independently runnable OULAD external evaluation.

NOT compatible with semester indices used by the synthetic simulator: windows
here are numbers of calendar days prior to a *fixed* day-84 prediction.

Never export identified/pseudonymous individual predictions. No test label is
consulted during model, window, method, or calibration selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_FILES = ("studentInfo.csv", "studentAssessment.csv", "assessments.csv",
                  "studentRegistration.csv", "studentVle.csv", "courses.csv", "vle.csv")
REQUIRED_COLUMNS = {
    "studentInfo.csv": {"code_module", "code_presentation", "id_student", "final_result",
                        "studied_credits", "num_of_prev_attempts"},
    "assessments.csv": {"code_module", "code_presentation", "id_assessment", "date", "assessment_type"},
    "studentAssessment.csv": {"id_assessment", "id_student", "date_submitted", "score"},
    "studentRegistration.csv": {"code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"},
    "studentVle.csv": {"code_module", "code_presentation", "id_student", "date", "sum_click"},
    "courses.csv": {"code_module", "code_presentation"},
    "vle.csv": {"code_module", "code_presentation", "id_site"},
}
KEYS = ["code_module", "code_presentation", "id_student"]
WINDOWS = (14, 28, 56, 84)
DAY = 84
FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting", "extra_trees", "knn", "gaussian_nb")
METHODS = ("fixed_val", "adaptive_gate", "uniform_ensemble", "weighted_ensemble")


def validate_files(directory: Path) -> dict:
    """Only metadata and hashes; no record values sent into the final report."""
    checks = {}
    for filename in REQUIRED_FILES:
        p = directory / filename
        if not p.is_file():
            raise FileNotFoundError(f"Missing OULAD file: {p}")
        cols = set(pd.read_csv(p, nrows=0).columns)
        missing = REQUIRED_COLUMNS[filename] - cols
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):
                h.update(chunk)
        checks[filename] = {"sha256":h.hexdigest(), "bytes":p.stat().st_size}
    return checks



def source_provenance(directory: Path, checks: dict, fixture: bool) -> dict:
    marker = directory / 'edurisk-source.json'
    marked = False
    if marker.exists():
        marked = json.loads(marker.read_text(encoding='utf-8')).get('kind') == 'synthetic_fixture'
    known_fixture = checks['studentInfo.csv']['sha256'] == '0de9701864ab5527ab7914a0ab22c17c1bbae7287bead73517e5797e7b78a381'
    if (marked or known_fixture) and not fixture:
        raise ValueError('Synthetic fixture detected: use --fixture; cannot label it as real OULAD')
    if marker.exists() and not fixture:
        receipt = json.loads(marker.read_text(encoding='utf-8'))
        if receipt.get('kind') == 'uci_download':
            from .oulad_download import URL
            if receipt.get('source_url') != URL or receipt.get('files') != checks:
                raise ValueError('Downloaded OULAD files do not match acquisition receipt')
            return {'kind':'uci_download', 'verified_against_official_source':False,
                    'matches_download_receipt':True, 'source_url':URL,
                    'retrieved_at':receipt['retrieved_at'],
                    'archive_sha256':receipt['archive_sha256'],
                    'citation':receipt['citation'], 'license':receipt['license'],
                    'description':'Downloaded directly from UCI over HTTPS; file hashes match local acquisition receipt. No independent signed checksum is supplied.'}
    return {'kind': 'synthetic_fixture' if fixture else 'user_supplied_oulad',
            'verified_against_official_source': False,
            'description': 'Generated schema test' if fixture else 'Origin declared by experimenter; hashes identify files, not authenticity'}


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def _window_stats(frame: pd.DataFrame, w: int, cutoff: int, key: str,
                  value: str, name: str) -> pd.DataFrame:
    recent=frame.loc[(frame[key] >= cutoff-w+1) & (frame[key] <= cutoff)].copy()
    if recent.empty:
        return pd.DataFrame(columns=KEYS+[f'{name}_{w}_sum',f'{name}_{w}_count'])
    a=recent.groupby(KEYS,as_index=False)[value].agg(['sum','count']).reset_index()
    return a.rename(columns={'sum':f'{name}_{w}_sum', 'count':f'{name}_{w}_count'})


def build_features(raw_dir: str | Path, *, cutoff: int=DAY,
                   windows: tuple[int,...]=WINDOWS, chunksize: int=350_000) -> pd.DataFrame:
    """Create one row per enrolment; only use events known by day `cutoff`.

    Assignment scores are conservatively used only if both submission and
    nominal due date are <= cutoff. The real grade-publication time is unknown.
    `date_unregistration` and `final_result` are NEVER features.
    """
    root=Path(raw_dir)
    validate_files(root)
    if not windows or min(windows)<1 or max(windows)>cutoff:
        raise ValueError('Windows must be positive and <= cutoff')
    info=pd.read_csv(root/'studentInfo.csv', usecols=sorted(REQUIRED_COLUMNS['studentInfo.csv']))
    info['final_result']=info['final_result'].astype(str).str.strip()
    if not info['final_result'].isin(['Fail','Withdrawn','Pass','Distinction']).all():
        raise ValueError('Unknown final_result categories')
    if info.duplicated(KEYS).any():
        raise ValueError('Duplicate enrolment keys')
    info['target_risk']=info['final_result'].isin(['Fail','Withdrawn']).astype(int)
    info=info.drop(columns='final_result')
    for col in ('studied_credits','num_of_prev_attempts'):
        info[col]=_numeric(info[col])
    reg=pd.read_csv(root/'studentRegistration.csv',usecols=KEYS+['date_registration','date_unregistration'])
    reg['date_registration']=_numeric(reg['date_registration'])
    reg['date_unregistration']=_numeric(reg['date_unregistration'])
    base=info.merge(reg,on=KEYS,how='left',validate='one_to_one')
    # A prediction on day cutoff is not meaningful if the learner already
    # withdrew, or had not yet registered, by that day. Exclude, do not
    # leak the withdrawal timestamp as a feature.
    eligible=((base['date_registration'].isna()) | (base['date_registration']<=cutoff)) & \
             ((base['date_unregistration'].isna()) | (base['date_unregistration']>cutoff))
    base=base.loc[eligible].drop(columns='date_unregistration').copy()

    a=pd.read_csv(root/'assessments.csv')
    a=a.loc[a['assessment_type']!='Exam',KEYS[:2]+['id_assessment','date']].copy()
    a['date']=_numeric(a['date'])
    sa=pd.read_csv(root/'studentAssessment.csv')
    sa['date_submitted']=_numeric(sa['date_submitted'])
    sa['score']=_numeric(sa['score'])
    sa=sa.merge(a,on='id_assessment',how='inner',validate='many_to_one')
    sa=sa.loc[(sa['date']<=cutoff) & (sa['date_submitted']<=cutoff) &
              sa['date'].notna() & sa['date_submitted'].notna()].copy()
    # In a real learning system, grades may only be published after due date;
    # this dataset lacks publication timestamps, so there remains uncertainty.
    for w in windows:
        sub=sa.loc[(sa['date']>=cutoff-w+1) & (sa['date']<=cutoff)].copy()
        if not sub.empty:
            agg=sub.groupby(KEYS)['score'].agg(['count','mean','std']).reset_index()
            agg=agg.rename(columns={'count':f'assessed_{w}', 'mean':f'score_mean_{w}',
                                    'std':f'score_std_{w}'})
            base=base.merge(agg,on=KEYS,how='left',validate='one_to_one')
        else:
            for field in (f'assessed_{w}', f'score_mean_{w}', f'score_std_{w}'):
                base[field]=np.nan
    # Stream the large studentVle.csv (~10.6M rows) in bounded chunks.
    # Use count of interaction rows rather than unique activity days: day can
    # straddle chunks, making naive per-chunk nunique aggregation incorrect.
    accumulator={int(w):[] for w in windows}
    for block in pd.read_csv(root/'studentVle.csv',chunksize=chunksize,
                             usecols=KEYS+['date','sum_click']):
        block['date']=_numeric(block['date'])
        block['sum_click']=_numeric(block['sum_click'])
        block=block.loc[(block['date']>=0)&(block['date']<=cutoff)&
                        block['sum_click'].notna()&(block['sum_click']>=0)]
        for w in windows:
            recent=block.loc[block['date']>=cutoff-w+1]
            if not recent.empty:
                groups=recent.groupby(KEYS)['sum_click'].agg(['sum','size']).reset_index()
                accumulator[int(w)].append(groups)
    for w in windows:
        if accumulator[int(w)]:
            assembled=pd.concat(accumulator[int(w)],ignore_index=True)
            ag=assembled.groupby(KEYS,as_index=False)[['sum','size']].sum()
            ag=ag.rename(columns={'sum':f'clicks_{w}', 'size':f'vle_rows_{w}'})
            base=base.merge(ag,on=KEYS,how='left',validate='one_to_one')
        else:
            base[f'clicks_{w}']=0.
            base[f'vle_rows_{w}']=0.
        for key in (f'clicks_{w}', f'vle_rows_{w}', f'assessed_{w}'):
            base[key]=base[key].fillna(0.)
    return base


def split_no_overlap(frame:pd.DataFrame) -> dict[str,pd.DataFrame]:
    """2013 train, 2014B validation, 2014J unseen presentation holdout.

    Individual identifiers are used ONLY to ensure that the same student
    does not occur in multiple partitions, never as predictive features.
    """
    parts={
        'train':frame[frame['code_presentation'].isin(['2013B','2013J'])].copy(),
        'val':frame[frame['code_presentation']=='2014B'].copy(),
        'test':frame[frame['code_presentation']=='2014J'].copy(),
    }
    if any(x.empty for x in parts.values()):
        raise ValueError('Need 2013(B/J), 2014B, and 2014J presentations')
    # Prioritize the held-out cohort. Prevent an individual participating in
    # both test and train/val, or val and train.
    test_ids=set(parts['test'].id_student)
    val_ids=set(parts['val'].id_student)-test_ids
    parts['val']=parts['val'][parts['val'].id_student.isin(val_ids)].copy()
    parts['train']=parts['train'][~parts['train'].id_student.isin(test_ids|val_ids)].copy()
    if any(x.empty for x in parts.values()):
        raise ValueError('Empty partition after excluding overlapping IDs')
    if any(parts[k].target_risk.nunique()!=2 for k in parts):
        raise ValueError('Each partition requires both outcome classes')
    assert not(set(parts['train'].id_student)&set(parts['val'].id_student))
    assert not(set(parts['test'].id_student)&(set(parts['train'].id_student)|set(parts['val'].id_student)))
    return parts


def _columns(w:int) -> list[str]:
    return ['code_module','studied_credits','num_of_prev_attempts','date_registration',
            f'clicks_{w}',f'vle_rows_{w}',f'assessed_{w}',f'score_mean_{w}',f'score_std_{w}']


def _model(family:str,seed:int):
    if family not in FAMILIES:
        raise ValueError(family)
    prep=ColumnTransformer([
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['code_module']),
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0.,add_indicator=True,keep_empty_features=True)),
                             ('scale',StandardScaler())]),['studied_credits','num_of_prev_attempts',
                               'date_registration','clicks','vle_rows','assessed','score_mean','score_std'])
    ])
    clf={
        'logistic_regression':lambda:LogisticRegression(max_iter=700,random_state=seed),
        'random_forest':lambda:RandomForestClassifier(n_estimators=100,min_samples_leaf=5,random_state=seed,n_jobs=-1),
        'hist_gradient_boosting':lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=seed),
        'extra_trees':lambda:ExtraTreesClassifier(n_estimators=150,min_samples_leaf=5,random_state=seed,n_jobs=-1),
        'knn':lambda:KNeighborsClassifier(n_neighbors=25,weights='distance',n_jobs=-1),
        'gaussian_nb':lambda:GaussianNB(var_smoothing=1e-7),
    }[family]()
    return Pipeline([('prep',prep),('clf',clf)])


def _as_model_frame(frame:pd.DataFrame,w:int)->pd.DataFrame:
    cols=_columns(w)
    x=frame[cols].copy()
    return x.rename(columns={f'clicks_{w}':'clicks',f'vle_rows_{w}':'vle_rows',
        f'assessed_{w}':'assessed',f'score_mean_{w}':'score_mean',f'score_std_{w}':'score_std'})


def _ap(y,p)->float:
    return float(average_precision_score(y,p))


def _metrics(y,p)->dict:
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    buckets=np.minimum((p*10).astype(int),9)
    ece=sum(float(np.mean(buckets==b))*abs(float(y[buckets==b].mean())-float(p[buckets==b].mean()))
            for b in range(10) if np.any(buckets==b))
    return {'n':len(y),'prevalence':round(float(y.mean()),5),
            'ap':round(_ap(y,p),5),'roc_auc':round(float(roc_auc_score(y,p)),5),
            'brier':round(float(brier_score_loss(y,p)),5),'ece10':round(ece,5)}


def _adaptive_indices(frame:pd.DataFrame, windows:tuple[int,...]) -> np.ndarray:
    """Prespecified unsupervised gate. Never accesses outcome labels."""
    quality=[]
    for w in windows:
        clicks=np.asarray(frame[f'clicks_{w}'],float)
        rows=np.asarray(frame[f'vle_rows_{w}'],float)
        assessed=np.asarray(frame[f'assessed_{w}'],float)
        # Normalize activity to the window duration to avoid automatic selection
        # of the longest window; reward observed assessments, penalize complexity.
        quality.append(np.log1p(clicks/float(w))+.8*np.log1p(rows/float(w))+
                       .35*np.log1p(assessed)-.15*(w/min(windows)-1))
    return np.argmax(np.vstack(quality),axis=0)


def _candidate_probs(part:pd.DataFrame,pred:dict[int,np.ndarray],
                     windows:tuple[int,...],weights:np.ndarray) -> dict[str,np.ndarray]:
    matrix=np.vstack([pred[w] for w in windows]).T
    adaptive=matrix[np.arange(len(part)),_adaptive_indices(part,windows)]
    return {'adaptive_gate':adaptive,'uniform_ensemble':matrix.mean(axis=1),
            'weighted_ensemble':np.average(matrix,axis=1,weights=weights)}


def _group_stability(val:pd.DataFrame,preds:dict[str,np.ndarray],
                     lam:float=1.) -> dict:
    y=val.target_risk.to_numpy(dtype=int)
    groups=val.code_module.astype(str).to_numpy()
    result={}
    for method,p in preds.items():
        fold=[]
        for g in sorted(set(groups)):
            mask=groups==g
            if len(set(y[mask]))==2 and mask.sum()>=5:
                fold.append(_ap(y[mask],p[mask]))
        avg=float(np.mean(fold)) if fold else _ap(y,p)
        se=float(np.std(fold,ddof=1)/np.sqrt(len(fold))) if len(fold)>=2 else 0.
        result[method]={'mean_ap':round(avg,6),'descriptive_se':round(se,6),
                        'n_modules_with_both_classes':len(fold),'score':round(avg-lam*se,6)}
    return result


def _bootstrap_delta(y,p_new,p_base,ids,reps=600,seed=2026) -> dict:
    # Paired cluster bootstrap: a student may appear multiple times in test.
    rng=np.random.default_rng(seed)
    ids=np.asarray(ids)
    unique,inverse=np.unique(ids,return_inverse=True)
    blocks=[np.flatnonzero(inverse==k) for k in range(len(unique))]
    deltas=[]
    for _ in range(reps):
        choice=rng.integers(0,len(unique),size=len(unique))
        take=np.concatenate([blocks[k] for k in choice])
        yy=np.asarray(y)[take]
        if len(np.unique(yy))<2:continue
        deltas.append(_ap(yy,np.asarray(p_new)[take])-_ap(yy,np.asarray(p_base)[take]))
    return {'delta_ap':round(_ap(y,p_new)-_ap(y,p_base),5),
            'ci95':[round(float(v),5) for v in np.quantile(deltas,[.025,.975])]
            if deltas else None,'resamples':len(deltas),
            'unit':'id_student in held-out cohort'}


def _platt_from_validation(yval,pval,ptest) -> np.ndarray:
    """Platt-sigmoid probability calibration; fit ONLY on validation labels."""
    eps=1e-5
    logit=lambda x:np.log(np.clip(x,eps,1-eps)/(1-np.clip(x,eps,1-eps)))
    calibrator=LogisticRegression(max_iter=250)
    calibrator.fit(logit(pval).reshape(-1,1),np.asarray(yval,dtype=int))
    return calibrator.predict_proba(logit(ptest).reshape(-1,1))[:,1]


def subgroup_audit(train, test, predictions, window):
    """Descriptive holdout audit; activity boundaries use training features only."""
    positive = train.loc[train[f'clicks_{window}'] > 0, f'clicks_{window}']
    boundary = float(positive.median()) if len(positive) else 0.0
    activity = test[f'clicks_{window}'].to_numpy()
    groups = [('course', str(g), test.code_module.astype(str).to_numpy() == str(g))
              for g in sorted(test.code_module.unique())]
    groups += [('activity', 'no_activity', activity == 0),
               ('activity', 'lower_activity', (activity > 0) & (activity <= boundary)),
               ('activity', 'higher_activity', activity > boundary)]
    y = test.target_risk.to_numpy(dtype=int)
    rows = []
    for dimension, group, mask in groups:
        n = int(mask.sum())
        if not n:
            continue
        yy = y[mask]
        # Small cells carry counts only, never unstable headline metrics.
        limited = n < 30 or min(int(yy.sum()), int((1-yy).sum())) < 5
        for method, prob in predictions.items():
            p = np.asarray(prob)[mask]
            row = dict(dimension=dimension, group=group, method=method, n=n,
                       limited=limited, prevalence=None, ap=None, brier=None,
                       false_positives=None, false_negatives=None, error=None)
            if not limited:
                fp = int(((p >= .5) & (yy == 0)).sum())
                fn = int(((p < .5) & (yy == 1)).sum())
                row.update(prevalence=round(float(yy.mean()),5), ap=round(_ap(yy,p),5),
                           brier=round(float(np.mean((p-yy)**2)),5),
                           false_positives=fp, false_negatives=fn,
                           error=round((fp+fn)/n,5))
            rows.append(row)
    return dict(exploratory=True, threshold=.5, activity_window_days=window,
                activity_boundary=boundary, boundary_source='positive training clicks median',
                small_group_rule='n < 30 or either class < 5: metrics suppressed', rows=rows)


def split_calibration(frame, seed=2026):
    """Stable student-level split independent of row order and outcomes."""
    ids=sorted(frame.id_student.unique(), key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    selected=set(ids[:len(ids)//2])
    mask=frame.id_student.isin(selected)
    selection,calibration=frame.loc[mask].copy(),frame.loc[~mask].copy()
    if any(part.target_risk.nunique()!=2 for part in (selection,calibration)):
        raise ValueError('Selection and calibration partitions must each contain both classes')
    return selection,calibration


def run_external(raw_dir:str|Path,out_dir:str|Path, *, family='logistic_regression',
                 windows:tuple[int,...]=WINDOWS,cutoff:int=DAY, seed:int=2026,
                 fixture:bool=False, bootstrap_reps:int=600, separate_calibration:bool=False)->dict:
    """Train/val decisions are completed before the holdout labels are accessed."""
    if bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be positive')
    if len(set(windows)) != len(windows) or tuple(sorted(windows)) != tuple(windows):
        raise ValueError('Windows must be unique and sorted ascending')
    source=validate_files(Path(raw_dir))
    provenance=source_provenance(Path(raw_dir), source, fixture)
    features=build_features(raw_dir,cutoff=cutoff,windows=windows)
    splits=split_no_overlap(features)
    tr,va,te=(splits[k] for k in ('train','val','test'))
    va,ca=split_calibration(va,seed) if separate_calibration else (va,va)
    ytrain=tr.target_risk.to_numpy(dtype=int)
    yval=va.target_risk.to_numpy(dtype=int)
    # --- PRE-HOLDOUT: only train and validation outcomes can be read here.
    val_window={}; test_window={}; cal_window={}
    for w in windows:
        model=_model(family,seed)
        model.fit(_as_model_frame(tr,w),ytrain)
        val_window[w]=model.predict_proba(_as_model_frame(va,w))[:,1]
        test_window[w]=model.predict_proba(_as_model_frame(te,w))[:,1]
        cal_window[w]=model.predict_proba(_as_model_frame(ca,w))[:,1]
    ap_val={w:_ap(yval,val_window[w]) for w in windows}
    best_w=max(windows,key=lambda w:(ap_val[w],-w))
    weights=np.array([max(ap_val[w]-yval.mean(),.001) for w in windows],float)
    val_pred={'fixed_val':val_window[best_w],
              **_candidate_probs(va,val_window,windows,weights)}
    test_pred={'fixed_val':test_window[best_w],
               **_candidate_probs(te,test_window,windows,weights)}
    stability=_group_stability(va,val_pred,lam=1.)
    method=max(METHODS,key=lambda m:(stability[m]['score'],-METHODS.index(m)))
    # Fit calibration with validation only. Test outcomes are not accessed.
    cal_pred={'fixed_val':cal_window[best_w],**_candidate_probs(ca,cal_window,windows,weights)}
    calibrated=_platt_from_validation(ca.target_risk.to_numpy(dtype=int),cal_pred[method],test_pred[method])
    # --- FROZEN METHOD, FIRST TEST METRIC ACCESS ---
    ytest=te.target_risk.to_numpy(dtype=int)
    methods={m:_metrics(ytest,p) for m,p in test_pred.items()}
    methods['selected_calibrated']=_metrics(ytest,calibrated)
    boot=_bootstrap_delta(ytest,test_pred[method],test_pred['fixed_val'],
                          te.id_student.to_numpy(),reps=bootstrap_reps,seed=seed)
    selective=[]
    for margin in (0.,.1,.2,.3):
        keep=np.abs(calibrated-.5)>=margin
        if int(keep.sum())==0:continue
        yy=ytest[keep]
        selective.append({'margin':margin,'coverage':round(float(keep.mean()),4),
                          'n':int(keep.sum()),
                          'classification_error':round(float(np.mean((calibrated[keep]>=.5)!=yy)),5),
                          'ap':round(_ap(yy,calibrated[keep]),5) if np.unique(yy).size==2 else None})
    result={
      'mode':'SYNTHETIC_SCHEMA_FIXTURE' if fixture else 'REAL_OULAD_EXTERNAL',
      'version':'0.8.0','source':'Generated OULAD-like fixture' if fixture else 'Local files declared as OULAD by experimenter',
      'provenance':provenance,
      'environment':{'python':platform.python_version(),'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**{p:version(p) for p in ('numpy','pandas','scikit-learn')}},
      'design':{'target':'Fail/Withdrawn versus Pass/Distinction (enrolled at cutoff)','cutoff_day':cutoff,
                'calibration_protocol':'separate' if separate_calibration else 'shared',
                'windows_days':list(windows),'family':family,
                'train_presentations':['2013B','2013J'],
                'validation_presentation':'2014B','holdout_presentation':'2014J',
                'duplicate_students_excluded_across_splits':True,
                'selection':'module-level validation mean AP minus 1 descriptive SE'},
      'n_rows':{k:int(len(v)) for k,v in splits.items()},
      'cohort_audit':{'eligible_enrolments':len(features),'split_enrolments':sum(len(v) for v in splits.values()),'unique_students':{k:int(v.id_student.nunique()) for k,v in splits.items()}},
      'calibration_audit':{'selection_rows':len(va),'calibration_rows':len(ca),
                           'selection_students':int(va.id_student.nunique()),'calibration_students':int(ca.id_student.nunique()),
                           'overlap_students':len(set(va.id_student)&set(ca.id_student)),
                           'split_seed':seed,'split_rule':'SHA256 student ordering half split' if separate_calibration else 'shared validation'},
      'classes':{k:round(float(v.target_risk.mean()),5) for k,v in splits.items()},
      'selection':{'fixed_window_days':int(best_w),'selected_method':method,
                   'validation_ap_by_window':{str(w):round(ap_val[w],5) for w in windows},
                   'validation_method_stability':stability},
      'metrics':methods,
      'comparison':{'selected_minus_fixed_ap':round(methods[method]['ap']-methods['fixed_val']['ap'],5)},
      'bootstrap':boot,'selective':selective,
      'subgroup_audit':subgroup_audit(tr, te, {'fixed_val':test_pred['fixed_val'],
                         'selected':test_pred[method], 'selected_calibrated':calibrated}, max(windows)),
      'source_file_hashes':source,
      'limitations':['OULAD is course-weekly data, not multi-semester TUSUR trajectories',
                     'External holdout is 2014J; temporal and institutional transport is not established',
                     'The dataset does not provide the date grades were released; due-date filtering is only a proxy',
                     'Observed non-submission, no VLE activity, and missing data have different meanings',
                     'Students unregistered by day 84 are excluded; the target is conditional on still being enrolled',
                     'Validation modules are few/dependent; their AP SE is only a descriptive heuristic',
                     'Separate selection and calibration students within 2014B; reduced selection sample' if separate_calibration else 'Platt calibration is fitted on the same validation pool used for method selection',
                     '2014J has already been inspected; this comparison is exploratory, not new independent validation',
                     'The 95% student-cluster bootstrap interval is conditional on this held-out 2014J cohort',
                     'No demographic features, fairness audit, or intervention effects assessed',
                     'Never use predictions for student sanctions or automated consequential decisions'],
    }
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'external_oulad_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    # Save only aggregated tables, not individual risk probabilities.
    pd.DataFrame([{'method':k,**v} for k,v in methods.items()]).to_csv(out/'external_oulad_aggregates.csv',index=False)
    return result


def main():
    p=argparse.ArgumentParser(description='External evaluation: local OULAD CSVs, no download')
    p.add_argument('--raw',default='data/oulad/raw')
    p.add_argument('--out',default='reports')
    p.add_argument('--family',choices=FAMILIES,default='logistic_regression')
    p.add_argument('--cutoff',type=int,default=DAY)
    p.add_argument('--windows',default='14,28,56,84')
    p.add_argument('--bootstrap',type=int,default=600)
    p.add_argument('--separate-calibration',action='store_true')
    p.add_argument('--fixture',action='store_true',help='Use ONLY with generated fake CSV schema fixture')
    a=p.parse_args()
    result=run_external(a.raw,a.out,family=a.family,cutoff=a.cutoff,
        windows=tuple(map(int,a.windows.split(','))),fixture=a.fixture,bootstrap_reps=a.bootstrap,separate_calibration=a.separate_calibration)
    print(json.dumps({'mode':result['mode'],'n_rows':result['n_rows'],
                      'selection':result['selection']['selected_method'],
                      'metrics':result['metrics']['selected_calibrated']},ensure_ascii=False))

if __name__=='__main__':
    main()
