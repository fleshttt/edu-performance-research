"""EduRisk v0.7: held-out comparison of stability-aware strategy selectors.

All experiments are synthetic. All candidate selectors share the *same*
trained model predictions: the number of model fits does not increase when
adding a policy. No outer test labels are used in model/policy selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ablation import MEASURES, validate_experiment_data
from .generate_data import generate_synthetic
from .nested_selection import (FAMILIES, SCENARIOS, WINDOWS, INNER_VALIDATION_TERMS,
                               bootstrap_paired_delta, nested_evaluate)
from .robustness import inject_structured_missingness
from .selection_policy import POLICIES

VERSION = '0.7.0'
METHODS_TO_REPORT = ('fixed_val', 'adaptive_gate', 'uniform_ensemble',
                     'weighted_ensemble', 'selected_nested') + POLICIES
SEEDS = (103, 211, 307, 419, 523, 631, 743, 853)


def run_stability_study(output_dir: str | Path, *, seeds: tuple[int, ...] = SEEDS,
                        n_students: int = 120, scenarios=SCENARIOS, families=FAMILIES,
                        windows=WINDOWS, gate_grid=None) -> dict:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or n_students < 35:
        raise ValueError('Require >=3 distinct seeds and >=35 synthetic students')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons, selections, fold_details, masks = [], [], [], []
    fits = 0
    for seed in seeds:
        original = generate_synthetic(n_students=n_students, n_terms=10, seed=int(seed))
        validate_experiment_data(original)
        for mechanism, rate in scenarios:
            masked = inject_structured_missingness(original, mechanism, rate, 10000 + int(seed))
            masks.append({'seed':int(seed),'mechanism':mechanism,'missingness':rate,
                          'actual_missing':float(masked[MEASURES].isna().to_numpy().mean())})
            for family in families:
                kwargs={'family':family,'windows':windows,'include_uncertainty':True}
                if gate_grid is not None:
                    kwargs['gate_grid']=gate_grid
                result=nested_evaluate(masked,**kwargs)
                fits += result.n_fits
                key={'seed':int(seed),'mechanism':mechanism,'missingness':float(rate),'family':family}
                choices=result.selection['stability_policies']
                selections.append({**key, 'selected_v06':result.selection['chosen_method'],
                                   'fixed_window':result.selection['fixed_window'],
                                   **{p:choices[p] for p in POLICIES}})
                for method, score in result.selection['stability_fold_scores'].items():
                    fold_details.append({**key,'method':method,
                        'ap_fold_6':score['fold_ap'][0],
                        'ap_fold_7':score['fold_ap'][1],
                        'ap_fold_8':score['fold_ap'][2],
                        'ap_fold_mean':score['mean_ap'],
                        'ap_fold_se':score['se_ap'],
                        'ap_fold_min':score['min_ap'],
                        'ap_fold_sd':score['sd_ap'],
                        'brier_fold_mean':score['mean_brier']})
                for record in result.rows:
                    if record['method'] not in METHODS_TO_REPORT:
                        continue
                    comparisons.append({**key,**record})
    runs=pd.DataFrame(comparisons)
    choices_df=pd.DataFrame(selections)
    fold_df=pd.DataFrame(fold_details)
    missing_df=pd.DataFrame(masks)
    runs.to_csv(out/'uncertainty_runs_synthetic.csv', index=False)
    choices_df.to_csv(out/'uncertainty_selections_synthetic.csv',index=False)
    fold_df.to_csv(out/'uncertainty_fold_scores_synthetic.csv',index=False)
    missing_df.to_csv(out/'uncertainty_missingness_synthetic.csv',index=False)

    piv = runs.pivot(index=['seed','mechanism','missingness','family'],
                     columns='method',values=['test_average_precision','test_brier'])
    rows=[]
    for (seed,mechanism,rate,family), vals in piv.iterrows():
        anchor_ap = vals['test_average_precision']['fixed_val']
        v06_ap = vals['test_average_precision']['selected_nested']
        for policy in ('selected_nested',) + POLICIES:
            rows.append({'seed':seed,'mechanism':mechanism,'missingness':rate,'family':family,
                'policy':policy,'ap':float(vals['test_average_precision'][policy]),
                'brier':float(vals['test_brier'][policy]),
                'delta_vs_fixed':float(vals['test_average_precision'][policy]-anchor_ap),
                'delta_vs_v06':float(vals['test_average_precision'][policy]-v06_ap)})
    deltas=pd.DataFrame(rows)
    deltas.to_csv(out/'uncertainty_paired_deltas_synthetic.csv',index=False)
    summary=[]
    for (mechanism,family,policy),group in deltas.groupby(['mechanism','family','policy'],sort=True):
        df_run=runs[(runs.mechanism==mechanism)&(runs.family==family)&(runs.method==policy)]
        a=bootstrap_paired_delta(group.delta_vs_fixed.to_numpy(),seed=2027,draws=5000)
        b=bootstrap_paired_delta(group.delta_vs_v06.to_numpy(),seed=2028,draws=5000)
        summary.append({'mechanism':mechanism,'family':family,'policy':policy,
            'replicates':int(group.seed.nunique()),
            'ap_mean':float(group.ap.mean()),
            'ap_sd':float(group.ap.std(ddof=1)),
            'brier_mean':float(group.brier.mean()),
            'delta_fixed_mean':a['mean'],'delta_fixed_ci_lo':a['ci_low'],
            'delta_fixed_ci_hi':a['ci_high'],'positive_vs_fixed':a['positive_count'],
            'delta_v06_mean':b['mean'],'delta_v06_ci_lo':b['ci_low'],
            'delta_v06_ci_hi':b['ci_high'],'positive_vs_v06':b['positive_count'],
            'mean_test_prevalence':float(df_run.test_prevalence.mean())})
    summ=pd.DataFrame(summary)
    summ.to_csv(out/'uncertainty_summary_synthetic.csv',index=False)
    freqs=(choices_df.melt(id_vars=['seed','mechanism','missingness','family','fixed_window'],
                          value_vars=['selected_v06']+list(POLICIES),
                          var_name='selector',value_name='chosen_method')
           .groupby(['mechanism','family','selector','chosen_method'],sort=True)
           .size().rename('count').reset_index())
    freqs.to_csv(out/'uncertainty_policy_frequencies_synthetic.csv',index=False)
    result={'mode':'SYNTHETIC_DEMO_ONLY','version':VERSION,
      'protocol':{'description':'predeclared temporal-fold stability policies; final term is untouched',
        'seed_replicates':list(seeds),'n_students_per_seed':n_students,
        'nested_inner_validation_terms':list(INNER_VALIDATION_TERMS),
        'outer_holdout_term':9,'model_families':list(families),'windows':list(windows),
        'scenarios':[{'name':s,'nominal_rate':r} for s,r in scenarios],
        'policy_candidates':['selected_nested']+list(POLICIES),
        'method_candidates':['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble'],
        'heuristics':'mean fold AP; mean minus one or two fold standard errors; one-SE simplest',
        'fold_SE_warning':'Only 3 correlated time folds; SE is a descriptive variability penalty, not a confidence interval',
        'bootstrap':'5000 paired resamples of independent synthetic seeds; descriptive generator-conditional intervals',
        'test_policy':'outer term 9 labels read only after all selectors are chosen; no retrospective choice on holdout'},
      'n_model_fits':int(fits),'n_scenario_families_seeds':int(len(choices_df)),
      'summaries':json.loads(summ.to_json(orient='records')),
      'frequencies':json.loads(freqs.to_json(orient='records')),
      'missingness':json.loads(missing_df.groupby('mechanism')['actual_missing'].mean().rename('actual_mean').reset_index().to_json(orient='records')),
      'limitations':['Synthetic datasets; no claims about actual learners',
        'The same inner folds are used for gate and selector tuning; evaluation is only unbiased on a fresh outer holdout',
        'Time folds are overlapping/serially dependent: fold SE is NOT a statistical confidence interval',
        'Eight seed replicates do not guarantee stable statistical inference',
        'Bootstrap intervals describe between-seed variation under one generator, not institution transport',
        'The candidate methods and lambdas were chosen before evaluating outer term 9',
        'One held-out time point per synthetic seed; external longitudinal dataset validation is still missing',
        'AP depends on class prevalence; Brier is not a pure calibration score'],
    }
    (out/'uncertainty_summary_synthetic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description='v0.7 uncertainty-aware strategy comparison (synthetic only)')
    parser.add_argument('--output',default='reports')
    parser.add_argument('--seeds',default='103,211,307,419,523,631')
    parser.add_argument('--windows',default='1,3,5')
    parser.add_argument('--n-students',type=int,default=90)
    args=parser.parse_args()
    result=run_stability_study(args.output,
        seeds=tuple(int(x) for x in args.seeds.split(',') if x.strip()),
        n_students=args.n_students,
        windows=tuple(int(x) for x in args.windows.split(',') if x.strip()))
    print(json.dumps({'version':result['version'],'n_experiments':result['n_scenario_families_seeds'],
                      'n_model_fits':result['n_model_fits']},ensure_ascii=False))


if __name__=='__main__':
    main()
