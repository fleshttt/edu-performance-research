"""EduRisk v0.6: repeated nested temporal validation with a method selector.

Research demonstration on SYNTHETIC data only. The inner rolling-origin
validations choose a history window, gate, ensemble weights, and forecasting
method. The final term 9 labels are *never* read before selection/refitting.
Repeated synthetic cohorts support paired, descriptive bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ablation import TARGET, MEASURES, _model, evaluate, validate_experiment_data, window_features
from .adaptive import GATE_GRID, GateParameters, gate_statistics, select_windows, combine_predictions
from .generate_data import generate_synthetic
from .robustness import inject_structured_missingness, validation_softmax_weights, _average
from .selection_policy import estimate_fold_stability, choose_stability_policies, POLICIES

VERSION = '0.6.0'
WINDOWS = (1, 2, 3, 4, 5)
INNER_VALIDATION_TERMS = (6, 7, 8)
OUTER_TEST_TERM = 9
TRAIN_START_TERM = 4
SCENARIOS = (('complete', 0.0), ('mar_attendance', 0.25),
             ('mnar_value', 0.25), ('contiguous_terms', 0.25))
FAMILIES = ('logistic_regression', 'random_forest')
METHODS = ('fixed_val', 'adaptive_gate', 'uniform_ensemble', 'weighted_ensemble')
BASELINES = ('snapshot_1', 'longest')


def temporal_masks(frame: pd.DataFrame, *, inner_terms: tuple[int, ...] = INNER_VALIDATION_TERMS,
                   test_term: int = OUTER_TEST_TERM, train_start: int = TRAIN_START_TERM) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Strict nested time order; all inner validation rows precede the untouched holdout."""
    if not inner_terms or tuple(sorted(set(inner_terms))) != tuple(inner_terms):
        raise ValueError('Inner terms must be unique and increasing')
    if min(inner_terms) <= train_start or max(inner_terms) >= test_term:
        raise ValueError('Invalid inner/outer chronology')
    term = frame['term_index'].to_numpy()
    inner = []
    for val_term in inner_terms:
        training = (term >= train_start) & (term < val_term)
        validating = term == val_term
        if not training.any() or not validating.any():
            raise ValueError(f'Missing training or validation term {val_term}')
        if int(term[training].max()) >= int(term[validating].min()):
            raise ValueError('Validation not strictly later than inner training')
        inner.append((training, validating))
    development = (term >= train_start) & (term < test_term)
    holdout = term == test_term
    if not development.any() or not holdout.any() or int(term[development].max()) >= int(term[holdout].min()):
        raise ValueError('Invalid outer train/holdout split')
    if any(np.logical_and(holdout, a).any() or np.logical_and(holdout, b).any() for a, b in inner):
        raise ValueError('Inner/outer overlap')
    return inner, development, holdout


def rank(metrics: dict, name: str) -> tuple[float, float, str]:
    return (-float(metrics['average_precision']), float(metrics['brier']), name)


def method_predictions(window_preds: dict[int, np.ndarray], gate_choice: np.ndarray,
                       fixed_window: int, weights: dict[int, float]) -> dict[str, np.ndarray]:
    return {
        'snapshot_1': window_preds[min(window_preds)],
        'longest': window_preds[max(window_preds)],
        'fixed_val': window_preds[fixed_window],
        'adaptive_gate': combine_predictions(window_preds, gate_choice),
        'uniform_ensemble': _average(window_preds),
        'weighted_ensemble': _average(window_preds, weights),
    }


@dataclass
class NestedOutcome:
    selection: dict
    rows: list[dict]
    n_fits: int


def nested_evaluate(masked: pd.DataFrame, *, family: str = 'logistic_regression',
                    windows: tuple[int, ...] = WINDOWS,
                    inner_terms: tuple[int, ...] = INNER_VALIDATION_TERMS,
                    gate_grid: tuple[GateParameters, ...] = GATE_GRID,
                    test_term: int = OUTER_TEST_TERM,
                    include_uncertainty: bool = False) -> NestedOutcome:
    """Fit inside rolling validation, select by OOF AP/Brier, refit and score outer holdout.

    This function uses target labels up through `test_term-1` for fitting and
    selection. Test labels are accessed only in the very last scoring block.
    """
    validate_experiment_data(masked)
    if (not windows or tuple(sorted(set(windows))) != windows or
        any(not isinstance(k, int) or k < 1 or k > 8 for k in windows)):
        raise ValueError('Windows must be increasing distinct integers in 1..8')
    if not gate_grid:
        raise ValueError('Need at least one gate candidate')
    inner, development, holdout = temporal_masks(masked, inner_terms=inner_terms, test_term=test_term)
    # Feature engineering is a pure function of observed data at t and prior terms.
    features = {k: window_features(masked, k) for k in windows}
    stats = gate_statistics(masked, windows)
    heldout_positions = np.flatnonzero(holdout)
    predicted_oof = {k: [] for k in windows}
    oof_y, oof_positions = [], []
    oof_fold_sizes = []
    fits = 0

    for train_mask, val_mask in inner:
        labels = masked.loc[train_mask, TARGET]
        if labels.nunique() != 2 or masked.loc[val_mask, TARGET].nunique() != 2:
            raise ValueError('Inner fold requires both classes')
        oof_y.append(masked.loc[val_mask, TARGET].to_numpy(dtype=int))
        oof_fold_sizes.append(int(val_mask.sum()))
        oof_positions.extend(np.flatnonzero(val_mask).tolist())
        for k in windows:
            X, cols = features[k]
            clf = _model(family, cols)
            clf.fit(X.loc[train_mask, cols], labels)
            predicted_oof[k].append(clf.predict_proba(X.loc[val_mask, cols])[:, 1])
            fits += 1

    y_oof = np.concatenate(oof_y)
    index_oof = np.asarray(oof_positions, dtype=int)
    oof = {k: np.concatenate(predicted_oof[k]) for k in windows}
    val_by_window = {k: evaluate(y_oof, oof[k]) for k in windows}
    best_fixed = min(windows, key=lambda k: rank(val_by_window[k], str(k)))
    weights = validation_softmax_weights({k: val_by_window[k]['average_precision'] for k in windows})

    gate_candidates = {}
    for config in gate_grid:
        chosen = select_windows(stats, config, index_oof)
        prediction = combine_predictions(oof, chosen)
        gate_candidates[config.name] = evaluate(y_oof, prediction)
    best_gate = min(gate_grid, key=lambda cfg: rank(gate_candidates[cfg.name], cfg.name))
    oof_choices = select_windows(stats, best_gate, index_oof)
    method_oof_probs = method_predictions(oof, oof_choices, best_fixed, weights)
    method_oof_metrics = {m: evaluate(y_oof, p) for m, p in method_oof_probs.items()}
    best_method = min(METHODS, key=lambda m: rank(method_oof_metrics[m], m))
    policy_choices = {}
    policy_scores = {}
    if include_uncertainty:
        policy_scores = estimate_fold_stability(y_oof, method_oof_probs, oof_fold_sizes)
        policy_choices = choose_stability_policies(policy_scores)


    # Refit only AFTER all selection decisions have been frozen; do not use holdout labels.
    train_y = masked.loc[development, TARGET]
    if train_y.nunique() != 2:
        raise ValueError('Development sample needs both classes')
    test_preds = {}
    for k in windows:
        X, cols = features[k]
        clf = _model(family, cols)
        clf.fit(X.loc[development, cols], train_y)
        test_preds[k] = clf.predict_proba(X.loc[holdout, cols])[:, 1]
        fits += 1
    test_choices = select_windows(stats, best_gate, heldout_positions)
    all_methods = method_predictions(test_preds, test_choices, best_fixed, weights)
    all_methods['selected_nested'] = all_methods[best_method]
    if include_uncertainty:
        for policy, chosen in policy_choices.items():
            all_methods[policy] = all_methods[chosen]

    selection = {
        'family':family, 'chosen_method':best_method,
        'fixed_window':int(best_fixed), 'chosen_gate':best_gate.name,
        'weights':json.dumps(weights, ensure_ascii=False, sort_keys=True),
        'inner_terms':','.join(map(str, inner_terms)), 'outer_test_term':int(test_term),
        'inner_n':int(len(y_oof)), 'development_n':int(development.sum()),
        'test_n':int(holdout.sum()),
        'inner_ap_selected':float(method_oof_metrics[best_method]['average_precision']),
        'inner_brier_selected':float(method_oof_metrics[best_method]['brier']),
        'inner_ap_fixed':float(method_oof_metrics['fixed_val']['average_precision']),
        'inner_ap_adaptive':float(method_oof_metrics['adaptive_gate']['average_precision']),
        'inner_ap_uniform':float(method_oof_metrics['uniform_ensemble']['average_precision']),
        'inner_ap_weighted':float(method_oof_metrics['weighted_ensemble']['average_precision']),
    }
    if include_uncertainty:
        selection['stability_policies'] = dict(policy_choices)
        selection['stability_fold_scores'] = {m: {
            'mean_ap': s.mean_ap, 'sd_ap': s.sd_ap, 'se_ap': s.se_ap,
            'min_ap': s.min_ap, 'mean_brier': s.mean_brier,
            'fold_ap': list(s.ap_by_fold)}
            for m, s in policy_scores.items()}
    # First exposure to held-out targets, after every model and selection is set.
    y_test = masked.loc[holdout, TARGET]
    rows = []
    for method, prob in all_methods.items():
        score = evaluate(y_test, prob)
        rows.append({**{'family':family,'method':method,'chosen_method':best_method},
                     **{f'test_{key}':val for key,val in score.items()},
                     'test_prevalence':float(y_test.mean()),
                     'policy_base_method':policy_choices.get(method, best_method if method=='selected_nested' else method)})
    return NestedOutcome(selection, rows, fits)


def bootstrap_paired_delta(diff: np.ndarray, seed: int = 2026, draws: int = 2000) -> dict:
    """Percentile bootstrap over INDEPENDENT synthetic-generation seeds, not rows.

    These are exploratory Monte-Carlo intervals conditional on a generator;
    they do NOT quantify transport to a real university population.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 1 or not np.isfinite(diff).all() or not len(diff):
        raise ValueError('Nonempty finite paired differences required')
    output = {'mean':float(diff.mean()), 'std':float(diff.std(ddof=1)) if len(diff)>1 else None,
              'n_replicates':int(len(diff)), 'positive_count':int((diff>0).sum())}
    if len(diff)<3:
        output.update({'ci_low':None,'ci_high':None})
        return output
    rng = np.random.default_rng(seed)
    sample_means = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(axis=1)
    output.update({'ci_low':float(np.quantile(sample_means,.025)),
                   'ci_high':float(np.quantile(sample_means,.975))})
    return output


def run_nested_study(output_dir: str | Path, *, seeds: tuple[int, ...] = (103,211,307,419,523,631),
                     n_students: int = 160,
                     scenarios: tuple[tuple[str,float], ...] = SCENARIOS,
                     families: tuple[str,...] = FAMILIES,
                     windows: tuple[int,...] = WINDOWS,
                     inner_terms: tuple[int,...] = INNER_VALIDATION_TERMS,
                     gate_grid: tuple[GateParameters,...] = GATE_GRID) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or n_students < 30:
        raise ValueError('Need distinct seeds and at least 30 synthetic students')
    if not scenarios or not families:
        raise ValueError('Empty experiment dimensions')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selections, metrics, masks = [], [], []
    count_fits = 0
    for seed in seeds:
        original = generate_synthetic(n_students=n_students, n_terms=10, seed=int(seed))
        validate_experiment_data(original)
        for mechanism, rate in scenarios:
            mask_seed = 10000 + int(seed)
            masked = inject_structured_missingness(original, mechanism, rate, mask_seed)
            actual_rate = float(masked[MEASURES].isna().to_numpy().mean())
            masks.append({'replicate_seed':int(seed),'mask_seed':mask_seed,'mechanism':mechanism,
                          'rate':float(rate),'actual_missing_fraction':actual_rate})
            for family in families:
                experiment = nested_evaluate(masked, family=family, windows=windows,
                                             inner_terms=inner_terms, gate_grid=gate_grid)
                count_fits += experiment.n_fits
                group = {'replicate_seed':int(seed), 'mechanism':mechanism,
                         'rate':float(rate), 'family':family}
                selections.append({**group,**experiment.selection})
                for record in experiment.rows:
                    metrics.append({**group,**record})
    selections = pd.DataFrame(selections)
    metrics = pd.DataFrame(metrics)
    masks = pd.DataFrame(masks)
    metrics.to_csv(out / 'nested_runs_synthetic.csv', index=False)
    selections.to_csv(out / 'nested_selections_synthetic.csv', index=False)
    masks.to_csv(out / 'nested_missingness_synthetic.csv', index=False)

    deltas=[]
    for key, group in metrics.groupby(['replicate_seed','mechanism','rate','family'],sort=True):
        reference = group[group.method.eq('fixed_val')].iloc[0]
        for method in ('adaptive_gate','uniform_ensemble','weighted_ensemble','selected_nested'):
            target = group[group.method.eq(method)].iloc[0]
            deltas.append(dict(zip(['replicate_seed','mechanism','rate','family'],key)) | {
                'method':method,
                'delta_ap_vs_fixed':float(target.test_average_precision-reference.test_average_precision),
                'delta_brier_vs_fixed':float(target.test_brier-reference.test_brier)})
    deltas=pd.DataFrame(deltas)
    deltas.to_csv(out / 'nested_paired_deltas_synthetic.csv',index=False)

    summaries=[]
    group_cols = ['mechanism','rate','family','method']
    for key, group in metrics.groupby(group_cols, sort=True):
        pair = deltas[(deltas.mechanism == key[0]) & (deltas.rate == key[1]) &
                      (deltas.family == key[2]) & (deltas.method == key[3])]
        row = dict(zip(group_cols,key))
        row.update({'n_replicates':int(group.replicate_seed.nunique()),
                    'test_ap_mean':float(group.test_average_precision.mean()),
                    'test_ap_sd':float(group.test_average_precision.std(ddof=1)),
                    'test_brier_mean':float(group.test_brier.mean()),
                    'test_roc_auc_mean':float(group.test_roc_auc.mean()),
                    'mean_prevalence':float(group.test_prevalence.mean())})
        if not pair.empty:
            interval=bootstrap_paired_delta(pair.delta_ap_vs_fixed.to_numpy())
            row.update({'delta_ap_vs_fixed':interval['mean'],'delta_ap_ci_low':interval['ci_low'],
                        'delta_ap_ci_high':interval['ci_high'], 'paired_wins':interval['positive_count']})
        else:
            row.update({'delta_ap_vs_fixed':0.0 if key[3]=='fixed_val' else None,
                        'delta_ap_ci_low':None,'delta_ap_ci_high':None,'paired_wins':None})
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    summary.to_csv(out / 'nested_summary_synthetic.csv',index=False)
    distribution=(selections.groupby(['mechanism','family','chosen_method'],sort=True)
                  .size().rename('n_selected').reset_index())
    distribution.to_csv(out/'nested_method_frequencies_synthetic.csv',index=False)
    output={
        'mode':'SYNTHETIC_DEMO_ONLY', 'version':VERSION,
        'protocol':{
            'description':'Repeated synthetic cohorts; inner expanding time validation, untouched outer term 9',
            'replicate_seeds':list(seeds), 'n_students_per_replicate':n_students,
            'n_terms':10, 'inner_train_from_term':TRAIN_START_TERM,
            'inner_validation_terms':list(inner_terms),
            'outer_development_terms':list(range(TRAIN_START_TERM, OUTER_TEST_TERM)),
            'outer_test_term':OUTER_TEST_TERM, 'windows':list(windows),
            'models':list(families), 'scenarios':[{'mechanism':m,'rate':r} for m,r in scenarios],
            'gate_candidates':len(gate_grid),
            'method_candidates':list(METHODS),
            'selection_metric':'pooled inner out-of-fold Average Precision; tie-break Brier then name',
            'ci':'paired seed-level nonparametric bootstrap, 2000 draws, percentile 2.5–97.5',
            'test_policy':'final test labels read only after selecting method and retraining on development data',
        },
        'n_model_fits':int(count_fits),
        'n_seed_replicates':len(seeds),
        'n_experiments':int(len(selections)),
        'mean_actual_missing_by_mechanism':json.loads(masks.groupby('mechanism')['actual_missing_fraction'].mean().rename('mean_missing_fraction').reset_index().to_json(orient='records')),
        'summaries':json.loads(summary.to_json(orient='records')),
        'selection_counts':json.loads(distribution.to_json(orient='records')),
        'limitations':[
            'Only synthetic data; the generator assumptions may favor particular rules.',
            'Bootstrap intervals quantify synthetic-seed Monte-Carlo variation, not real-world generalization.',
            'AP is prevalence-sensitive; compare matched replicates and inspect prevalence.',
            'The inner OOF score is used for method selection and is optimistic as a performance estimate.',
            'Repeated generated cohorts are independent random seeds from the same generator, not external institutions.',
            'MNAR is simulated via latent complete values; its mechanism is not identifiable from observed real-world data.',
            'Forecast probabilities are experimental, not validated for real educational decisions.',
        ],
    }
    (out / 'nested_summary_synthetic.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description='EduRisk v0.6 nested time validation on synthetic seeds')
    parser.add_argument('--output',default='reports')
    parser.add_argument('--n-students',type=int,default=160)
    parser.add_argument('--seeds',default='103,211,307,419,523,631')
    args=parser.parse_args()
    seeds=tuple(int(s) for s in args.seeds.split(',') if s.strip())
    result=run_nested_study(args.output, seeds=seeds, n_students=args.n_students)
    print(json.dumps({'version':result['version'],'n_experiments':result['n_experiments'],
                      'n_model_fits':result['n_model_fits']},ensure_ascii=False))

if __name__=='__main__':
    main()
