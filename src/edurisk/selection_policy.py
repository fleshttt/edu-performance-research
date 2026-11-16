"""v0.7: preregistered stability-aware policy comparisons on inner time folds.

Selection uses ONLY validation predictions and labels, never the outer test.
Fold AP standard error is a heuristic across a *small number of correlated
rolling temporal splits*, NOT a calibrated confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

METHOD_ORDER = ('fixed_val', 'uniform_ensemble', 'weighted_ensemble', 'adaptive_gate')
POLICIES = ('selected_fold_mean', 'selected_stability_1',
            'selected_stability_2', 'selected_one_se')


@dataclass(frozen=True)
class FoldStability:
    method: str
    ap_by_fold: tuple[float, ...]
    mean_ap: float
    sd_ap: float
    se_ap: float
    mean_brier: float
    min_ap: float


def estimate_fold_stability(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                            fold_sizes: Sequence[int]) -> dict[str, FoldStability]:
    """Estimate descriptive fold-wise AP/variability without independent-fold claims."""
    y = np.asarray(y, dtype=int)
    sizes = [int(n) for n in fold_sizes]
    if len(sizes) < 2 or any(n < 1 for n in sizes) or sum(sizes) != len(y):
        raise ValueError('Need at least two nonempty time folds covering OOF targets')
    boundaries = np.cumsum([0] + sizes)
    output: dict[str, FoldStability] = {}
    for method in METHOD_ORDER:
        if method not in predictions:
            raise ValueError(f'Missing strategy: {method}')
        p = np.asarray(predictions[method], dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probability vector')
        ap_values: list[float] = []
        brier_values: list[float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            yy, pp = y[start:end], p[start:end]
            if np.unique(yy).size != 2:
                raise ValueError('Every temporal validation fold needs both target classes')
            ap_values.append(float(average_precision_score(yy, pp)))
            brier_values.append(float(brier_score_loss(yy, pp)))
        sd = float(np.std(ap_values, ddof=1))
        output[method] = FoldStability(method, tuple(ap_values),
            float(np.mean(ap_values)), sd, float(sd / np.sqrt(len(sizes))),
            float(np.mean(brier_values)), min(ap_values))
    return output


def choose_stability_policies(scores: Mapping[str, FoldStability]) -> dict[str, str]:
    """Fixed, predeclared selectors: no lambda tuning against outer holdout."""
    if set(scores) != set(METHOD_ORDER):
        raise ValueError('Exactly four strategies expected')
    def best_penalized(lam: float) -> str:
        return min(METHOD_ORDER, key=lambda m: (
            -(scores[m].mean_ap - lam * scores[m].se_ap),
            scores[m].mean_brier, METHOD_ORDER.index(m)))
    mean_best = best_penalized(0)
    # Classical one-SE simplicity heuristic. Applied to time-fold AP; this
    # is a comparator, not a statistical test or a formal CI.
    limit = scores[mean_best].mean_ap - scores[mean_best].se_ap
    eligible = [m for m in METHOD_ORDER if scores[m].mean_ap >= limit - 1e-12]
    return {
        'selected_fold_mean': mean_best,
        'selected_stability_1': best_penalized(1.0),
        'selected_stability_2': best_penalized(2.0),
        'selected_one_se': eligible[0],
    }
