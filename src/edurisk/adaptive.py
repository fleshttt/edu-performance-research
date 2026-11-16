"""v0.4: validation-selected, interpretable per-student history-window gate.

All experiments are SYNTHETIC. Gate inputs are only available predictor measurements
at t,t-1,...,t-k+1; no target labels, no post-t observations. Gate hyperparameters
and the competing fixed window are selected on validation (t=8), not test (t=9).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ablation import (
    MEASURES, PATTERNS, RATES, SEEDS, TRAIN_TERMS, VAL_TERM, TEST_TERM,
    TARGET, WINDOWS, _aligned_lags, _model, evaluate, inject_missingness,
    validate_experiment_data, window_features,
)

FAMILIES = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class GateParameters:
    """Pre-declared gate configurations, estimated only through validation."""
    coverage_weight: float
    volatility_weight: float
    complexity_weight: float

    @property
    def name(self) -> str:
        return (f"c{self.coverage_weight:g}_v{self.volatility_weight:g}"
                f"_p{self.complexity_weight:g}")


# 12 configurations = small predeclared grid, not tuned on test data.
GATE_GRID = tuple(GateParameters(a, b, c)
                  for a in (0.0, 0.5)
                  for b in (0.0, 0.3)
                  for c in (0.25, 0.8, 1.3))


def gate_statistics(masked: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> dict[int, pd.DataFrame]:
    """Past-only, per-row quality indicators for every possible lookback window.

    grade_count = observed grade measurements within the exact last k semesters;
    coverage = fraction of observed values among 5k measurement cells;
    volatility = std of observed grade measurements divided by 1.5 (grade range=3).
    For 0/1 grade measurements, std=0 (no reliable volatility estimate).
    """
    if not windows or len(set(windows)) != len(windows) or any(not 1 <= k <= 8 for k in windows):
        raise ValueError("Windows must be unique integers between 1 and 8")
    out = {}
    for k in windows:
        lag = _aligned_lags(masked, k)
        grades = lag["current_avg_grade"]
        observed = np.isfinite(grades)
        count = observed.sum(axis=1).astype(float)
        sums = np.where(observed, grades, 0).sum(axis=1)
        mean = np.divide(sums, count, out=np.zeros(len(masked)), where=count > 0)
        sq = np.where(observed, (grades - mean[:, None]) ** 2, 0).sum(axis=1)
        volatility = np.sqrt(np.divide(sq, count, out=np.zeros(len(masked)), where=count > 1)) / 1.5
        total_measurements = sum(np.isfinite(lag[col]).sum(axis=1) for col in MEASURES)
        coverage = total_measurements / float(len(MEASURES) * k)
        out[k] = pd.DataFrame({
            "grade_count": count, "coverage": coverage,
            "normalized_volatility": volatility,
        }, index=masked.index)
    return out


def select_windows(stats: dict[int, pd.DataFrame], params: GateParameters,
                   rows: np.ndarray | None = None) -> np.ndarray:
    """Pure rule: maximize predeclared quality score, tie-break to shorter window.

    Q_ik = ln(1+n_ik) + a*c_ik - b*v_ik - d*(k-1)/4.
    Neither the choice nor its inputs depend on the student's target or predicted risk.
    """
    windows = sorted(stats)
    if not windows:
        raise ValueError("No candidate windows")
    arr = []
    for k in windows:
        data = stats[k] if rows is None else stats[k].iloc[rows]
        if not (data["grade_count"].ge(0).all() and data["coverage"].between(0, 1).all()):
            raise ValueError("Invalid quality statistics")
        score = (np.log1p(data["grade_count"].to_numpy(dtype=float))
                 + params.coverage_weight * data["coverage"].to_numpy(dtype=float)
                 - params.volatility_weight * data["normalized_volatility"].to_numpy(dtype=float)
                 - params.complexity_weight * ((k - 1) / 4))
        arr.append(score)
    return np.asarray(windows, dtype=int)[np.argmax(np.stack(arr, axis=0), axis=0)]


def combine_predictions(window_probabilities: dict[int, np.ndarray], chosen_windows: np.ndarray) -> np.ndarray:
    """Use ONLY the already trained predictor associated with each row's selected k."""
    windows = sorted(window_probabilities)
    if not windows or not np.isin(chosen_windows, windows).all():
        raise ValueError("Selected window outside trained models")
    matrix = np.stack([window_probabilities[k] for k in windows], axis=0)
    if matrix.ndim != 2 or len(chosen_windows) != matrix.shape[1]:
        raise ValueError("Prediction length mismatch")
    index = np.searchsorted(np.asarray(windows), chosen_windows)
    values = matrix[index, np.arange(len(chosen_windows))]
    if not (np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()):
        raise ValueError("Predictions must be finite probabilities")
    return values


def _metric_rows(y_val: pd.Series, y_test: pd.Series, val: np.ndarray, test: np.ndarray) -> dict:
    v, t = evaluate(y_val, val), evaluate(y_test, test)
    return {**{f"validation_{k}": value for k, value in v.items()},
            **{f"test_{k}": value for k, value in t.items()}}


def _best(group: pd.DataFrame, id_col: str) -> dict:
    # Aggregate across repeated masks before selecting; no test metrics used.
    agg = group.groupby(id_col, sort=False).agg(
        val_ap=("validation_average_precision", "mean"),
        val_brier=("validation_brier", "mean")
    ).reset_index()
    agg[id_col] = agg[id_col].astype(str)
    chosen = agg.sort_values(["val_ap", "val_brier", id_col],
                             ascending=[False, True, True], kind="stable").iloc[0]
    return {"id": str(chosen[id_col]), "validation_ap": float(chosen["val_ap"]),
            "validation_brier": float(chosen["val_brier"])}


def run_adaptive(csv_path: str | Path, output_dir: str | Path,
                 *, windows: tuple[int, ...] = WINDOWS,
                 rates: tuple[float, ...] = RATES,
                 patterns: tuple[str, ...] = PATTERNS,
                 seeds: tuple[int, ...] = SEEDS,
                 families: tuple[str, ...] = FAMILIES,
                 gate_grid: tuple[GateParameters, ...] = GATE_GRID) -> dict:
    """Match v0.3 data, masks, splits and base models for a paired comparison."""
    raw = pd.read_csv(csv_path)
    validate_experiment_data(raw)
    if not windows or tuple(sorted(set(windows))) != tuple(windows) or any(not 1 <= k <= 8 for k in windows):
        raise ValueError("windows must be ordered and unique, 1..8")
    if not rates or not patterns or not seeds or not families or not gate_grid:
        raise ValueError("Experiment dimensions cannot be empty")
    train = raw.term_index.isin(TRAIN_TERMS).to_numpy()
    val = raw.term_index.eq(VAL_TERM).to_numpy()
    test = raw.term_index.eq(TEST_TERM).to_numpy()
    val_positions, test_positions = np.flatnonzero(val), np.flatnonzero(test)
    y_train, y_val, y_test = [raw.loc[sel, TARGET] for sel in (train, val, test)]
    for y in (y_train, y_val, y_test):
        if len(y) == 0 or y.nunique() != 2:
            raise ValueError("Split missing or has one class")

    run_rows: list[dict] = []
    choices: dict[tuple, dict] = {}
    fits = 0
    for rate in rates:
        settings = [("mcar_cells", 7)] if rate == 0 else [
            (p, seed) for p in patterns for seed in seeds]
        for pattern, seed in settings:
            masked = inject_missingness(raw, rate, pattern, seed)
            stats = gate_statistics(masked, windows)
            features = {k: window_features(masked, k) for k in windows}
            for family in families:
                val_probs, test_probs = {}, {}
                for k in windows:
                    X, cols = features[k]
                    model = _model(family, cols)
                    model.fit(X.loc[train, cols], y_train)
                    fits += 1
                    val_probs[k] = model.predict_proba(X.loc[val, cols])[:, 1]
                    test_probs[k] = model.predict_proba(X.loc[test, cols])[:, 1]
                group_id = {"pattern": pattern, "missing_rate": rate, "mask_seed": seed,
                            "model": family}
                for k in windows:
                    run_rows.append({**group_id, "method": "fixed", "candidate": str(k),
                                     **_metric_rows(y_val, y_test, val_probs[k], test_probs[k])})
                for config in gate_grid:
                    val_choices = select_windows(stats, config, val_positions)
                    test_choices = select_windows(stats, config, test_positions)
                    choices[(pattern, rate, seed, family, config.name)] = {
                        "test_choices": test_choices,
                        "validation_choices": val_choices,
                    }
                    v_pred = combine_predictions(val_probs, val_choices)
                    t_pred = combine_predictions(test_probs, test_choices)
                    run_rows.append({**group_id, "method": "adaptive", "candidate": config.name,
                                     **_metric_rows(y_val, y_test, v_pred, t_pred)})

    all_runs = pd.DataFrame(run_rows)
    selections, test_rows, usage = [], [], []
    grouping = ("pattern", "missing_rate", "model")
    for (pattern, rate, family), group in all_runs.groupby(list(grouping), sort=True):
        fixed = _best(group[group.method == "fixed"], "candidate")
        adaptive = _best(group[group.method == "adaptive"], "candidate")
        key = {"pattern": pattern, "missing_rate": float(rate), "model": family}
        selections.append({**key, "selected_fixed_window": int(fixed["id"]),
                           "selected_adaptive_gate": adaptive["id"],
                           "validation_ap_fixed": fixed["validation_ap"],
                           "validation_ap_adaptive": adaptive["validation_ap"],
                           "validation_brier_fixed": fixed["validation_brier"],
                           "validation_brier_adaptive": adaptive["validation_brier"]})
        for seed in sorted(group.mask_seed.unique()):
            rows = group[group.mask_seed == seed]
            for label, method, candidate in [
                ("snapshot_k1", "fixed", "1"),
                ("longest_window", "fixed", str(max(windows))),
                ("validation_selected_fixed", "fixed", fixed["id"]),
                ("adaptive", "adaptive", adaptive["id"]),
            ]:
                chosen = rows[(rows.method == method) & (rows.candidate == candidate)]
                if chosen.empty:
                    continue  # e.g. user-supplied windows without k=1
                row = chosen.iloc[0]
                test_rows.append({**key, "mask_seed": int(seed), "approach": label,
                                  "candidate": candidate,
                                  "validation_ap": float(row["validation_average_precision"]),
                                  "test_ap": float(row["test_average_precision"]),
                                  "test_roc_auc": float(row["test_roc_auc"]),
                                  "test_brier": float(row["test_brier"])})
            selected_choices = choices[(pattern, rate, int(seed), family, adaptive["id"])]["test_choices"]
            for k in windows:
                usage.append({**key, "mask_seed": int(seed), "window": k,
                              "selected_rows": int((selected_choices == k).sum()),
                              "fraction": float((selected_choices == k).mean())})

    tests = pd.DataFrame(test_rows)
    if tests.empty:
        raise ValueError("No comparisons")
    summaries = []
    for (pattern, rate, family), group in tests.groupby(list(grouping), sort=True):
        summary = {"pattern": pattern, "missing_rate": float(rate), "model": family,
                   "n_masks": int(group.mask_seed.nunique())}
        for label, key in [("snapshot", "snapshot_k1"), ("longest", "longest_window"),
                           ("fixed", "validation_selected_fixed"), ("adaptive", "adaptive")]:
            matching = group[group.approach == key]
            if not matching.empty:
                for col in ("test_ap", "test_roc_auc", "test_brier"):
                    summary[f"{label}_{col}_mean"] = float(matching[col].mean())
                    summary[f"{label}_{col}_std"] = (float(matching[col].std(ddof=1))
                                                   if len(matching) > 1 else None)
        wide = group.pivot(index="mask_seed", columns="approach", values="test_ap")
        if {"adaptive", "validation_selected_fixed"}.issubset(wide.columns):
            paired = wide["adaptive"] - wide["validation_selected_fixed"]
            summary["adaptive_minus_fixed_ap_mean"] = float(paired.mean())
            summary["adaptive_minus_fixed_ap_std"] = (float(paired.std(ddof=1))
                                                       if len(paired) > 1 else None)
        summaries.append(summary)

    result = {
        "mode": "SYNTHETIC_DEMO_ONLY", "version": "0.4.0",
        "algorithm": "quality_gated_past_history_window_selection",
        "formula": "Q(k)=ln(1+observed_grades_k) + alpha*coverage_k - beta*normalized_grade_std_k - gamma*(k-1)/4; k*=argmax Q(k)",
        "design": {
            "windows": list(windows), "rates": list(rates), "patterns": list(patterns),
            "seeds": list(seeds), "train_terms": list(TRAIN_TERMS),
            "validation_term": VAL_TERM, "test_term": TEST_TERM,
            "number_of_students": int(raw.student_id.nunique()),
            "rows_total": int(len(raw)), "n_model_fits": int(fits),
            "n_gate_configs": len(gate_grid),
            "gate_grid": [dict(name=g.name, coverage_weight=g.coverage_weight,
                               volatility_weight=g.volatility_weight,
                               complexity_weight=g.complexity_weight) for g in gate_grid],
            "selection_metric": "mean validation average precision across mask seeds; tie: Brier score, then name",
            "data_source": "SYNTHETIC ONLY: synthetic_long_10_terms.csv, seed=2026",
            "model_settings": "identical to v0.3 ablation._model",
        },
        "limitations": [
            "Synthetic repeated-student trajectories do not establish benefit for real students.",
            "Simple MCAR/cell and blocked-term simulated missingness cannot represent MAR/MNAR.",
            "Mask repeats are not independent cohort replicates or confidence intervals.",
            "Grid selection on one validation term can overfit; independent cohorts and outer validation are needed.",
            "Global validation selection is performed for each missingness scenario; actual deployment may not know the scenario mechanism.",
            "Per-row adaptive gate chooses among trained predictors; it is a candidate algorithm, not established scientific novelty.",
            "No automated adverse academic decisions should rely on this demonstration.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(output / "adaptive_all_candidates_synthetic.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "adaptive_selections_synthetic.csv", index=False)
    tests.to_csv(output / "adaptive_test_comparison_synthetic.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "adaptive_comparison_synthetic.csv", index=False)
    pd.DataFrame(usage).to_csv(output / "adaptive_window_usage_synthetic.csv", index=False)
    (output / "adaptive_summary_synthetic.json").write_text(
        json.dumps({**result, "selections": selections, "comparison": summaries},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return {**result, "selections": selections, "comparison": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.4 quality-gated adaptive history research (synthetic only)")
    parser.add_argument("--input", default="data/synthetic_long_10_terms.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    result = run_adaptive(args.input, args.output)
    print(f"v{result['version']} completed: {result['design']['n_model_fits']} model fits, "
          f"{result['design']['n_gate_configs']} candidate gates")
    for item in result["comparison"]:
        if item["missing_rate"] in (0, 0.25):
            ap = item.get("adaptive_test_ap_mean")
            delta = item.get("adaptive_minus_fixed_ap_mean")
            print(f"{item['model']:20s} {item['pattern']:14s} {item['missing_rate']:.0%}: "
                  f"adaptive_AP={ap:.4f} paired_delta_vs_fixed={delta:+.4f}")


if __name__ == "__main__":
    main()
