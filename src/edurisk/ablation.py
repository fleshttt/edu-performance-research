"""v0.3: prospective-window/missing-data ablation on SYNTHETIC records only.

Window k refers to t, t-1, ..., t-(k-1). Targets are NEVER used to
construct features. All preprocessing is fitted to training rows only.
The same cohort and chronological split are held fixed across conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .history import BASE_COLUMNS, MEASURES

TARGET = "next_term_low_performance"
WINDOWS = (1, 2, 3, 4, 5)
RATES = (0.0, 0.1, 0.25, 0.4)
PATTERNS = ("mcar_cells", "blocked_terms")
SEEDS = (7, 21, 42)
TRAIN_TERMS = (5, 6, 7)
VAL_TERM, TEST_TERM = 8, 9
FEATURE_METRICS = {
    "current_avg_grade": "grade",
    "attendance_rate": "attendance",
    "assignment_completion_rate": "assignments",
    "credits_completion_rate": "credits",
    "late_submission_rate": "lateness",
}


def validate_experiment_data(df: pd.DataFrame) -> None:
    required = {"student_id", "term_index", TARGET, *BASE_COLUMNS, *MEASURES}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df[["student_id", "term_index", TARGET]].isna().any().any():
        raise ValueError("Missing key fields")
    if df.duplicated(["student_id", "term_index"]).any():
        raise ValueError("Duplicate student+term")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be 0/1")
    if not np.all(np.isfinite(df["term_index"].to_numpy(dtype=float))):
        raise ValueError("Term index must be finite")
    if not np.all(df["term_index"] == df["term_index"].astype(int)):
        raise ValueError("Term index must be integer")
    if not set([*TRAIN_TERMS, VAL_TERM, TEST_TERM]).issubset(set(df["term_index"])):
        raise ValueError("Must contain forecast times t=5,6,7,8,9; generate with --terms 10")
    if not df["student_id"].astype(str).str.fullmatch(r"SYN-\d+").all():
        raise ValueError("Only synthetic student IDs may be used in this demonstration")
    for name in MEASURES:
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f"Not numeric: {name}")
    if df[MEASURES].isin([np.inf, -np.inf]).any().any():
        raise ValueError("Nonfinite measurements not supported")


def inject_missingness(df: pd.DataFrame, rate: float, pattern: str, seed: int) -> pd.DataFrame:
    """Deterministic MCAR cell loss or whole-observation (term) loss.

    Underlying labels, keys, row counts remain fixed. Mask random draws are
    monotone in rate for a given seed/pattern, enabling paired comparisons.
    """
    if not 0 <= rate < 1 or pattern not in PATTERNS:
        raise ValueError("Invalid missingness rate or mechanism")
    out = df.copy(deep=True)
    random = np.random.default_rng(seed)
    if pattern == "mcar_cells":
        mask = random.random((len(out), len(MEASURES))) < rate
    else:
        mask = np.broadcast_to((random.random((len(out), 1)) < rate),
                               (len(out), len(MEASURES)))
    out.loc[:, MEASURES] = out[MEASURES].mask(mask)
    # Avoid giving prior-derived shortcut grade_trend to the estimator.
    return out


def _aligned_lags(frame: pd.DataFrame, window: int) -> dict[str, np.ndarray]:
    """Exact term t-j joins, not merely previous observed row."""
    keys = ["student_id", "term_index"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate student-term observations")
    if not 1 <= window <= 8:
        raise ValueError("Unsupported history window")
    result = {m: np.full((len(frame), window), np.nan) for m in MEASURES}
    # Lookup via exact student+calendar-semester keys; a dropped semester is NaN.
    ref = frame.set_index(keys)[MEASURES]
    for j in range(window):
        shifted = pd.MultiIndex.from_arrays(
            [frame["student_id"].to_numpy(), frame["term_index"].to_numpy() - j],
            names=keys,
        )
        past = ref.reindex(shifted).to_numpy(dtype=float)
        for i, name in enumerate(MEASURES):
            result[name][:, j] = past[:, i]
    return result


def _slope(values: np.ndarray) -> np.ndarray:
    """Least-squares slope per semester using observed entries in the window."""
    n, k = values.shape
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    x = -np.arange(k, dtype=float)[None, :]  # t, t-1, ..., t-k+1
    cx = np.divide((valid * x).sum(axis=1), counts,
                   out=np.zeros(n, dtype=float), where=counts > 0)
    y = np.where(valid, values, 0.0)
    cy = np.divide(y.sum(axis=1), counts, out=np.zeros(n), where=counts > 0)
    centered_x = x - cx[:, None]
    num = (np.where(valid, centered_x * (values - cy[:, None]), 0.0)).sum(axis=1)
    den = (np.where(valid, centered_x * centered_x, 0.0)).sum(axis=1)
    return np.divide(num, den, out=np.full(n, np.nan), where=(counts > 1) & (den > 0))


def window_features(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Compute fixed past-only features in original row order; no target access."""
    if not 1 <= window <= 8:
        raise ValueError("window out of range")
    if df.duplicated(["student_id", "term_index"]).any():
        raise ValueError("Duplicate student-term")
    features = df[BASE_COLUMNS].copy()
    if window == 1:
        return features, list(features.columns)
    lag = _aligned_lags(df, window)
    for measure, alias in FEATURE_METRICS.items():
        values = lag[measure]
        observed = np.isfinite(values)
        counts = observed.sum(axis=1)
        summed = np.where(observed, values, 0.0).sum(axis=1)
        mean = np.divide(summed, counts, out=np.full(len(df), np.nan), where=counts > 0)
        features[f"{alias}_mean_{window}"] = mean
        features[f"{alias}_slope_{window}"] = _slope(values)
        features[f"{alias}_n_{window}"] = counts
        if measure == "current_avg_grade":
            sq = np.where(observed, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
            features[f"grade_std_{window}"] = np.sqrt(np.divide(
                sq, counts, out=np.full(len(df), np.nan), where=counts > 0))
            for j in range(1, window):
                features[f"grade_lag_{j}"] = values[:, j]
        features[f"{alias}_delta_1"] = values[:, 0] - values[:, 1]
    # Allows an estimator to distinguish missing longitudinal records from low grades.
    # Each observed semester can have only some measurements present.
    present_terms = np.stack([np.isfinite(lag[m]) for m in MEASURES], axis=2).any(axis=2)
    features["observed_semesters"] = present_terms.sum(axis=1)
    features["missing_semester_fraction"] = 1 - features["observed_semesters"] / window
    return features, list(features.columns)


def _model(family: str, columns: list[str]) -> Pipeline:
    if family == "logistic_regression":
        clf = LogisticRegression(max_iter=1500, random_state=42)
    elif family == "random_forest":
        clf = RandomForestClassifier(n_estimators=70, max_depth=7, min_samples_leaf=8,
                                     n_jobs=1, random_state=42)
    else:
        raise ValueError(f"Unknown model: {family}")
    return Pipeline([
        ("prepare", ColumnTransformer([("numerical", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), columns)])),
        ("model", clf),
    ])


def evaluate(y: pd.Series, prob: np.ndarray) -> dict:
    a = np.asarray(y, dtype=int)
    return {
        "n": int(len(a)), "positives": int(a.sum()),
        "average_precision": float(average_precision_score(a, prob)) if a.sum() else None,
        "roc_auc": float(roc_auc_score(a, prob)) if len(np.unique(a)) == 2 else None,
        "brier": float(brier_score_loss(a, prob)),
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 6)


def run_ablation(csv_path: str | Path, output_dir: str | Path,
                 *, windows: tuple[int, ...] = WINDOWS,
                 rates: tuple[float, ...] = RATES,
                 patterns: tuple[str, ...] = PATTERNS,
                 seeds: tuple[int, ...] = SEEDS,
                 families: tuple[str, ...] = ("logistic_regression", "random_forest")) -> dict:
    raw = pd.read_csv(csv_path)
    validate_experiment_data(raw)
    if not windows or min(windows) < 1 or max(windows) > 8:
        raise ValueError("Need valid windows")
    # Same rows/cohort for all conditions, not a random train/test split.
    roles = raw["term_index"]
    train_idx = roles.isin(TRAIN_TERMS).to_numpy()
    val_idx = roles.eq(VAL_TERM).to_numpy()
    test_idx = roles.eq(TEST_TERM).to_numpy()
    for idx in (train_idx, val_idx, test_idx):
        if not idx.any():
            raise ValueError("Empty split")
        if raw.loc[idx, TARGET].nunique() != 2:
            raise ValueError("Each split must have both target classes")
    rows = []
    meta = []
    for rate in rates:
        # Zero-rate controls are identical under all seeds/mechanisms; run once.
        settings = [("mcar_cells", 7)] if rate == 0 else [
            (p, seed) for p in patterns for seed in seeds
        ]
        for pattern, seed in settings:
            masked = inject_missingness(raw, rate, pattern, seed)
            actual_rate = float(masked[MEASURES].isna().to_numpy().mean())
            meta.append({"missing_rate_requested": rate, "pattern": pattern,
                         "mask_seed": seed, "actual_missing_fraction": round(actual_rate, 6)})
            for window in windows:
                X, columns = window_features(masked, window)
                for family in families:
                    estimator = _model(family, columns)
                    estimator.fit(X.loc[train_idx, columns], raw.loc[train_idx, TARGET])
                    val_prob = estimator.predict_proba(X.loc[val_idx, columns])[:, 1]
                    test_prob = estimator.predict_proba(X.loc[test_idx, columns])[:, 1]
                    row = {"pattern": pattern, "missing_rate": rate,
                           "mask_seed": seed, "window_semesters": window, "model": family,
                           "n_features": len(columns), "n_train": int(train_idx.sum()),
                           "n_validation": int(val_idx.sum()), "n_test": int(test_idx.sum())}
                    for split_name, y, probs in [
                        ("validation", raw.loc[val_idx, TARGET], val_prob),
                        ("test", raw.loc[test_idx, TARGET], test_prob),
                    ]:
                        for metric, val in evaluate(y, probs).items():
                            row[f"{split_name}_{metric}"] = val
                    rows.append(row)
    detailed = pd.DataFrame(rows)
    # Summarize repeated missingness masks, not independent population samples.
    metrics = [f"{split}_{metric}" for split in ("validation", "test")
               for metric in ("average_precision", "roc_auc", "brier")]
    grouper = ["pattern", "missing_rate", "window_semesters", "model"]
    agg = detailed.groupby(grouper, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in agg.columns]
    # Stdev is undefined at 0% because we deliberately avoid duplicate zero-rate fits.
    agg = agg.round(6)
    # Window selection uses validation only. This selection is exploratory per scenario;
    # compare test results separately, and do not claim an unbiased algorithm advantage.
    picks = []
    for (pattern, rate, model), group in agg.groupby(["pattern", "missing_rate", "model"]):
        chosen = group.sort_values(["validation_average_precision_mean",
                                    "validation_brier_mean", "window_semesters"],
                                   ascending=[False, True, True]).iloc[0]
        baseline = group.loc[group["window_semesters"] == 1].iloc[0]
        picks.append({
            "pattern": pattern, "missing_rate": float(rate), "model": model,
            "chosen_window_on_validation": int(chosen["window_semesters"]),
            "validation_ap": _rounded(chosen["validation_average_precision_mean"]),
            "test_ap_chosen": _rounded(chosen["test_average_precision_mean"]),
            "test_ap_snapshot": _rounded(baseline["test_average_precision_mean"]),
            "test_ap_delta_vs_snapshot": _rounded(
                chosen["test_average_precision_mean"] - baseline["test_average_precision_mean"]),
        })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output / "ablation_runs_synthetic.csv", index=False)
    agg.to_csv(output / "ablation_aggregates_synthetic.csv", index=False)
    pd.DataFrame(picks).to_csv(output / "ablation_selections_synthetic.csv", index=False)
    report = {
        "mode": "SYNTHETIC_DEMO_ONLY", "version": "0.3.0",
        "research_question": "How do 1–5 term history windows and missing data affect next-term risk forecasts?",
        "limitations": [
            "All data are synthetic; no claims about real students or educational value.",
            "Same artificial students recur; no external-cohort generalization evaluated.",
            "MCAR cell loss and random full-term deletion are simplified simulation mechanisms, not realistic MAR/MNAR missingness.",
            "Mask seeds are repeated missingness perturbations of ONE cohort, not independent population samples or confidence intervals.",
            "Comparison of test metrics across all windows is exploratory; validation alone determines selected window per scenario.",
            "A candidate adaptive-history algorithm is NOT scientifically novel until compared rigorously with prior methods.",
        ],
        "design": {
            "windows": list(windows), "missing_rates": list(rates),
            "missingness_patterns": list(patterns), "mask_seeds": list(seeds),
            "train_terms": list(TRAIN_TERMS), "validation_term": VAL_TERM, "test_term": TEST_TERM,
            "target": "next-term grade < 3.0 on generated 2–5 scale",
            "target_not_masked": True,
            "features": "Only observed data from t, t-1, ..., t-(window-1); median imputation fitted on training split",
            "population": "Forecast returning synthetic students; rows of the same students occur in all chronological splits",
            "window_selection": "Maximum mean validation AP, tie-break minimum validation Brier; test excluded",
            "n_students": int(raw["student_id"].nunique()),
            "n_observations": int(len(raw)),
            "split_rows": {"train": int(train_idx.sum()), "validation": int(val_idx.sum()),
                           "test": int(test_idx.sum())},
            "prevalence": {label: float(raw.loc[idx, TARGET].mean()) for label, idx in [
                ("train", train_idx), ("validation", val_idx), ("test", test_idx)]},
        },
        "n_model_fits": int(len(detailed)),
        "actual_missing_rates": meta,
        "window_selections": picks,
    }
    (output / "ablation_summary_synthetic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.3 synthetic history-window/missingness ablation")
    parser.add_argument("--input", default="data/synthetic_long_10_terms.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    result = run_ablation(args.input, args.output)
    print(f"Synthetic ablation complete: {result['n_model_fits']} fits, "
          f"{result['design']['n_students']} synthetic students")
    for p in result["window_selections"]:
        if p["missing_rate"] in (0.0, 0.25):
            print(f"{p['model']:<19} {p['pattern']:<14} "
                  f"missing={p['missing_rate']:.0%} window={p['chosen_window_on_validation']} "
                  f"test_delta_AP={p['test_ap_delta_vs_snapshot']:+.4f}")
    print(f"Synthetic-only outputs: {args.output}/ablation_*_synthetic.*")


if __name__ == "__main__":
    main()
