"""Reproducible snapshot vs longitudinal baselines on synthetic data only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import (FEATURE_COLUMNS, HISTORY_COLUMNS, LONGITUDINAL_COLUMNS,
                     TARGET_COLUMN, TRAIN_TERMS, VALIDATION_TERM, TEST_TERM)
from .dataset import split_chronologically
from .history import add_history_features


def scores(y: pd.Series, p: np.ndarray, threshold: float = 0.5) -> dict:
    y_arr, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    prediction = (p >= threshold).astype(int)
    return {
        "support": int(len(y_arr)), "positives": int(y_arr.sum()),
        "base_rate": float(y_arr.mean()),
        "precision": float(precision_score(y_arr, prediction, zero_division=0)),
        "recall": float(recall_score(y_arr, prediction, zero_division=0)),
        "f1": float(f1_score(y_arr, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_arr, p)),
        "confusion_matrix": confusion_matrix(y_arr, prediction, labels=[0, 1]).tolist(),
        "average_precision": float(average_precision_score(y_arr, p)) if y_arr.sum() > 0 else None,
        "roc_auc": float(roc_auc_score(y_arr, p)) if len(np.unique(y_arr)) == 2 else None,
    }


def build_models() -> dict:
    """Each estimator includes train-fitted imputation/scaling; no leakage."""
    definitions = [
        ("dummy_prior", FEATURE_COLUMNS, DummyClassifier(strategy="prior")),
        ("logistic_regression", FEATURE_COLUMNS,
         LogisticRegression(max_iter=1000, random_state=42)),
        ("random_forest", FEATURE_COLUMNS,
         RandomForestClassifier(n_estimators=120, min_samples_leaf=8,
                                max_depth=7, random_state=42, n_jobs=-1)),
        ("history_logistic_regression", LONGITUDINAL_COLUMNS,
         LogisticRegression(max_iter=1000, random_state=42)),
        ("history_random_forest", LONGITUDINAL_COLUMNS,
         RandomForestClassifier(n_estimators=120, min_samples_leaf=8,
                                max_depth=7, random_state=42, n_jobs=-1)),
    ]
    models = {}
    for name, columns, estimator in definitions:
        preparation = ColumnTransformer([("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), columns)])
        models[name] = (Pipeline([("prepare", preparation), ("model", estimator)]), columns)
    return models


def _metric_delta(base: dict, history: dict) -> dict:
    """Positive AP/ROC and negative Brier are improvements; not significance tests."""
    return {
        name: None if base[name] is None or history[name] is None
        else round(float(history[name] - base[name]), 6)
        for name in ("average_precision", "roc_auc", "brier", "recall", "f1")
    }


def run_experiment(csv_path: str | Path, output_dir: str | Path) -> dict:
    data = pd.read_csv(csv_path)
    # Validation is on raw measurements before feature engineering.
    split_chronologically(data)
    enriched = add_history_features(data)
    train, validation, test = split_chronologically(enriched)
    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data needs positive and negative target examples")
    models = build_models()
    model_metrics = {}
    for name, (model, columns) in models.items():
        model.fit(train[columns], train[TARGET_COLUMN])
        model_metrics[name] = {
            "feature_set": "history" if name.startswith("history_") else "snapshot",
            "validation": scores(validation[TARGET_COLUMN],
                                 model.predict_proba(validation[columns])[:, 1]),
            "test": scores(test[TARGET_COLUMN], model.predict_proba(test[columns])[:, 1]),
        }

    # Select ONLY on validation; no tuning or feature selection based on test.
    ranked = sorted(models, key=lambda name: (
        -(model_metrics[name]["validation"]["average_precision"] or 0),
        model_metrics[name]["validation"]["brier"], name))
    selected_name = ranked[0]
    selected_model, selected_columns = models[selected_name]
    probability = selected_model.predict_proba(test[selected_columns])[:, 1]
    predictions = test[["student_id", "term_index", "current_avg_grade", TARGET_COLUMN]].copy()
    predictions["risk_probability"] = np.round(probability, 6)
    predictions["predicted_risk"] = (probability >= 0.5).astype(int)
    predictions = predictions.sort_values("risk_probability", ascending=False)
    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(result_dir / "predictions_synthetic.csv", index=False)
    enriched[["student_id", "term_index", *LONGITUDINAL_COLUMNS[1:]]].to_csv(
        result_dir / "history_features_synthetic.csv", index=False, na_rep="")

    importance = []
    if selected_name != "dummy_prior" and validation[TARGET_COLUMN].nunique() == 2:
        perm = permutation_importance(
            selected_model, validation[selected_columns], validation[TARGET_COLUMN],
            n_repeats=4, random_state=42, scoring="average_precision")
        importance = sorted(
            ({"feature": col, "mean_permutation_score_drop": round(float(score), 5)}
             for col, score in zip(selected_columns, perm.importances_mean)),
            key=lambda item: item["mean_permutation_score_drop"], reverse=True)

    comparisons = {}
    for family in ("logistic_regression", "random_forest"):
        comparisons[family] = {
            split: _metric_delta(model_metrics[family][split],
                                 model_metrics["history_" + family][split])
            for split in ("validation", "test")
        }

    report = {
        "mode": "SYNTHETIC_DEMO_ONLY", "version": "0.2.0",
        "disclaimer": "Synthetic experiment only. Not evidence of accuracy on real students. No automated decisions.",
        "task": "Next-term grade below 3.0 on a synthetic 2–5 scale",
        "design": {
            "train_terms": list(TRAIN_TERMS), "validation_term": VALIDATION_TERM,
            "test_term": TEST_TERM, "decision_threshold": 0.5,
            "features": list(selected_columns),
            "snapshot_features": FEATURE_COLUMNS,
            "history_features": HISTORY_COLUMNS,
            "longitudinal_features": LONGITUDINAL_COLUMNS,
            "history_window_terms": 3,
            "split_note": "Same synthetic students recur; testing forecast for returning students, not transfer to unseen students.",
            "availability_note": "All historical variables at t are calculated solely from the current term and periods t-1/t-2; no future labels used.",
        },
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected_model": selected_name,
        "selection_method": "Maximum validation AP, then minimum validation Brier; test excluded from selection",
        "metrics": model_metrics,
        "history_vs_snapshot": comparisons,
        "permutation_importance_validation": importance,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(selected_model, result_dir / "selected_model_synthetic.joblib")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare snapshot and longitudinal models on synthetic records")
    parser.add_argument("--input", default="data/synthetic_demo.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    report = run_experiment(args.input, args.output)
    print("SYNTHETIC DEMO — not a real-data research result")
    print(f"Selected on validation only: {report['selected_model']}")
    for name, item in report["metrics"].items():
        m = item["test"]
        ap = "N/A" if m["average_precision"] is None else f"{m['average_precision']:.3f}"
        auc = "N/A" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        print(f"{name:29s} test AP={ap} ROC-AUC={auc} F1={m['f1']:.3f}")
    print(f"Reports saved in {args.output}/")


if __name__ == "__main__":
    main()
