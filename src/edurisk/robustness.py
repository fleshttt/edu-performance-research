"""EduRisk v0.5: prospective time / unseen-cohort validation and missing-data stress tests.

Synthetic research ONLY. No real academic decisions, no production model.
The test labels are referenced exclusively by the final scoring functions.
All hyperparameter/model/window selection is done on validation predictions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ablation import (
    MEASURES, TARGET, _model, evaluate, inject_missingness,
    validate_experiment_data, window_features,
)
from .adaptive import GateParameters, GATE_GRID, gate_statistics, select_windows, combine_predictions
from .generate_data import generate_synthetic

WINDOWS_V05 = (1, 2, 3, 4, 5)
SCENARIOS = (
    ("complete", 0.0),
    ("mar_attendance", 0.25),
    ("mnar_value", 0.25),
    ("contiguous_terms", 0.25),
)
FAMILIES_V05 = ("logistic_regression", "random_forest")
WEIGHT_TEMPERATURE = 0.05  # fixed before inspecting any holdout outcomes


@dataclass
class StudyFold:
    protocol: str
    fold_id: str
    data: pd.DataFrame
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    description: str

    def validate(self) -> None:
        validate_experiment_data(self.data)
        if len(self.train) != len(self.data) or len(self.validation) != len(self.data) or len(self.test) != len(self.data):
            raise ValueError("Split masks must match dataset length")
        if np.logical_and(self.train, self.validation).any() or np.logical_and(self.train, self.test).any() or np.logical_and(self.validation, self.test).any():
            raise ValueError("Train, validation and test observations must be disjoint")
        if not all(idx.any() for idx in (self.train, self.validation, self.test)):
            raise ValueError("All splits must be nonempty")
        if not all(self.data.loc[idx, TARGET].nunique() == 2 for idx in (self.train, self.validation, self.test)):
            raise ValueError("Each split must contain positive and negative synthetic labels")
        if self.data.loc[self.train, "term_index"].max() >= self.data.loc[self.validation, "term_index"].min():
            raise ValueError("Validation must occur after training periods")
        if self.data.loc[self.validation, "term_index"].max() >= self.data.loc[self.test, "term_index"].min():
            raise ValueError("Test must occur after validation periods")
        if self.protocol == "cohort_holdout":
            train_ids = set(self.data.loc[self.train, "student_id"])
            val_ids = set(self.data.loc[self.validation, "student_id"])
            test_ids = set(self.data.loc[self.test, "student_id"])
            if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
                raise ValueError("Cohort holdout student sets must not overlap")


def _term_split(data: pd.DataFrame, start: int, protocol: str, fold_id: str, description: str) -> StudyFold:
    term = data["term_index"]
    fold = StudyFold(protocol, fold_id, data,
                     term.between(start, start + 2).to_numpy(),
                     term.eq(start + 3).to_numpy(),
                     term.eq(start + 4).to_numpy(), description)
    fold.validate()
    return fold


def build_cohort_dataset(n_per_cohort: int = 120, n_terms: int = 10) -> pd.DataFrame:
    """Four non-overlapping synthetic groups from independent RNG seeds.

    A/B/C/D have the same stipulated generative mechanism; this is not
    an external real-world cohort or a measured institution shift.
    """
    chunks = []
    for idx, label in enumerate("ABCD"):
        df = generate_synthetic(n_students=n_per_cohort, n_terms=n_terms, seed=501 + idx * 117)
        numeric_id = df["student_id"].str.removeprefix("SYN-").astype(int)
        df["student_id"] = [f"SYN-{(idx + 1) * 100000 + i}" for i in numeric_id]
        df["cohort"] = label
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def create_folds(legacy_csv: str | Path,
                 protocols: tuple[str, ...] = ("legacy_anchor", "temporal_forward", "cohort_holdout"),
                 *, forward_students: int = 300, cohort_students: int = 120,
                 forward_starts: tuple[int, ...] = (5, 6, 7),
                 cohort_partitions: tuple[tuple[str, str, str, str], ...] = (("A", "B", "C", "D"), ("B", "C", "D", "A"))) -> list[StudyFold]:
    folds = []
    if "legacy_anchor" in protocols:
        raw = pd.read_csv(legacy_csv)
        folds.append(_term_split(raw, 5, "legacy_anchor", "legacy_5_7__8__9",
                                 "Same original v0.4 synthetic file; train 5–7, validation 8, test 9"))
    if "temporal_forward" in protocols:
        raw = generate_synthetic(n_students=forward_students, n_terms=max(forward_starts) + 6, seed=2031)
        for start in forward_starts:
            folds.append(_term_split(raw, start, "temporal_forward", f"forward_{start}_{start+2}__{start+3}__{start+4}",
                                     f"Same synthetic individuals; train {start}–{start+2}; validation {start+3}; test {start+4}"))
    if "cohort_holdout" in protocols:
        raw = build_cohort_dataset(n_per_cohort=cohort_students)
        for first, second, val, test in cohort_partitions:
            train = raw["cohort"].isin((first, second)) & raw["term_index"].between(5, 7)
            validation = raw["cohort"].eq(val) & raw["term_index"].eq(8)
            test_mask = raw["cohort"].eq(test) & raw["term_index"].eq(9)
            fold = StudyFold("cohort_holdout", f"train_{first}{second}_val_{val}_test_{test}", raw,
                             train.to_numpy(), validation.to_numpy(), test_mask.to_numpy(),
                             f"Unseen synthetic student cohorts: train {first}+{second} (terms 5–7); val {val} (8); test {test} (9)")
            fold.validate()
            folds.append(fold)
    if len(folds) == 0:
        raise ValueError("No valid protocols")
    return folds


def _normalize_weights(weights: np.ndarray, mean_target: float) -> np.ndarray:
    """Normalize relative risks to requested *expected* cell missingness, clipping at 0.95."""
    relative = np.maximum(np.asarray(weights, dtype=float), 0.05)
    relative = relative / float(relative.mean())
    return np.clip(mean_target * relative, 0.0, 0.95)


def inject_structured_missingness(data: pd.DataFrame, mechanism: str, rate: float, seed: int = 7) -> pd.DataFrame:
    """Simulated mechanisms; labels and original feature values kept separate.

    MAR: missing non-attendance cells depends only on available attendance at t;
         attendance itself stays observed. Thus rate target refers to all five measures.
    MNAR: every cell's missing probability depends on the very value being hidden;
          only possible as synthetic stress test, not an identifiable real mechanism.
    CONTIGUOUS: a random consecutive block of semesters vanishes for each student;
                this is NOT a claim of missing-at-random.
    """
    if not 0 <= rate < 1:
        raise ValueError("Invalid missingness rate")
    if mechanism not in {"complete", "mar_attendance", "mnar_value", "contiguous_terms", "mcar_cells", "blocked_terms"}:
        raise ValueError(f"Unknown mechanism {mechanism}")
    if mechanism == "complete":
        if rate != 0:
            raise ValueError("Complete control must have zero missingness")
        return data.copy(deep=True)
    if mechanism in {"mcar_cells", "blocked_terms"}:
        return inject_missingness(data, rate, mechanism, seed)
    rng = np.random.default_rng(seed)
    out = data.copy(deep=True)
    n = len(out)
    if mechanism == "mar_attendance":
        att = out["attendance_rate"].to_numpy(dtype=float)
        if not np.isfinite(att).all():
            raise ValueError("MAR driver attendance must be observed")
        # Higher missingness when *observed* attendance is low.
        relative = 0.25 + 2.4 * (1 - att)
        cell_prob = _normalize_weights(relative, rate * len(MEASURES) / (len(MEASURES) - 1))
        other = [m for m in MEASURES if m != "attendance_rate"]
        mask = rng.random((n, len(other))) < cell_prob[:, None]
        out[other] = out[other].mask(mask)
    elif mechanism == "mnar_value":
        for col in MEASURES:
            x = out[col].to_numpy(dtype=float)
            if not np.isfinite(x).all():
                raise ValueError("Synthetic MNAR simulation requires latent complete values")
            relative = (5.0 - x) / 3.0 if col == "current_avg_grade" else 1 - x
            probability = _normalize_weights(0.20 + np.maximum(relative, 0), rate)
            out.loc[:, col] = out[col].mask(rng.random(n) < probability)
    elif mechanism == "contiguous_terms":
        # One run per synthetic student, otherwise leave observations intact.
        for _, positions in out.groupby("student_id", sort=False).indices.items():
            idx = np.asarray(positions, dtype=int)
            terms = out.iloc[idx]["term_index"].to_numpy(dtype=int)
            number = max(1, int(np.rint(len(idx) * rate))) if rate else 0
            if number:
                first = int(rng.integers(0, len(idx) - number + 1))
                ordered = idx[np.argsort(terms)]
                out.loc[ordered[first:first+number], MEASURES] = np.nan
    if not out[["student_id", "term_index", TARGET]].equals(data[["student_id", "term_index", TARGET]]):
        raise RuntimeError("Missingness simulation changed identifiers or labels")
    return out


def _scoring(y: pd.Series, probabilities: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_{key}": val for key, val in evaluate(y, probabilities).items()}


def _rank(ap: float | None, brier: float | None, name: str) -> tuple[float, float, str]:
    return (-(float(ap) if ap is not None else -np.inf),
            float(brier) if brier is not None else np.inf, name)


def validation_softmax_weights(validation_ap: dict[int, float], temperature: float = WEIGHT_TEMPERATURE) -> dict[int, float]:
    """Pre-fixed softmax smoothing of the window-wise AP on validation only."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be positive")
    windows = sorted(validation_ap)
    if not windows:
        raise ValueError("No windows supplied")
    values = np.array([validation_ap[k] for k in windows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Validation AP values must be finite")
    logits = (values - values.max()) / temperature
    weights = np.exp(logits)
    weights /= weights.sum()
    return dict(zip(windows, weights.tolist()))


def _average(window_prob: dict[int, np.ndarray], weights: dict[int, float] | None = None) -> np.ndarray:
    windows = sorted(window_prob)
    stack = np.stack([window_prob[k] for k in windows], axis=0)
    if weights is None:
        return stack.mean(axis=0)
    w = np.array([weights[k] for k in windows], dtype=float)
    if not np.all(w >= 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Ensemble weights must be non-negative and sum to one")
    return w @ stack


def evaluate_fold(fold: StudyFold, *, scenarios: tuple[tuple[str, float], ...] = SCENARIOS,
                  windows: tuple[int, ...] = WINDOWS_V05,
                  families: tuple[str, ...] = FAMILIES_V05,
                  gate_grid: tuple[GateParameters, ...] = GATE_GRID,
                  mask_seed: int = 7) -> tuple[list[dict], list[dict], list[dict]]:
    """Return per-method metrics, validation-only selections and missingness details."""
    fold.validate()
    if not windows or tuple(sorted(set(windows))) != tuple(windows) or not set(windows).issubset(set(range(1, 9))):
        raise ValueError("Expected unique sorted history window lengths 1..8")
    if not gate_grid or not families or not scenarios:
        raise ValueError("Missing experiment dimensions")
    outputs, selections, masks = [], [], []
    y_train, y_validation, y_test = (fold.data.loc[idx, TARGET] for idx in (fold.train, fold.validation, fold.test))
    val_positions, test_positions = np.flatnonzero(fold.validation), np.flatnonzero(fold.test)
    for mechanism, rate in scenarios:
        masked = inject_structured_missingness(fold.data, mechanism, rate, mask_seed)
        masks.append({"protocol":fold.protocol, "fold_id":fold.fold_id, "mechanism":mechanism,
                      "missing_rate_requested":rate,
                      "missing_rate_actual":float(masked[MEASURES].isna().to_numpy().mean()),
                      "mask_seed": mask_seed})
        feats = {k: window_features(masked, k) for k in windows}
        gate_stats = gate_statistics(masked, windows)
        for family in families:
            val_predictions = {}
            test_predictions = {}
            for k in windows:
                X, columns = feats[k]
                estimator = _model(family, columns)
                estimator.fit(X.loc[fold.train, columns], y_train)
                val_predictions[k] = estimator.predict_proba(X.loc[fold.validation, columns])[:, 1]
                test_predictions[k] = estimator.predict_proba(X.loc[fold.test, columns])[:, 1]
            val_measures = {k: evaluate(y_validation, val_predictions[k]) for k in windows}
            # Never access test probabilities or test labels while selecting hyperparameters.
            best_fixed = min(windows, key=lambda k: _rank(val_measures[k]["average_precision"], val_measures[k]["brier"], str(k)))
            weighted = validation_softmax_weights({k: val_measures[k]["average_precision"] for k in windows})
            gate_val = {}
            for config in gate_grid:
                choices = select_windows(gate_stats, config, val_positions)
                preds = combine_predictions(val_predictions, choices)
                gate_val[config.name] = evaluate(y_validation, preds)
            best_gate = min(gate_grid, key=lambda config: _rank(gate_val[config.name]["average_precision"],
                                                                 gate_val[config.name]["brier"], config.name))
            test_choices = select_windows(gate_stats, best_gate, test_positions)
            selected = {"protocol":fold.protocol, "fold_id":fold.fold_id,
                        "mechanism":mechanism, "missing_rate_requested":rate,
                        "model":family, "fixed_window":best_fixed,
                        "gate":best_gate.name,
                        "ensemble_temperature":WEIGHT_TEMPERATURE,
                        "ensemble_weights":json.dumps(weighted, ensure_ascii=False),
                        "validation_ap_fixed":val_measures[best_fixed]["average_precision"],
                        "validation_ap_adaptive":gate_val[best_gate.name]["average_precision"]}
            selections.append(selected)
            predictions = {
                "snapshot_1": test_predictions[1] if 1 in test_predictions else test_predictions[min(windows)],
                "longest": test_predictions[max(windows)],
                "fixed_val": test_predictions[best_fixed],
                "adaptive_gate": combine_predictions(test_predictions, test_choices),
                "uniform_ensemble": _average(test_predictions),
                "weighted_ensemble": _average(test_predictions, weighted),
            }
            val_pred_methods = {
                "snapshot_1":val_predictions[1] if 1 in val_predictions else val_predictions[min(windows)],
                "longest":val_predictions[max(windows)],
                "fixed_val":val_predictions[best_fixed],
                "adaptive_gate":combine_predictions(val_predictions, select_windows(gate_stats, best_gate, val_positions)),
                "uniform_ensemble":_average(val_predictions),
                "weighted_ensemble":_average(val_predictions, weighted),
            }
            for method, probability in predictions.items():
                outputs.append({"protocol":fold.protocol,"fold_id":fold.fold_id,
                                "mechanism":mechanism,"missing_rate_requested":rate,
                                "model":family,"method":method,
                                "n_train":int(fold.train.sum()),"n_val":int(fold.validation.sum()),"n_test":int(fold.test.sum()),
                                "train_terms":sorted(fold.data.loc[fold.train,"term_index"].unique().astype(int).tolist()),
                                "val_term":int(fold.data.loc[fold.validation,"term_index"].unique()[0]),
                                "test_term":int(fold.data.loc[fold.test,"term_index"].unique()[0]),
                                **_scoring(y_validation,val_pred_methods[method],"validation"),
                                **_scoring(y_test,probability,"test")})
    return outputs, selections, masks


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Descriptive cross-fold aggregation; folds are dependent, NOT CIs."""
    cols = ["test_average_precision", "test_roc_auc", "test_brier"]
    summary = runs.groupby(["protocol", "mechanism", "missing_rate_requested", "model", "method"], dropna=False).agg(
        n_folds=("fold_id", "nunique"),
        **{f"{x}_mean":(x,"mean") for x in cols},
        **{f"{x}_std":(x,"std") for x in cols},
    ).reset_index()
    return summary


def run_robustness(legacy_csv: str | Path, output_dir: str | Path,
                   *, protocols: tuple[str, ...] = ("legacy_anchor", "temporal_forward", "cohort_holdout"),
                   scenarios: tuple[tuple[str, float], ...] = SCENARIOS,
                   windows: tuple[int, ...] = WINDOWS_V05,
                   families: tuple[str, ...] = FAMILIES_V05,
                   gate_grid: tuple[GateParameters, ...] = GATE_GRID,
                   mask_seed: int = 7,
                   forward_students: int = 300, cohort_students: int = 120,
                   forward_starts: tuple[int, ...] = (5, 6, 7),
                   cohort_partitions: tuple[tuple[str, str, str, str], ...] = (("A", "B", "C", "D"), ("B", "C", "D", "A"))) -> dict:
    folds = create_folds(legacy_csv, protocols, forward_students=forward_students,
                         cohort_students=cohort_students, forward_starts=forward_starts,
                         cohort_partitions=cohort_partitions)
    raw_records, raw_selections, raw_masks = [], [], []
    for fold in folds:
        records, choices, masks = evaluate_fold(fold, scenarios=scenarios, windows=windows,
                                               families=families, gate_grid=gate_grid, mask_seed=mask_seed)
        raw_records.extend(records)
        raw_selections.extend(choices)
        raw_masks.extend(masks)
    runs = pd.DataFrame(raw_records)
    selections = pd.DataFrame(raw_selections)
    summary = summarize_runs(runs)
    missing = pd.DataFrame(raw_masks)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_runs = runs.copy()
    csv_runs["train_terms"] = csv_runs["train_terms"].apply(lambda x: ",".join(map(str, x)))
    csv_runs.to_csv(out / "robustness_runs_synthetic.csv", index=False)
    selections.to_csv(out / "robustness_selections_synthetic.csv", index=False)
    summary.to_csv(out / "robustness_aggregates_synthetic.csv", index=False)
    missing.to_csv(out / "robustness_missingness_synthetic.csv", index=False)

    deltas=[]
    for group_key, group in runs.groupby(["protocol", "fold_id", "mechanism", "missing_rate_requested", "model"]):
        fixed = group[group.method.eq("fixed_val")].iloc[0]
        for approach in ("adaptive_gate", "uniform_ensemble", "weighted_ensemble"):
            variant = group[group.method.eq(approach)].iloc[0]
            deltas.append(dict(zip(["protocol", "fold_id", "mechanism", "missing_rate_requested", "model"],group_key)) |
                          {"method":approach,
                           "delta_ap_vs_fixed":float(variant.test_average_precision-fixed.test_average_precision),
                           "delta_brier_vs_fixed":float(variant.test_brier-fixed.test_brier)})
    delta_df=pd.DataFrame(deltas)
    delta_df.to_csv(out / "robustness_deltas_synthetic.csv", index=False)
    summary_delta=delta_df.groupby(["protocol","mechanism","missing_rate_requested","model","method"],sort=True).agg(
        n_folds=("fold_id","nunique"),mean_delta_ap=("delta_ap_vs_fixed","mean"),
        min_delta_ap=("delta_ap_vs_fixed","min"),max_delta_ap=("delta_ap_vs_fixed","max"),
        mean_delta_brier=("delta_brier_vs_fixed","mean")
    ).reset_index()
    summary_delta.to_csv(out / "robustness_delta_summary_synthetic.csv",index=False)

    meta = {"mode":"SYNTHETIC_DEMO_ONLY","version":"0.5.0",
            "research_focus":"Temporal transfer, disjoint synthetic cohorts and structured missingness",
            "protocols":[{"protocol":f.protocol,"fold_id":f.fold_id,"description":f.description,
                          "train_rows":int(f.train.sum()),"validation_rows":int(f.validation.sum()),"test_rows":int(f.test.sum()),
                          "training_student_ids":int(f.data.loc[f.train,"student_id"].nunique()),
                          "validation_student_ids":int(f.data.loc[f.validation,"student_id"].nunique()),
                          "test_student_ids":int(f.data.loc[f.test,"student_id"].nunique()),
                          "train_prevalence":float(f.data.loc[f.train,TARGET].mean()),
                          "validation_prevalence":float(f.data.loc[f.validation,TARGET].mean()),
                          "test_prevalence":float(f.data.loc[f.test,TARGET].mean())} for f in folds],
            "design":{"window_semesters":list(windows),"missingness_scenarios":[{"mechanism":p,"requested_rate":r} for p,r in scenarios],
                      "families":list(families),"mask_seed":mask_seed,
                      "methods":["snapshot_1","longest","fixed_val","adaptive_gate","uniform_ensemble","weighted_ensemble"],
                      "validation_selection":"best fixed window and gate parameters chosen by AP, then Brier, validation only",
                      "ensemble":"uniform mean or exp(AP_validation / 0.05) normalized weights; temperature fixed a priori",
                      "train_only_preprocessing":True,
                      "split_order":"train times strictly before validation strictly before test",
                      "cohorts":"synthetic independent random seeds, same underlying generator"},
            "n_folds":len(folds),"n_trained_models":len(folds)*len(scenarios)*len(families)*len(windows),
            "limitations":[
                "All observations and cohort labels are synthetic; not a real external validation.",
                "Forward folds share some students and periods; fold standard deviations are descriptive, not confidence intervals.",
                "Cohorts use different random seeds but the same data-generating process; no causal or distribution-shift claim.",
                "MAR is simulated conditional on always-observed attendance; MNAR uses the value masked by the simulator, not an identifiable real process.",
                "A single fixed missingness mask seed per scenario gives no uncertainty estimate over missing-data realizations.",
                "Separate tuning on one validation partition in each fold can still overfit; held-out test is never used for selection.",
                "No assertion of novel method, improvement on real student data or automatic academic decision-making.",
            ],
            "aggregate_results":json.loads(summary.to_json(orient="records")),
            "comparative_deltas":json.loads(summary_delta.to_json(orient="records")),
            "mask_realizations":json.loads(missing.to_json(orient="records")),
            }
    (out / "robustness_summary_synthetic.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    return meta


def main() -> None:
    parser=argparse.ArgumentParser(description="EduRisk v0.5 - synthetic temporal, cohort and missingness robustness")
    parser.add_argument("--legacy",default="data/synthetic_long_10_terms.csv",help="Original v0.4 reference dataset")
    parser.add_argument("--output",default="reports")
    parser.add_argument("--fast",action="store_true",help="Small smoke experiment, not for thesis reporting")
    args=parser.parse_args()
    if args.fast:
        result=run_robustness(args.legacy,args.output,protocols=("temporal_forward",),
                              scenarios=SCENARIOS[:2],windows=(1,2,3),
                              families=("logistic_regression",),gate_grid=GATE_GRID[:2],
                              forward_students=85,forward_starts=(5,))
    else:
        result=run_robustness(args.legacy,args.output)
    print(f"v{result['version']}: {result['n_trained_models']} models; {result['n_folds']} folds; "
          f"{len(result['aggregate_results'])} aggregated rows")
    print(f"Synthetic-only report: {Path(args.output) / 'robustness_summary_synthetic.json'}")

if __name__ == "__main__":
    main()
