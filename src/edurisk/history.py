"""Past-only longitudinal feature extraction for student-term observations.

A feature for period t uses values from periods t-2, t-1, t only.
Targets, metadata and observations t+1 onward are NEVER inspected when building it.
The function is deterministic, tolerates missing periods and preserves input row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "term_index", "attendance_rate", "assignment_completion_rate",
    "late_submission_rate", "current_avg_grade", "credits_completion_rate",
]
HISTORY_COLUMNS = [
    "grade_lag_1", "grade_lag_2", "grade_mean_3", "grade_std_3",
    "grade_slope_3", "grade_change_1", "attendance_change_1",
    "assignment_change_1", "credits_change_1", "history_terms_available",
]
LONGITUDINAL_COLUMNS = BASE_COLUMNS + HISTORY_COLUMNS
MEASURES = [
    "attendance_rate", "assignment_completion_rate", "late_submission_rate",
    "current_avg_grade", "credits_completion_rate",
]


def _number(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _delta(current: object, previous: object) -> float:
    a, b = _number(current), _number(previous)
    return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with longitudinal features, using solely observed past/current rows.

    `student_id` and `term_index` identify observations. Missing semesters do not
    get mistaken for consecutive ones: lag_1 refers exactly to term t-1.
    Missing grades in a lookback window are ignored in mean/std/slope, but
    the count feature reports how many *observed non-missing grades* remain.
    """
    required = {"student_id", "term_index", *MEASURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for history: {sorted(missing)}")
    if df.duplicated(["student_id", "term_index"]).any():
        raise ValueError("Duplicate student_id + term_index entries")
    if df["term_index"].isna().any() or not (df["term_index"] % 1 == 0).all():
        raise ValueError("Term index must be an integer")

    # Work in original row order to make it impossible to silently realign labels.
    result = df.copy()
    values = {c: np.full(len(result), np.nan, dtype=float) for c in HISTORY_COLUMNS}
    ordered = result.assign(_row_position=np.arange(len(result)))
    ordered = ordered.sort_values(["student_id", "term_index"], kind="stable")
    for _, group in ordered.groupby("student_id", sort=False):
        by_term = {int(row["term_index"]): row for _, row in group.iterrows()}
        for _, row in group.iterrows():
            pos, term = int(row["_row_position"]), int(row["term_index"])
            prior = by_term.get(term - 1)
            prior_2 = by_term.get(term - 2)
            now_grade = _number(row["current_avg_grade"])
            previous_grade = _number(prior["current_avg_grade"]) if prior is not None else np.nan
            prev2_grade = _number(prior_2["current_avg_grade"]) if prior_2 is not None else np.nan
            observations = [(i, _number(by_term[i]["current_avg_grade"]))
                            for i in (term - 2, term - 1, term) if i in by_term]
            observations = [(i, g) for i, g in observations if np.isfinite(g)]
            values["grade_lag_1"][pos] = previous_grade
            values["grade_lag_2"][pos] = prev2_grade
            values["history_terms_available"][pos] = len(observations)
            if observations:
                grades = np.array([g for _, g in observations], dtype=float)
                values["grade_mean_3"][pos] = grades.mean()
                if len(grades) >= 2:
                    values["grade_std_3"][pos] = grades.std(ddof=0)
                    terms = np.array([i for i, _ in observations], dtype=float)
                    centered = terms - terms.mean()
                    values["grade_slope_3"][pos] = (
                        np.dot(centered, grades - grades.mean()) / np.dot(centered, centered)
                    )
            values["grade_change_1"][pos] = _delta(now_grade, previous_grade)
            for source, output in [
                ("attendance_rate", "attendance_change_1"),
                ("assignment_completion_rate", "assignment_change_1"),
                ("credits_completion_rate", "credits_change_1"),
            ]:
                values[output][pos] = _delta(
                    row[source], prior[source] if prior is not None else np.nan
                )
    for column, data in values.items():
        result[column] = data
    return result
