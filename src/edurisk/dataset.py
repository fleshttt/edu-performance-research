"""Dataset validation and chronological split; no random mixing of semesters."""
import numpy as np
import pandas as pd
from .config import (FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN, TIME_COLUMN,
                     TRAIN_TERMS, VALIDATION_TERM, TEST_TERM)


RATE_COLUMNS = ["attendance_rate", "assignment_completion_rate", "late_submission_rate",
                "credits_completion_rate"]


def validate_dataset(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, ID_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df[[ID_COLUMN, TIME_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Dataset or required student/term/target values are empty")
    if df.duplicated([ID_COLUMN, TIME_COLUMN]).any():
        raise ValueError("Duplicate student_id + term_index entries")
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary 0/1")
    for c in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Feature must be numeric: {c}")
        numeric = df[c].to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(f"Infinite features not allowed: {c}")
    for c in RATE_COLUMNS:
        values = df[c].dropna()
        if not values.between(0, 1).all():
            raise ValueError(f"Out-of-range rate in {c}")
    if not df["current_avg_grade"].dropna().between(2, 5).all():
        raise ValueError("current_avg_grade must use example's 2–5 grade scale")
    if not df[TIME_COLUMN].isin([*TRAIN_TERMS, VALIDATION_TERM, TEST_TERM]).all():
        raise ValueError("Dataset includes incompatible terms; expected 1..5")


def split_chronologically(df: pd.DataFrame):
    """Train 1..3, validate 4, test 5. Students may recur across terms.

    This estimates forecasting for returning students, not transfer to unseen students.
    Training at term 4 includes labels from term 3 (outcome measured at term 4).
    """
    validate_dataset(df)
    train = df[df[TIME_COLUMN].isin(TRAIN_TERMS)].copy()
    validation = df[df[TIME_COLUMN] == VALIDATION_TERM].copy()
    test = df[df[TIME_COLUMN] == TEST_TERM].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("All chronological partitions must have observations")
    assert train[TIME_COLUMN].max() < validation[TIME_COLUMN].min() < test[TIME_COLUMN].min()
    return train, validation, test
