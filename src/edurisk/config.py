"""Research contract: available snapshots and time-safe longitudinal features."""
from .history import BASE_COLUMNS, HISTORY_COLUMNS, LONGITUDINAL_COLUMNS

# v0.2 compares an actual single-term snapshot with a multi-term history.
# The original grade_trend column is deliberately NOT a snapshot feature,
# because it depends on a previous semester. It is recalculated from grades.
FEATURE_COLUMNS = BASE_COLUMNS
TARGET_COLUMN = "next_term_low_performance"
ID_COLUMN = "student_id"
TIME_COLUMN = "term_index"
TRAIN_TERMS = (1, 2, 3)
VALIDATION_TERM = 4
TEST_TERM = 5
RISK_GRADE_THRESHOLD = 3.0  # only a synthetic demonstrator assumption
