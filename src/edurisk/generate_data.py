"""Deterministic longitudinal synthetic student-term observations.

No real student data, no personal attributes; generating a dataset is NOT evidence
that a method will work on an educational institution's real records.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic(n_students: int = 420, n_terms: int = 6, seed: int = 2026) -> pd.DataFrame:
    if n_students < 2 or n_terms < 6:
        raise ValueError("Need at least 2 students and 6 terms for the fixed time split")
    rng = np.random.default_rng(seed)
    records = []
    for student in range(1, n_students + 1):
        latent_preparation = float(rng.normal(0, 0.85))
        persistent_engagement = float(rng.normal(0, 0.65))
        previous_grade = None
        history = []
        for t in range(1, n_terms + 1):
            current_shock = float(rng.normal(0, 0.6))
            e = persistent_engagement + 0.30 * latent_preparation + 0.40 * current_shock
            attendance = float(np.clip(0.73 + 0.18 * e + rng.normal(0, 0.09), 0.08, 1))
            assignments = float(np.clip(0.69 + 0.19 * e + rng.normal(0, 0.11), 0.05, 1))
            late = float(np.clip(0.34 - 0.19 * e + rng.normal(0, 0.10), 0, 1))
            credits = float(np.clip(0.63 + 0.17 * e + rng.normal(0, 0.10), 0, 1))
            score = (3.38 + 0.54 * latent_preparation + 0.35 * e + 0.25 * (attendance - 0.7)
                     + 0.22 * (assignments - 0.7) - 0.28 * late + rng.normal(0, 0.37))
            grade = float(np.clip(score, 2.0, 5.0))
            history.append({
                "student_id": f"SYN-{student:04d}",
                "term_index": t,
                "attendance_rate": round(attendance, 4),
                "assignment_completion_rate": round(assignments, 4),
                "late_submission_rate": round(late, 4),
                "current_avg_grade": round(grade, 4),
                "grade_trend": round(grade - previous_grade, 4) if previous_grade is not None else 0.0,
                "credits_completion_rate": round(credits, 4),
            })
            previous_grade = grade
        for t in range(n_terms - 1):
            # Label for row t depends ONLY on the future observed synthetic grade t+1.
            # It is never included in model feature columns.
            row = dict(history[t])
            row["next_term_low_performance"] = int(history[t + 1]["current_avg_grade"] < 3.0)
            records.append(row)
    return pd.DataFrame.from_records(records).sort_values(["term_index", "student_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic longitudinal educational records")
    parser.add_argument("--out", default="data/synthetic_demo.csv")
    parser.add_argument("--students", type=int, default=420)
    parser.add_argument("--terms", type=int, default=6, help="Number of simulated terms (>=6)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic(n_students=args.students, n_terms=args.terms, seed=args.seed)
    frame.to_csv(out, index=False)
    print(f"Synthetic-only: generated {len(frame)} rows for {args.students} students -> {out}")


if __name__ == "__main__":
    main()
