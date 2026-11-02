"""Carve a subject-disjoint TEST split out of an existing metadata CSV.

All rows of a subject (column ``id``) go to the same side, so no person appears in both.
Default: take 50% of the validation subjects as test, which keeps the training set (and
therefore training cost) unchanged; the new val is used for early stopping and test is
scored once at the end.

Subjects are stratified by their highest 3-grade forehead/glabella wrinkle label by default.

  python notebooks/dy/make_subject_test_split.py --source data/processed/dy_forehead_glabella_val_metadata.csv \
      --fraction 0.5 --out-keep data/processed/dy_fg_val2_metadata.csv \
      --out-test data/processed/dy_fg_test_metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def three_grade(value: float) -> int:
    return 0 if value <= 1 else (1 if value <= 3 else 2)


def subject_strata(frame: pd.DataFrame, ids: pd.Series, columns: list[str]) -> pd.Series:
    """Per-subject key: the subject's highest 3-grade in each column (missing -> -1)."""
    keys = {}
    for subject, group in frame.assign(_id=ids).groupby("_id"):
        key = []
        for column in columns:
            values = group[column].dropna()
            key.append(three_grade(float(values.max())) if len(values) else -1)
        keys[subject] = tuple(key)
    return pd.Series(keys)


def split_by_subject(frame: pd.DataFrame, fraction: float, seed: int, stratify: list[str] | str | None = None):
    ids = frame["id"].astype("string").str.strip()
    subjects = np.array(sorted(ids.unique()))
    rng = np.random.default_rng(seed)
    columns = [stratify] if isinstance(stratify, str) else list(stratify or [])
    columns = [c for c in columns if c in frame]
    if columns:
        strata = subject_strata(frame, ids, columns)
        test_subjects, carry = set(), 0.0
        for _, group in strata.groupby(strata):
            names = np.array(sorted(group.index))
            exact = len(names) * fraction + carry       # carry rounding so the total stays on target
            take = int(np.floor(exact + 0.5))
            carry = exact - take
            test_subjects.update(rng.choice(names, min(take, len(names)), replace=False).tolist())
    else:
        take = int(round(len(subjects) * fraction))
        test_subjects = set(rng.choice(subjects, take, replace=False).tolist())
    is_test = ids.isin(test_subjects)
    return frame[~is_test].reset_index(drop=True), frame[is_test].reset_index(drop=True)


def grade_table(frame: pd.DataFrame, columns: list[str]) -> str:
    parts = []
    for column in columns:
        if column in frame:
            counts = frame[column].dropna().map(three_grade).value_counts().reindex([0, 1, 2], fill_value=0)
            parts.append(f"{column}={counts.tolist()}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--stratify", nargs="*", default=["label_forehead_wrinkle", "label_glabella_wrinkle"],
                        help="grade columns used to balance subjects (subject's max 3-grade per column); none to disable")
    parser.add_argument("--out-keep", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be between 0 and 1")
    frame = pd.read_csv(args.source, dtype={"id": "string"})
    keep, test = split_by_subject(frame, args.fraction, args.seed, args.stratify)
    overlap = set(keep["id"].str.strip()) & set(test["id"].str.strip())
    if overlap or keep.empty or test.empty:
        raise SystemExit(f"Bad split: overlap={len(overlap)} keep={len(keep)} test={len(test)}")
    args.out_keep.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(args.out_keep, index=False)
    test.to_csv(args.out_test, index=False)
    report = ["label_forehead_pigmentation", "label_forehead_wrinkle", "label_glabella_wrinkle"]
    print(f"keep: {len(keep)} rows / {keep['id'].nunique()} subjects -> {args.out_keep}  [{grade_table(keep, report)}]")
    print(f"test: {len(test)} rows / {test['id'].nunique()} subjects -> {args.out_test}  [{grade_table(test, report)}]")


if __name__ == "__main__":
    main()
