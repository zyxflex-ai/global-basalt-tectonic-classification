"""Print the principal frozen results without refitting any model."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    validation_rows = read_rows(
        ROOT / "05_results" / "validation_bias" / "random_vs_grouped_cv_metrics_v4.csv"
    )
    frozen_oof = {
        row["scheme"]: row
        for row in validation_rows
        if row["repeat"] == "-1" and row["level"] == "oof" and row["weighting"] == "sample"
    }
    holdout_rows = read_rows(
        ROOT / "05_results" / "final_test" / "final_independent_test_metrics_v4.csv"
    )
    xgb = next(
        row for row in holdout_rows if row["model"] == "XGBoost" and row["level"] == "overall"
    )

    grouped = float(frozen_oof["publication_grouped"]["macro_f1"])
    random = float(frozen_oof["sample_random"]["macro_f1"])
    print("Frozen repository results")
    print(f"  Publication-grouped OOF macro-F1 : {grouped:.4f}")
    print(f"  Sample-random OOF macro-F1       : {random:.4f}")
    print(f"  Apparent inflation                : {random - grouped:.4f}")
    print(f"  XGBoost holdout accuracy          : {float(xgb['accuracy']):.4f}")
    print(f"  XGBoost holdout balanced accuracy : {float(xgb['balanced_accuracy']):.4f}")
    print(f"  XGBoost holdout macro-F1          : {float(xgb['macro_f1']):.4f}")


if __name__ == "__main__":
    main()
