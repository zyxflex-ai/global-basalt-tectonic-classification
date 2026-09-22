from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "05_results" / "final_test" / "final_independent_test_predictions_v4.csv"
OUT_DIR = ROOT / "05_results" / "sensitivity"

SCHEMES = {
    "eight_class_original": {},
    "merge_IAB_IOAB": {"IAB": "ARC_OCEANIC", "IOAB": "ARC_OCEANIC"},
    "merge_BABB_MORB": {"BABB": "RIDGE_BACKARC", "MORB": "RIDGE_BACKARC"},
    "merge_both_boundaries": {
        "IAB": "ARC_OCEANIC", "IOAB": "ARC_OCEANIC",
        "BABB": "RIDGE_BACKARC", "MORB": "RIDGE_BACKARC",
    },
}


def transform(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return values.map(lambda value: mapping.get(str(value), str(value)))


def metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    labels = sorted(set(true) | set(pred))
    return {
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        "n_classes": len(labels),
    }


def main() -> None:
    data = pd.read_csv(INPUT, low_memory=False)
    group_indices = {
        group: values.index.to_numpy()
        for group, values in data.groupby("CV_GROUP_V4", observed=True)
    }
    groups = np.asarray(list(group_indices), dtype=object)
    rng = np.random.default_rng(20260922)
    rows = []
    for name, mapping in SCHEMES.items():
        true = transform(data["TRUE_LABEL"], mapping)
        pred = transform(data["XGB_PRED_LABEL"], mapping)
        point = metrics(true, pred)
        draws = []
        for _ in range(500):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            indices = np.concatenate([group_indices[group] for group in sampled])
            draws.append(metrics(true.iloc[indices], pred.iloc[indices]))
        boot = pd.DataFrame(draws)
        rows.append({
            "scheme": name,
            "mapping": "|".join(f"{key}->{value}" for key, value in mapping.items()) or "none",
            **point,
            "accuracy_ci_low": boot["accuracy"].quantile(0.025),
            "accuracy_ci_high": boot["accuracy"].quantile(0.975),
            "balanced_accuracy_ci_low": boot["balanced_accuracy"].quantile(0.025),
            "balanced_accuracy_ci_high": boot["balanced_accuracy"].quantile(0.975),
            "macro_f1_ci_low": boot["macro_f1"].quantile(0.025),
            "macro_f1_ci_high": boot["macro_f1"].quantile(0.975),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "label_boundary_collapse_sensitivity_v9.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
