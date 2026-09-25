"""Reviewer-requested sensitivity analyses for the Chemical Geology submission.

This script does not change model training. It audits the frozen predictions under
three transparent restrictions: publication size, direct label evidence, and
major-oxide quality-control subsets. The output is intended for Supplementary
Information and for cautious qualification of the main text.
"""

from pathlib import Path
import sys

MODEL_DEPS = Path(__file__).resolve().parents[1] / "modeldeps"
sys.path.insert(0, str(MODEL_DEPS))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


FINAL = Path(__file__).resolve().parents[1]
MERGED = FINAL / "02_merged" / "Basalt_8classes_paperid_v4.csv"
TEST = FINAL / "03_model_data" / "Basalt_test_stable50_main_v4.csv"
PRED = FINAL / "05_results" / "final_test" / "final_independent_test_predictions_v4.csv"
CV_PRED = FINAL / "05_results" / "validation_bias" / "random_vs_grouped_cv_primary_predictions_v4.csv"
OUT = FINAL / "05_results" / "chemical_geology_sensitivities"

MAJOR_OXIDES = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FE_TOTAL(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
]


def score(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    labels = sorted(pd.unique(y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def publication_size_sensitivity(pred: pd.DataFrame) -> pd.DataFrame:
    by_pub = (
        pred.groupby("CV_GROUP_V4", observed=True)
        .agg(n_samples=("XGB_CORRECT", "size"), accuracy=("XGB_CORRECT", "mean"))
        .reset_index()
    )
    rows = []
    for threshold in (1, 2, 3, 5, 10, 20):
        x = by_pub.loc[by_pub["n_samples"] >= threshold]
        rows.append(
            {
                "minimum_samples_per_publication": threshold,
                "n_publications": len(x),
                "n_samples": int(x["n_samples"].sum()),
                "publication_equal_mean_accuracy": x["accuracy"].mean(),
                "publication_accuracy_median": x["accuracy"].median(),
                "n_publications_accuracy_below_0_5": int((x["accuracy"] < 0.5).sum()),
                "percent_publications_accuracy_below_0_5": 100 * (x["accuracy"] < 0.5).mean(),
                "n_publications_accuracy_zero": int((x["accuracy"] == 0).sum()),
                "percent_publications_accuracy_zero": 100 * (x["accuracy"] == 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def direct_label_evaluation(merged: pd.DataFrame, cv_pred: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    label_map = merged[["GROUP_ID", "TECTONIC SETTING", "LABEL"]].copy()
    if label_map["GROUP_ID"].duplicated().any():
        raise RuntimeError("GROUP_ID must be unique in the merged data.")
    label_map["direct_label"] = label_map["TECTONIC SETTING"].fillna("").astype(str).str.strip().ne("")

    rows = []
    cv = cv_pred.merge(label_map[["GROUP_ID", "direct_label"]], on="GROUP_ID", how="left", validate="one_to_one")
    cv = cv.loc[cv["direct_label"]].copy()
    for design, col in (
        ("publication_grouped_oof", "PRED_PUBLICATION_GROUPED"),
        ("sample_random_oof", "PRED_SAMPLE_RANDOM"),
    ):
        metrics = score(cv["LABEL"], cv[col])
        rows.append(
            {
                "evaluation_set": "development_direct_label_rows",
                "design": design,
                "n_samples": len(cv),
                "n_publications": cv["CV_GROUP_V4"].nunique(),
                "n_classes": cv["LABEL"].nunique(),
                "classes": ";".join(sorted(cv["LABEL"].unique())),
                **metrics,
            }
        )

    hold = pred.merge(label_map[["GROUP_ID", "direct_label"]], on="GROUP_ID", how="left", validate="one_to_one")
    hold = hold.loc[hold["direct_label"]].copy()
    metrics = score(hold["TRUE_LABEL"], hold["XGB_PRED_LABEL"])
    rows.append(
        {
            "evaluation_set": "frozen_holdout_direct_label_rows",
            "design": "frozen_publication_disjoint",
            "n_samples": len(hold),
            "n_publications": hold["CV_GROUP_V4"].nunique(),
            "n_classes": hold["TRUE_LABEL"].nunique(),
            "classes": ";".join(sorted(hold["TRUE_LABEL"].unique())),
            **metrics,
        }
    )
    return pd.DataFrame(rows)


def oxide_qc_overview(merged: pd.DataFrame) -> pd.DataFrame:
    x = merged[MAJOR_OXIDES].apply(pd.to_numeric, errors="coerce")
    complete = x.notna().all(axis=1)
    total = x.sum(axis=1, min_count=len(MAJOR_OXIDES))
    valid = total.loc[complete]
    return pd.DataFrame(
        [
            {
                "n_all_samples": len(merged),
                "n_complete_ten_oxide_suite": int(complete.sum()),
                "percent_complete_ten_oxide_suite": 100 * complete.mean(),
                "oxide_total_mean": valid.mean(),
                "oxide_total_median": valid.median(),
                "oxide_total_q1": valid.quantile(0.25),
                "oxide_total_q3": valid.quantile(0.75),
                "oxide_total_min": valid.min(),
                "oxide_total_max": valid.max(),
                "n_total_outside_95_105": int((~valid.between(95, 105)).sum()),
                "percent_total_outside_95_105": 100 * (~valid.between(95, 105)).mean(),
                "n_total_outside_90_110": int((~valid.between(90, 110)).sum()),
                "percent_total_outside_90_110": 100 * (~valid.between(90, 110)).mean(),
            }
        ]
    )


def oxide_qc_holdout_sensitivity(test: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    cols = ["GROUP_ID", "CV_GROUP_V4", "LABEL", *MAJOR_OXIDES]
    joined = pred.merge(test[cols], on=["GROUP_ID", "CV_GROUP_V4"], how="left", validate="one_to_one")
    x = joined[MAJOR_OXIDES].apply(pd.to_numeric, errors="coerce")
    complete = x.notna().all(axis=1)
    total = x.sum(axis=1, min_count=len(MAJOR_OXIDES))
    subsets = {
        "all_frozen_holdout": np.ones(len(joined), dtype=bool),
        "complete_ten_oxide_suite": complete,
        "complete_and_total_95_105": complete & total.between(95, 105),
        "complete_and_total_90_110": complete & total.between(90, 110),
    }
    rows = []
    for name, mask in subsets.items():
        part = joined.loc[mask]
        metrics = score(part["TRUE_LABEL"], part["XGB_PRED_LABEL"])
        rows.append(
            {
                "subset": name,
                "n_samples": len(part),
                "n_publications": part["CV_GROUP_V4"].nunique(),
                "n_classes": part["TRUE_LABEL"].nunique(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(MERGED, low_memory=False)
    test = pd.read_csv(TEST, low_memory=False)
    pred = pd.read_csv(PRED, low_memory=False)
    cv_pred = pd.read_csv(CV_PRED, low_memory=False)

    outputs = {
        "publication_size_sensitivity.csv": publication_size_sensitivity(pred),
        "direct_label_evaluation_sensitivity.csv": direct_label_evaluation(merged, cv_pred, pred),
        "major_oxide_qc_overview.csv": oxide_qc_overview(merged),
        "major_oxide_qc_holdout_sensitivity.csv": oxide_qc_holdout_sensitivity(test, pred),
    }
    for name, table in outputs.items():
        table.to_csv(OUT / name, index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
