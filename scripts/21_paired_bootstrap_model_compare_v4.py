# AUTO-GENERATED V4 COPY. SOURCE: 21_paired_bootstrap_model_compare.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 21_paired_bootstrap_model_compare.py
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# XGBoost - RandomForest
# XGBoost - RBF-SVM
#
# Analysis step.
# Accuracy
# Balanced Accuracy
# Macro-F1
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
    r"\final_independent_test_predictions_v4.csv"
)

output_file = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
    r"\paired_bootstrap_model_compare_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)


classes = [
    "CAB",
    "IAB",
    "IOAB",
    "BABB",
    "MORB",
    "OIB",
    "OPB",
    "CFB"
]

class_to_int = {
    c: i
    for i, c in enumerate(classes)
}


df["TRUE_ID"] = (
    df["TRUE_LABEL"]
    .map(class_to_int)
    .astype(int)
)

df["XGB_ID"] = (
    df["XGB_PRED_LABEL"]
    .map(class_to_int)
    .astype(int)
)

df["RF_ID"] = (
    df["RF_PRED_LABEL"]
    .map(class_to_int)
    .astype(int)
)

df["SVM_ID"] = (
    df["SVM_PRED_LABEL"]
    .map(class_to_int)
    .astype(int)
)


print("测试样品：", len(df))
print(
    "CV_GROUP_V4：",
    df["CV_GROUP_V4"].nunique()
)


# ============================================================
# Analysis step.
# ============================================================

def metrics(y_true, y_pred):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(8),
                average="macro",
                zero_division=0
            )
    }


# ============================================================
# Analysis step.
# ============================================================

y_true = df["TRUE_ID"].to_numpy()

xgb_point = metrics(
    y_true,
    df["XGB_ID"].to_numpy()
)

rf_point = metrics(
    y_true,
    df["RF_ID"].to_numpy()
)

svm_point = metrics(
    y_true,
    df["SVM_ID"].to_numpy()
)


# ============================================================
# 4. CV_GROUP_V4 bootstrap
# ============================================================

groups = (
    df["CV_GROUP_V4"]
    .drop_duplicates()
    .to_numpy()
)

n_groups = len(groups)

group_indices = {
    g: np.where(
        df["CV_GROUP_V4"].to_numpy() == g
    )[0]
    for g in groups
}


rng = np.random.default_rng(42)

n_bootstrap = 5000

records = []


for b in range(n_bootstrap):

    sampled_groups = rng.choice(
        groups,
        size=n_groups,
        replace=True
    )

    indices = np.concatenate(
        [
            group_indices[g]
            for g in sampled_groups
        ]
    )

    boot = df.iloc[indices]

    yt = boot["TRUE_ID"].to_numpy()

    # Analysis step.
    if len(np.unique(yt)) < 8:
        continue


    mx = metrics(
        yt,
        boot["XGB_ID"].to_numpy()
    )

    mr = metrics(
        yt,
        boot["RF_ID"].to_numpy()
    )

    ms = metrics(
        yt,
        boot["SVM_ID"].to_numpy()
    )


    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1"
    ]:

        records.append({

            "bootstrap": b,

            "comparison":
                "XGBoost-RandomForest",

            "metric": metric,

            "difference":
                mx[metric] - mr[metric]
        })


        records.append({

            "bootstrap": b,

            "comparison":
                "XGBoost-RBF-SVM",

            "metric": metric,

            "difference":
                mx[metric] - ms[metric]
        })


    if (b + 1) % 500 == 0:

        print(
            f"完成 "
            f"{b + 1}/"
            f"{n_bootstrap}"
        )


boot_df = pd.DataFrame(
    records
)


# ============================================================
# Analysis step.
# ============================================================

summary_rows = []


comparisons = {

    "XGBoost-RandomForest":
        (
            xgb_point,
            rf_point
        ),

    "XGBoost-RBF-SVM":
        (
            xgb_point,
            svm_point
        )
}


for comparison, (
    a,
    b
) in comparisons.items():

    print("\n")
    print("=" * 80)
    print(comparison)
    print("=" * 80)


    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1"
    ]:

        values = (
            boot_df.loc[
                (
                    boot_df[
                        "comparison"
                    ] == comparison
                )
                &
                (
                    boot_df[
                        "metric"
                    ] == metric
                ),
                "difference"
            ]
            .to_numpy()
        )


        point_diff = (
            a[metric]
            -
            b[metric]
        )


        lower = np.percentile(
            values,
            2.5
        )

        upper = np.percentile(
            values,
            97.5
        )


        # Analysis step.
        p_lower = np.mean(
            values <= 0
        )

        p_upper = np.mean(
            values >= 0
        )

        p_value = min(
            1.0,
            2 * min(
                p_lower,
                p_upper
            )
        )


        significant = (
            "YES"
            if (
                lower > 0
                or
                upper < 0
            )
            else
            "NO"
        )


        summary_rows.append({

            "comparison":
                comparison,

            "metric":
                metric,

            "difference":
                point_diff,

            "ci_lower":
                lower,

            "ci_upper":
                upper,

            "bootstrap_p":
                p_value,

            "ci_excludes_zero":
                significant
        })


        print(
            f"{metric:20s}: "
            f"Δ={point_diff:+.4f} "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}] "
            f"p≈{p_value:.4f} "
            f"CI排除0={significant}"
        )


# ============================================================
# Analysis step.
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print("\n保存：")
print(output_file)