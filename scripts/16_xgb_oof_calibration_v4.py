# AUTO-GENERATED V4 COPY. SOURCE: 16_xgb_oof_calibration.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 16_xgb_oof_calibration.py
#
# Analysis step.
#
# Analysis step.
# 1. Multiclass Log Loss
# 2. Multiclass Brier Score
# 3. Top-label ECE
# 4. MCE
# 5. Reliability bins
# Analysis step.
# Analysis step.
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
    log_loss,
    accuracy_score
)
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    r"\05_results\diagnosis"
    r"\xgb_tuned_oof_predictions_v4.csv"
)

result_dir = (
    str(FINAL_DIR) + 
    r"\05_results\calibration"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

output_file = os.path.join(
    result_dir,
    "xgb_oof_calibration_bins_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

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


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)


print("=" * 80)
print("读取 Tuned XGBoost OOF")
print("=" * 80)

print(
    "OOF样品数：",
    len(df)
)


# ============================================================
# Analysis step.
# ============================================================

prob_cols = [
    f"PROB_{c}"
    for c in classes
]


missing_cols = [
    c
    for c in prob_cols
    if c not in df.columns
]

if missing_cols:

    raise RuntimeError(
        "缺少概率字段："
        + str(missing_cols)
    )


proba = (
    df[prob_cols]
    .to_numpy(dtype=float)
)

y_true = (
    df["TRUE_ID"]
    .to_numpy(dtype=int)
)

y_pred = (
    df["PRED_ID"]
    .to_numpy(dtype=int)
)


# ============================================================
# Analysis step.
# ============================================================

row_sum = (
    proba.sum(axis=1)
)

print("\n概率行和：")

print(
    "最小：",
    row_sum.min()
)

print(
    "最大：",
    row_sum.max()
)

print(
    "平均：",
    row_sum.mean()
)


if not np.allclose(
    row_sum,
    1.0,
    atol=1e-5
):

    raise RuntimeError(
        "部分预测概率之和不等于1。"
    )


# ============================================================
# Analysis step.
# ============================================================

y_onehot = np.zeros(
    (
        len(y_true),
        len(classes)
    ),
    dtype=float
)

y_onehot[
    np.arange(len(y_true)),
    y_true
] = 1.0


# ============================================================
# 6. Overall Accuracy
# ============================================================

acc = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# 7. Multiclass Log Loss
# ============================================================

ll = log_loss(
    y_true,
    proba,
    labels=np.arange(
        len(classes)
    )
)


# ============================================================
# 8. Multiclass Brier Score
#
# Analysis step.
# BS = mean_i sum_k (p_ik - y_ik)^2
#
# Analysis step.
# ============================================================

squared_error = (
    proba
    -
    y_onehot
) ** 2


brier_sum = np.mean(
    np.sum(
        squared_error,
        axis=1
    )
)

brier_per_class = np.mean(
    squared_error
)


# ============================================================
# 9. Top-label confidence
# ============================================================

confidence = np.max(
    proba,
    axis=1
)

prediction = np.argmax(
    proba,
    axis=1
)

correct = (
    prediction
    ==
    y_true
).astype(int)


# Analysis step.
if not np.array_equal(
    prediction,
    y_pred
):

    print(
        "\n警告：PRED_ID 与概率argmax存在不一致。"
    )


# ============================================================
# Analysis step.
#
# equal-width bins
# ECE = sum_b (n_b/N) * |acc_b - conf_b|
# ============================================================

def calibration_bins(
    confidence,
    correct,
    n_bins=10
):

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    rows = []

    total_n = len(
        confidence
    )


    for i in range(
        n_bins
    ):

        lower = edges[i]
        upper = edges[i + 1]


        if i == n_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )


        n = int(
            mask.sum()
        )


        if n == 0:

            mean_conf = np.nan
            bin_acc = np.nan
            gap = np.nan
            contribution = 0.0

        else:

            mean_conf = float(
                confidence[
                    mask
                ].mean()
            )

            bin_acc = float(
                correct[
                    mask
                ].mean()
            )

            gap = abs(
                bin_acc
                -
                mean_conf
            )

            contribution = (
                n
                /
                total_n
                *
                gap
            )


        rows.append({

            "bin": i + 1,

            "lower": lower,

            "upper": upper,

            "n": n,

            "coverage": (
                n / total_n
            ),

            "mean_confidence":
                mean_conf,

            "accuracy":
                bin_acc,

            "gap":
                gap,

            "ece_contribution":
                contribution
        })


    table = pd.DataFrame(
        rows
    )


    ece = (
        table[
            "ece_contribution"
        ]
        .sum()
    )


    nonempty = (
        table[
            "n"
        ] > 0
    )


    mce = (
        table.loc[
            nonempty,
            "gap"
        ]
        .max()
    )


    return (
        table,
        ece,
        mce
    )


# ============================================================
# 11. Overall top-label ECE
# ============================================================

cal_table, ece, mce = (
    calibration_bins(
        confidence,
        correct,
        n_bins=10
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("OOF 概率总体诊断")
print("=" * 80)


print(
    f"Accuracy                 : "
    f"{acc:.4f}"
)

print(
    f"Multiclass Log Loss      : "
    f"{ll:.4f}"
)

print(
    f"Brier Score (sum classes): "
    f"{brier_sum:.4f}"
)

print(
    f"Brier Score (/8 classes): "
    f"{brier_per_class:.4f}"
)

print(
    f"Top-label ECE (10 bins)  : "
    f"{ece:.4f}"
)

print(
    f"MCE                      : "
    f"{mce:.4f}"
)

print(
    f"Mean confidence          : "
    f"{confidence.mean():.4f}"
)

print(
    f"Accuracy - MeanConf      : "
    f"{acc - confidence.mean():+.4f}"
)


# ============================================================
# 13. Reliability bins
# ============================================================

print("\n")
print("=" * 80)
print("Reliability bins")
print("=" * 80)


show = cal_table.copy()

for c in [
    "coverage",
    "mean_confidence",
    "accuracy",
    "gap",
    "ece_contribution"
]:

    show[c] = (
        show[c]
        .round(4)
    )


print(
    show.to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
#
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("不同 MAX_PROB 阈值下的预测表现")
print("=" * 80)


threshold_rows = []


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
]:

    mask = (
        confidence
        >=
        threshold
    )

    n = int(
        mask.sum()
    )


    if n == 0:

        threshold_acc = np.nan

    else:

        threshold_acc = float(
            correct[
                mask
            ].mean()
        )


    coverage = (
        n / len(df)
    )


    threshold_rows.append({

        "threshold": threshold,

        "n": n,

        "coverage": coverage,

        "accuracy": threshold_acc,

        "error_rate":
            (
                1 - threshold_acc
                if n > 0
                else np.nan
            )
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


print(
    threshold_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("低置信度样品")
print("=" * 80)


for threshold in [
    0.50,
    0.60,
    0.70
]:

    mask = (
        confidence
        <
        threshold
    )

    n = int(
        mask.sum()
    )


    if n == 0:

        low_acc = np.nan

    else:

        low_acc = float(
            correct[
                mask
            ].mean()
        )


    print(
        f"MAX_PROB < {threshold:.2f}: "
        f"n={n}, "
        f"占比={n/len(df):.2%}, "
        f"Accuracy={low_acc:.4f}"
    )


# ============================================================
# Analysis step.
#
# Analysis step.
# p(class=k)
# Analysis step.
# ============================================================

def binary_ece(
    probability,
    truth,
    n_bins=10
):

    edges = np.linspace(
        0,
        1,
        n_bins + 1
    )

    n_total = len(
        truth
    )

    ece_value = 0.0


    for i in range(
        n_bins
    ):

        lower = edges[i]
        upper = edges[i + 1]


        if i == n_bins - 1:

            mask = (
                (probability >= lower)
                &
                (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                &
                (probability < upper)
            )


        n = mask.sum()


        if n == 0:

            continue


        mean_prob = (
            probability[
                mask
            ].mean()
        )

        observed = (
            truth[
                mask
            ].mean()
        )


        ece_value += (
            n
            /
            n_total
            *
            abs(
                observed
                -
                mean_prob
            )
        )


    return ece_value


print("\n")
print("=" * 80)
print("各类别 One-vs-Rest ECE")
print("=" * 80)


class_ece_rows = []


for k, class_name in enumerate(
    classes
):

    truth_binary = (
        y_true == k
    ).astype(int)

    class_probability = (
        proba[:, k]
    )


    value = binary_ece(
        class_probability,
        truth_binary,
        n_bins=10
    )


    class_ece_rows.append({
        "class": class_name,
        "ece": value
    })


class_ece_df = pd.DataFrame(
    class_ece_rows
)


print(
    class_ece_df
    .sort_values(
        "ece",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
#
# Analysis step.
# ============================================================

cal_table.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 80)
print("概率校准诊断完成")
print("=" * 80)

print(
    "Reliability bin 保存："
)

print(
    output_file
)

print(
    "\n独立测试集未读取。"
)

print(
    "本步骤没有拟合任何校准器。"
)