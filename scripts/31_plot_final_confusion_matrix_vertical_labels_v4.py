# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 31_plot_final_confusion_matrix_vertical_labels.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.

# ============================================================
# 31_plot_final_confusion_matrix.py
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# final_independent_test_predictions_v4.csv
#
# Analysis step.
# confusion_matrix_xgb_stable50_v4.png
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

# Analysis step.
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
    r"\final_independent_test_predictions_v4.csv"
)

figure_dir = (
    str(FINAL_DIR) + 
    r"\05_results\figures"
)

os.makedirs(
    figure_dir,
    exist_ok=True
)

output_file = os.path.join(
    figure_dir,
    "confusion_matrix_xgb_stable50_v4.png"
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

print("=" * 90)
print("XGBoost最终独立测试集归一化混淆矩阵")
print("=" * 90)

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)

print("读取文件：")
print(input_file)

print("\n样品总数：", len(df))


# ============================================================
# Analysis step.
# ============================================================

# Analysis step.
# Analysis step.


true_candidates = [
    "TRUE_LABEL",
    "TRUE",
    "Y_TRUE",
    "LABEL"
]

pred_candidates = [
    "XGB_PRED_LABEL",
    "PRED_LABEL",
    "XGB_PRED",
    "PRED"
]


true_col = None
pred_col = None


for col in true_candidates:
    if col in df.columns:
        true_col = col
        break


for col in pred_candidates:
    if col in df.columns:
        pred_col = col
        break


if true_col is None:
    print("\n当前CSV字段：")
    print(df.columns.tolist())

    raise RuntimeError(
        "没有找到真实类别字段。"
    )


if pred_col is None:
    print("\n当前CSV字段：")
    print(df.columns.tolist())

    raise RuntimeError(
        "没有找到XGBoost预测类别字段。"
    )


print("\n真实类别字段：", true_col)
print("预测类别字段：", pred_col)


# ============================================================
# Analysis step.
# ============================================================

# Analysis step.
# Analysis step.

df[true_col] = (
    df[true_col]
    .astype(str)
    .str.strip()
)

df[pred_col] = (
    df[pred_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# Analysis step.
# ============================================================

true_unique = sorted(
    df[true_col]
    .dropna()
    .unique()
)

pred_unique = sorted(
    df[pred_col]
    .dropna()
    .unique()
)


print("\n真实类别：")
print(true_unique)

print("\n预测类别：")
print(pred_unique)


unknown_true = (
    set(true_unique)
    - set(classes)
)

unknown_pred = (
    set(pred_unique)
    - set(classes)
)


if unknown_true:
    raise RuntimeError(
        f"真实标签中发现未知类别：{unknown_true}"
    )


if unknown_pred:
    raise RuntimeError(
        f"预测标签中发现未知类别：{unknown_pred}"
    )


# ============================================================
# Analysis step.
# ============================================================

cm = confusion_matrix(
    df[true_col],
    df[pred_col],
    labels=classes
)


# ============================================================
# Analysis step.
# ============================================================

row_sum = cm.sum(
    axis=1,
    keepdims=True
)


cm_norm = np.divide(
    cm,
    row_sum,
    out=np.zeros_like(
        cm,
        dtype=float
    ),
    where=row_sum != 0
)


# ============================================================
# Analysis step.
# ============================================================

print("\n" + "=" * 90)
print("各真实类别独立测试样品数")
print("=" * 90)

for i, class_name in enumerate(classes):

    print(
        f"{class_name:5s}："
        f"{cm[i].sum():5d}"
    )


print("\n" + "=" * 90)
print("各类别召回率（混淆矩阵对角线）")
print("=" * 90)

for i, class_name in enumerate(classes):

    print(
        f"{class_name:5s}："
        f"{cm_norm[i, i]:.4f} "
        f"({cm_norm[i, i] * 100:.2f}%)"
    )


# ============================================================
# Analysis step.
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.2, 7.6)
)


# Analysis step.
im = ax.imshow(
    cm_norm,
    cmap="Blues",
    vmin=0,
    vmax=1,
    aspect="equal"
)


# ============================================================
# Analysis step.
# ============================================================

positions = np.arange(
    len(classes)
)


ax.set_xticks(
    positions
)

ax.set_yticks(
    positions
)


ax.set_xticklabels(
    classes,
    fontsize=11,
    rotation=30,
    ha="right"
)

ax.set_yticklabels(
    classes,
    fontsize=11
)


ax.set_xlabel(
    "预测构造环境",
    fontsize=13,
    labelpad=10
)

ax.set_ylabel(
    "真\n实\n构\n造\n环\n境",
    fontsize=13,
    labelpad=10,
    rotation=0,
    ha="center",
    va="center"
)




# ============================================================
# Analysis step.
# ============================================================

for i in range(
    len(classes)
):

    for j in range(
        len(classes)
    ):

        count = cm[i, j]

        proportion = cm_norm[i, j]

        percentage = (
            proportion * 100
        )


        
        if count == 0:
            ax.text(
                j,
                i,
                "0",
                ha="center",
                va="center",
                fontsize=8,
                color="0.65"
            )
            continue


        # Analysis step.
        # Analysis step.
        if proportion >= 0.50:
            text_color = "white"
        else:
            text_color = "black"

        # Analysis step.
        if 0 < percentage < 0.1:
            percent_text = "<0.1%"
        else:
            percent_text = f"{percentage:.1f}%"

        ax.text(
            j,
            i,
            f"{percent_text}\n({count})",
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color
        )


# ============================================================
# Analysis step.
# ============================================================

ax.set_xticks(
    np.arange(
        -0.5,
        len(classes),
        1
    ),
    minor=True
)

ax.set_yticks(
    np.arange(
        -0.5,
        len(classes),
        1
    ),
    minor=True
)


ax.grid(
    which="minor",
    color="white",
    linestyle="-",
    linewidth=1.2
)


ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

ax.tick_params(
    axis="both",
    which="major",
    length=0
)


# ============================================================
# Analysis step.
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)


cbar.set_label(
    "真\n实\n类\n别\n内\n样\n品\n比\n例",
    fontsize=12,
    labelpad=10,
    rotation=0,
    ha="center",
    va="center"
)


cbar.ax.tick_params(
    labelsize=10
)


# Analysis step.
ticks = np.linspace(
    0,
    1,
    6
)

cbar.set_ticks(
    ticks
)

cbar.set_ticklabels(
    [
        f"{x * 100:.0f}%"
        for x in ticks
    ]
)


# ============================================================
# Analysis step.
# ============================================================

plt.tight_layout()


# ============================================================
# Analysis step.
# ============================================================

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# Analysis step.
# ============================================================

print("\n" + "=" * 90)
print("混淆矩阵绘制完成")
print("=" * 90)

print("\n保存位置：")
print(output_file)

print(
    "\n说明："
    "行表示真实构造环境，列表示预测构造环境；"
    "每格第一行为该真实类别内所占百分比，"
    "括号内为实际样品数。"
)
