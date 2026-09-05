# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 32_plot_shap_global_importance.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.

# ============================================================
# 32_plot_shap_importance_global_classes.py
#
# Stable-50 XGBoost
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# shap_importance_independent_test_v4.csv
#
# Analysis step.
# Analysis step.
# ============================================================


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    r"\05_results\shap"
    r"\shap_importance_independent_test_v4.csv"
)

output_dir = (
    str(FINAL_DIR) + 
    r"\05_results\figures\shap_importance"
)

os.makedirs(
    output_dir,
    exist_ok=True
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

feature_name_map = {

    "SIO2(WT%)": r"SiO$_2$",
    "TIO2(WT%)": r"TiO$_2$",
    "AL2O3(WT%)": r"Al$_2$O$_3$",
    "FE_TOTAL(WT%)": "全铁",
    "CAO(WT%)": "CaO",
    "MGO(WT%)": "MgO",
    "MNO(WT%)": "MnO",
    "K2O(WT%)": r"K$_2$O",
    "NA2O(WT%)": r"Na$_2$O",
    "P2O5(WT%)": r"P$_2$O$_5$",

    "V(PPM)": "V",
    "CR(PPM)": "Cr",
    "NI(PPM)": "Ni",
    "RB(PPM)": "Rb",
    "SR(PPM)": "Sr",
    "Y(PPM)": "Y",
    "ZR(PPM)": "Zr",
    "NB(PPM)": "Nb",
    "BA(PPM)": "Ba"
}


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)


print("=" * 100)
print("Stable-50 XGBoost SHAP重要性绘图")
print("=" * 100)

print("\n输入文件：")
print(input_file)

print("\n数据维度：")
print(df.shape)

print("\nCSV字段：")
for col in df.columns:
    print(" -", col)


# ============================================================
# Analysis step.
# ============================================================

feature_candidates = [
    "feature",
    "Feature",
    "FEATURE",
    "特征",
    "variable",
    "Variable"
]


feature_col = None


for col in feature_candidates:

    if col in df.columns:

        feature_col = col
        break


if feature_col is None:

    # Analysis step.
    feature_col = df.columns[0]


print("\n识别到的特征名称列：")
print(feature_col)


# ============================================================
# Analysis step.
# ============================================================

global_col = None


global_candidates = [
    "global_mean_abs_shap",
    "mean_abs_shap",
    "GLOBAL_MEAN_ABS_SHAP",
    "MeanAbsSHAP",
    "GLOBAL",
    "Global",
    "global"
]


for col in global_candidates:

    if col in df.columns:

        global_col = col
        break


# Analysis step.
# Analysis step.
if global_col is None:

    for col in df.columns:

        name = col.lower()

        if (
            "shap" in name
            and "mean" in name
            and "abs" in name
            and not any(
                c.lower() in name
                for c in classes
            )
        ):

            global_col = col
            break


# ============================================================
# Analysis step.
# ============================================================

class_cols = {}


for tectonic_class in classes:

    candidates = []


    for col in df.columns:

        upper_col = col.upper()

        # Analysis step.
        tokens = re.split(
            r"[^A-Z0-9]+",
            upper_col
        )

        if tectonic_class in tokens:

            candidates.append(col)


    # Analysis step.
    shap_candidates = [
        x for x in candidates
        if "SHAP" in x.upper()
    ]


    if len(shap_candidates) > 0:

        class_cols[tectonic_class] = shap_candidates[0]

    elif len(candidates) > 0:

        class_cols[tectonic_class] = candidates[0]


# ============================================================
# Analysis step.
# ============================================================

print("\n全局SHAP列：")
print(global_col)

print("\n类别SHAP列：")

for c in classes:

    print(
        f"{c:5s} -> "
        f"{class_cols.get(c)}"
    )


if global_col is None:

    raise RuntimeError(
        "\n没有自动找到全局mean(|SHAP|)字段。"
        "\n请把上方打印出来的CSV字段发给我，我帮你对应。"
    )


missing_class_cols = [
    c for c in classes
    if c not in class_cols
]


if len(missing_class_cols) > 0:

    raise RuntimeError(
        "\n以下类别没有自动识别到SHAP字段："
        + str(missing_class_cols)
        +
        "\n请把上方CSV字段列表发给我。"
    )


# ============================================================
# Analysis step.
# ============================================================

df[feature_col] = (
    df[feature_col]
    .astype(str)
    .str.strip()
)


# Analysis step.
df["显示名称"] = (
    df[feature_col]
    .map(feature_name_map)
)


# Analysis step.
# Analysis step.
df["显示名称"] = (
    df["显示名称"]
    .fillna(
        df[feature_col]
    )
)


# Analysis step.
needed_cols = [
    global_col
] + [
    class_cols[c]
    for c in classes
]


for col in needed_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# Analysis step.
# ============================================================

def plot_shap_bar(
    data,
    value_col,
    title_text,
    output_name,
    top_n=19
):

    plot_df = (
        data[
            ["显示名称", value_col]
        ]
        .dropna()
        .copy()
    )


    # Analysis step.
    plot_df = (
        plot_df
        .sort_values(
            value_col,
            ascending=False
        )
        .head(top_n)
    )


    # Analysis step.
    plot_df = (
        plot_df
        .sort_values(
            value_col,
            ascending=True
        )
    )


    fig_height = max(
        5.5,
        0.34 * len(plot_df) + 1.6
    )


    fig, ax = plt.subplots(
        figsize=(7.4, fig_height)
    )


    bars = ax.barh(
        plot_df["显示名称"],
        plot_df[value_col]
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    ax.set_xlabel(
        "平均绝对SHAP值",
        fontsize=12
    )

    ax.set_ylabel(
        "地球化学特征",
        fontsize=12
    )


    # Analysis step.
    # Analysis step.
    ax.set_title(
        title_text,
        fontsize=13,
        pad=10
    )


    ax.tick_params(
        axis="x",
        labelsize=10
    )

    ax.tick_params(
        axis="y",
        labelsize=10
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    xmax = (
        plot_df[value_col]
        .max()
    )


    if pd.isna(xmax) or xmax <= 0:

        xmax = 1


    for bar, value in zip(
        bars,
        plot_df[value_col]
    ):

        ax.text(
            value + xmax * 0.012,
            bar.get_y()
            + bar.get_height() / 2,

            f"{value:.3f}",

            va="center",
            ha="left",
            fontsize=8.5
        )


    ax.set_xlim(
        0,
        xmax * 1.16
    )


    # Analysis step.
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)


    # Analysis step.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    plt.tight_layout()


    output_path = os.path.join(
        output_dir,
        output_name
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "保存：",
        output_path
    )


# ============================================================
# Analysis step.
# ============================================================

print("\n" + "=" * 100)
print("绘制全局SHAP重要性")
print("=" * 100)


plot_shap_bar(

    data=df,

    value_col=global_col,

    title_text="XGBoost模型全局SHAP特征重要性",

    output_name=(
        "shap_global_importance_stable50_v4.png"
    ),

    top_n=19
)


# ============================================================
# Analysis step.
# ============================================================

print("\n" + "=" * 100)
print("绘制8类构造环境SHAP重要性")
print("=" * 100)


for tectonic_class in classes:

    plot_shap_bar(

        data=df,

        value_col=class_cols[
            tectonic_class
        ],

        title_text=(
            f"{tectonic_class}类别SHAP特征重要性"
        ),

        output_name=(
            f"shap_{tectonic_class}_importance_stable50_v4.png"
        ),

        top_n=19
    )


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 100)
print("各类别SHAP重要性前5位")
print("=" * 100)


# Analysis step.
global_top5 = (
    df[
        ["显示名称", global_col]
    ]
    .sort_values(
        global_col,
        ascending=False
    )
    .head(5)
)


print("\n全局：")

for rank, (_, row) in enumerate(
    global_top5.iterrows(),
    start=1
):

    print(
        f"{rank}. "
        f"{row['显示名称']} "
        f"{row[global_col]:.4f}"
    )


# Analysis step.
for tectonic_class in classes:

    col = class_cols[
        tectonic_class
    ]


    top5 = (
        df[
            ["显示名称", col]
        ]
        .sort_values(
            col,
            ascending=False
        )
        .head(5)
    )


    print(
        f"\n{tectonic_class}："
    )


    for rank, (_, row) in enumerate(
        top5.iterrows(),
        start=1
    ):

        print(
            f"{rank}. "
            f"{row['显示名称']} "
            f"{row[col]:.4f}"
        )


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 100)
print("SHAP重要性图绘制完成")
print("=" * 100)

print(
    "\n共生成9张PNG图片："
)

print(
    "1张全局图 + 8张类别特异图"
)

print(
    "\n保存目录："
)

print(
    output_dir
)

print(
    "\n注意："
    "平均绝对SHAP值反映模型贡献强度，"
    "不能直接解释为元素含量升高或降低对类别概率的影响方向。"
)