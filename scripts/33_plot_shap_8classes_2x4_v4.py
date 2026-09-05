# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 33_plot_shap_8classes_2x4.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.

"""Documentation for this analysis component."""

import os
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

INPUT_FILE = (
    str(FINAL_DIR) + 
    r"\05_results\shap"
    r"\shap_importance_independent_test_v4.csv"
)

OUTPUT_DIR = (
    str(FINAL_DIR) + 
    r"\05_results\figures\shap_importance"
)

FIGURE_GROUPS = [
    (
        ["CAB", "IAB", "IOAB", "BABB"],
        "shap_arc_4classes_2x2_stable50_v4.png",
    ),
    (
        ["MORB", "OIB", "OPB", "CFB"],
        "shap_nonarc_4classes_2x2_stable50_v4.png",
    ),
]

ALL_CLASSES = [
    tectonic_class
    for class_group, _ in FIGURE_GROUPS
    for tectonic_class in class_group
]

TOP_N = 8
FIGSIZE = (7.2, 6.2)  # Analysis step.
DPI = 300
BAR_COLOR = "#4472A8"
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


# ============================================================
# Analysis step.
# ============================================================


def select_chinese_font():
    """Documentation for this analysis component."""
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
    ]

    for font_name in candidates:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            return font_name
        except ValueError:
            continue

    raise RuntimeError(
        "未找到可用中文字体。请安装 Noto Sans CJK SC、思源黑体、微软雅黑或黑体。"
    )


CHINESE_FONT = select_chinese_font()

plt.rcParams.update(
    {
        "font.sans-serif": [CHINESE_FONT, "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# ============================================================
# Analysis step.
# ============================================================

FEATURE_NAME_MAP = {
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
    "BA(PPM)": "Ba",
}


# ============================================================
# Analysis step.
# ============================================================


def find_feature_column(data):
    candidates = ["feature", "Feature", "FEATURE", "特征", "variable", "Variable"]
    return next((column for column in candidates if column in data.columns), data.columns[0])


def find_class_columns(data):
    class_columns = {}

    for tectonic_class in ALL_CLASSES:
        candidates = []

        for column in data.columns:
            tokens = re.split(r"[^A-Z0-9]+", column.upper())
            if tectonic_class in tokens:
                candidates.append(column)

        shap_candidates = [
            column for column in candidates if "SHAP" in column.upper()
        ]

        if shap_candidates:
            class_columns[tectonic_class] = shap_candidates[0]
        elif candidates:
            class_columns[tectonic_class] = candidates[0]

    missing = [c for c in ALL_CLASSES if c not in class_columns]
    if missing:
        raise RuntimeError(f"以下类别没有识别到 SHAP 字段：{missing}")

    return class_columns


def load_top_data():
    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
    feature_column = find_feature_column(data)
    class_columns = find_class_columns(data)

    data[feature_column] = data[feature_column].astype(str).str.strip()
    data["显示名称"] = data[feature_column].map(FEATURE_NAME_MAP).fillna(data[feature_column])

    top_data = {}

    for tectonic_class in ALL_CLASSES:
        shap_column = class_columns[tectonic_class]
        data[shap_column] = pd.to_numeric(data[shap_column], errors="coerce")

        valid_values = data[shap_column].dropna()
        if valid_values.empty:
            raise RuntimeError(f"{shap_column} 没有可用于绘图的数值。")
        if (valid_values < 0).any():
            raise RuntimeError(f"{shap_column} 含负值，不符合 mean(|SHAP|) 的定义。")

        top_data[tectonic_class] = (
            data[["显示名称", shap_column]]
            .dropna()
            .sort_values(shap_column, ascending=False)
            .head(TOP_N)
            .copy()
        )

    return data, feature_column, class_columns, top_data


# ============================================================
# Analysis step.
# ============================================================


def draw_figure(class_group, output_name, class_columns, top_data):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        layout="constrained",
    )

    for index, (axis, tectonic_class) in enumerate(zip(axes.flat, class_group)):
        shap_column = class_columns[tectonic_class]
        plot_data = top_data[tectonic_class].sort_values(shap_column, ascending=True)
        local_max = float(plot_data[shap_column].max())

        axis.barh(
            plot_data["显示名称"],
            plot_data[shap_column],
            height=0.68,
            color=BAR_COLOR,
            edgecolor="none",
        )

        axis.set_title(
            f"{PANEL_LABELS[index]} {tectonic_class}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )

        # Analysis step.
        axis.set_xlim(0, local_max * 1.08)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.tick_params(axis="x", labelsize=8, length=3, width=0.7)
        axis.tick_params(axis="y", labelsize=8.5, length=0, pad=3)

        axis.grid(axis="x", color="#D9D9D9", linestyle="--", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#666666")
        axis.spines["bottom"].set_linewidth(0.7)

    fig.supxlabel("平均绝对 SHAP 值", fontsize=10.5)
    fig.supylabel("地球化学特征", fontsize=10.5)

    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Analysis step.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
        fig.savefig(output_path, dpi=DPI)

    missing_glyph_warnings = [
        str(item.message)
        for item in caught
        if "Glyph" in str(item.message) and "missing from font" in str(item.message)
    ]
    if missing_glyph_warnings:
        plt.close(fig)
        raise RuntimeError(
            "检测到字体缺字，已停止输出：\n" + "\n".join(missing_glyph_warnings)
        )

    plt.close(fig)
    return output_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data, feature_column, class_columns, top_data = load_top_data()

    print("=" * 88)
    print("Stable-50 XGBoost：8 类 SHAP 重要性，两张 2×2 中文投稿版图")
    print("=" * 88)
    print(f"输入文件：{INPUT_FILE}")
    print(f"数据维度：{data.shape}")
    print(f"特征名称列：{feature_column}")
    print(f"中文字体：{CHINESE_FONT}")
    print(f"每类展示：Top {TOP_N}")
    print("横轴设置：各子图独立范围（均从 0 起）")

    print("\n识别到的类别 SHAP 列：")
    for tectonic_class in ALL_CLASSES:
        print(f"  {tectonic_class:5s} -> {class_columns[tectonic_class]}")

    output_paths = []
    for class_group, output_name in FIGURE_GROUPS:
        output_paths.append(
            draw_figure(class_group, output_name, class_columns, top_data)
        )

    print("\n各类别 Top 8：")
    for tectonic_class in ALL_CLASSES:
        shap_column = class_columns[tectonic_class]
        ranking = top_data[tectonic_class].sort_values(shap_column, ascending=False)
        items = [
            f"{rank}. {row['显示名称']} ({row[shap_column]:.4f})"
            for rank, (_, row) in enumerate(ranking.iterrows(), start=1)
        ]
        print(f"\n{tectonic_class}：")
        print("  " + "；".join(items))

    print("\n输出完成：")
    for output_path in output_paths:
        print(f"  {output_path}")
    print("\n说明：各子图横轴尺度独立，条形长度仅用于类别内部比较。")


if __name__ == "__main__":
    main()
