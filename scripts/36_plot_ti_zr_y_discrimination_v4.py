# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 36_plot_ti_zr_y_discrimination.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.

# ============================================================
#
# Pearce & Cann (1973)
# Analysis step.
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# ti_zr_y_pearce_cann_independent_test_v4.png
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
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
    '\\03_model_data\\Basalt_test_stable50_main_v4.csv'
)

output_dir = (
    str(FINAL_DIR) + 
    r"\05_results\figures\traditional_diagrams"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = os.path.join(
    output_dir,
    "ti_zr_y_pearce_cann_independent_test_v4.png"
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


# Analysis step.
group_arc = [
    "CAB",
    "IAB",
    "IOAB",
    "BABB"
]

group_nonarc = [
    "MORB",
    "OIB",
    "OPB",
    "CFB"
]


# ============================================================
# Analysis step.
# ============================================================

# Analysis step.
# Analysis step.
colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]


class_colors = {
    c: colors[i % len(colors)]
    for i, c in enumerate(classes)
}


class_markers = {

    "CAB": "o",
    "IAB": "s",
    "IOAB": "^",
    "BABB": "D",

    "MORB": "o",
    "OIB": "s",
    "OPB": "^",
    "CFB": "D"

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
print("Pearce & Cann (1973) Zr-3Y-Ti/100 传统构造判别图")
print("=" * 100)

print("\n独立测试集：", df.shape)


required_cols = [

    "LABEL",

    "TIO2(WT%)",
    "ZR(PPM)",
    "Y(PPM)"

]


for col in required_cols:

    if col not in df.columns:

        raise RuntimeError(
            f"缺少字段：{col}"
        )


# ============================================================
# Analysis step.
# ============================================================

for col in [
    "TIO2(WT%)",
    "ZR(PPM)",
    "Y(PPM)"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["LABEL"] = (
    df["LABEL"]
    .astype(str)
    .str.strip()
)


# ============================================================
# Analysis step.
# ============================================================

valid = (

    df[
        [
            "TIO2(WT%)",
            "ZR(PPM)",
            "Y(PPM)"
        ]
    ]
    .notna()
    .all(axis=1)

    &

    (df["TIO2(WT%)"] > 0)

    &

    (df["ZR(PPM)"] > 0)

    &

    (df["Y(PPM)"] > 0)

)


plot_df = (
    df.loc[valid]
    .copy()
)


print("\n可用于 Ti-Zr-Y 图的样品：")

print(
    f"{len(plot_df)} / {len(df)} "
    f"({len(plot_df) / len(df) * 100:.2f}%)"
)


print("\n各类别可用样品数：")

for c in classes:

    n = (
        plot_df["LABEL"]
        .eq(c)
        .sum()
    )

    total = (
        df["LABEL"]
        .eq(c)
        .sum()
    )

    print(
        f"{c:5s}: "
        f"{n:5d} / {total:5d} "
        f"({n / total * 100:.2f}%)"
    )


# ============================================================
# 7. TiO2 -> Ti ppm
#
# Analysis step.
#
# Ti / TiO2
# =
# 47.867 /
# (47.867 + 2*15.999)
#
# wt.% -> ppm:
# ×10000
#
# Analysis step.
# ============================================================

atomic_Ti = 47.867
atomic_O = 15.999


ti_fraction = (
    atomic_Ti
    /
    (
        atomic_Ti
        + 2 * atomic_O
    )
)


plot_df["TI_PPM"] = (

    plot_df["TIO2(WT%)"]

    * 10000.0

    * ti_fraction

)


plot_df["TI_DIV100"] = (
    plot_df["TI_PPM"]
    / 100.0
)


plot_df["ZR_COMPONENT"] = (
    plot_df["ZR(PPM)"]
)


plot_df["Y3_COMPONENT"] = (
    plot_df["Y(PPM)"]
    * 3.0
)


# ============================================================
# Analysis step.
#
# Analysis step.
#
# T = Ti/100
# Z = Zr
# Y = 3Y
# ============================================================

component_sum = (

    plot_df["TI_DIV100"]

    + plot_df["ZR_COMPONENT"]

    + plot_df["Y3_COMPONENT"]

)


plot_df["T"] = (
    plot_df["TI_DIV100"]
    / component_sum
)


plot_df["Z"] = (
    plot_df["ZR_COMPONENT"]
    / component_sum
)


plot_df["Y3"] = (
    plot_df["Y3_COMPONENT"]
    / component_sum
)


# ============================================================
# Analysis step.
#
# Analysis step.
#
#                Ti/100
#                  /\
#                 /  \
#                /    \
#              Zr ---- 3Y
#
# ============================================================

sqrt3 = np.sqrt(3.0)


def ternary_to_xy(
    T,
    Z,
    Y3
):

    x = (
        Y3
        + 0.5 * T
    )

    y = (
        T
        * sqrt3
        / 2.0
    )

    return x, y


plot_df["X"], plot_df["YY"] = (
    ternary_to_xy(
        plot_df["T"].values,
        plot_df["Z"].values,
        plot_df["Y3"].values
    )
)


# ============================================================
# Analysis step.
#
# Analysis step.
#
# (Ti/100, Zr, 3Y)
#
# Analysis step.
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# ============================================================


# Analysis step.
P1 = (0.504, 0.399, 0.097)
P2 = (0.482, 0.260, 0.258)

P3 = (0.392, 0.227, 0.380)
P4 = (0.300, 0.282, 0.419)

P5 = (0.190, 0.435, 0.375)

P6 = (0.104, 0.561, 0.334)
P7 = (0.104, 0.638, 0.258)
P8 = (0.162, 0.643, 0.195)

P9 = (0.250, 0.564, 0.186)
P10 = (0.286, 0.602, 0.112)

J = (0.300, 0.457, 0.243)
K = (0.425, 0.324, 0.251)


# ------------------------------------------------------------
# Analysis step.
# ------------------------------------------------------------

field_A = [
    K,
    P2,
    P3,
    P4
]


# ------------------------------------------------------------
# Analysis step.
# ------------------------------------------------------------

field_B = [
    K,
    P4,
    P5,
    J
]


# ------------------------------------------------------------
# Analysis step.
# ------------------------------------------------------------

field_C = [
    P9,
    J,
    P5,
    P6,
    P7,
    P8
]


# ------------------------------------------------------------
# Analysis step.
# ------------------------------------------------------------

field_D = [
    P1,
    P2,
    K,
    J,
    P9,
    P10
]


field_dict = {

    "A": field_A,
    "B": field_B,
    "C": field_C,
    "D": field_D

}


# ============================================================
# Analysis step.
# ============================================================

def polygon_to_xy(
    polygon
):

    points = []

    for T, Z, Y3 in polygon:

        x, y = ternary_to_xy(
            T,
            Z,
            Y3
        )

        points.append(
            [x, y]
        )

    return np.asarray(
        points
    )


# ============================================================
# Analysis step.
# ============================================================

def draw_ternary_panel(
    ax,
    data,
    panel_classes,
    panel_title
):

    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [
                0.5,
                sqrt3 / 2.0
            ],
            [0.0, 0.0]
        ]
    )


    ax.plot(
        triangle[:, 0],
        triangle[:, 1],
        linewidth=1.2,
        color="black"
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    for field_name, polygon in field_dict.items():

        xy = polygon_to_xy(
            polygon
        )


        patch = Polygon(
            xy,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )


        ax.add_patch(
            patch
        )


        centre = (
            xy.mean(axis=0)
        )


        ax.text(
            centre[0],
            centre[1],
            field_name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=4
        )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    for c in panel_classes:

        sub = (
            data[
                data["LABEL"] == c
            ]
        )


        ax.scatter(
            sub["X"],
            sub["YY"],

            s=15,

            alpha=0.30,

            marker=class_markers[c],

            color=class_colors[c],

            edgecolors="none",

            label=(
                f"{c} "
                f"(n={len(sub)})"
            ),

            zorder=2
        )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    ax.text(
        0.50,
        sqrt3 / 2 + 0.035,
        "Ti/100",
        ha="center",
        va="bottom",
        fontsize=12
    )


    ax.text(
        -0.025,
        -0.025,
        "Zr",
        ha="right",
        va="top",
        fontsize=12
    )


    ax.text(
        1.025,
        -0.025,
        "3Y",
        ha="left",
        va="top",
        fontsize=12
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    ax.set_xlim(
        -0.08,
        1.08
    )

    ax.set_ylim(
        -0.08,
        sqrt3 / 2 + 0.08
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )


    ax.set_title(
        panel_title,
        fontsize=14,
        loc="left",
        pad=10
    )


    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        markerscale=1.5
    )


# ============================================================
# Analysis step.
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 7.4)
)


draw_ternary_panel(

    axes[0],

    plot_df,

    group_arc,

    "(a) 弧相关及过渡构造环境"

)


draw_ternary_panel(

    axes[1],

    plot_df,

    group_nonarc,

    "(b) 非弧及板内构造环境"

)


# ============================================================
# Analysis step.
# ============================================================

fig.text(

    0.5,
    0.025,

    "经典字段："
    "A—岛弧拉斑玄武岩；"
    "B—MORB、岛弧拉斑玄武岩与钙碱性玄武岩重叠区；"
    "C—钙碱性玄武岩；"
    "D—板内玄武岩",

    ha="center",
    fontsize=11

)


plt.subplots_adjust(
    left=0.035,
    right=0.985,
    top=0.94,
    bottom=0.10,
    wspace=0.10
)


# ============================================================
# Analysis step.
# ============================================================

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Analysis step.
# ============================================================

print("\n" + "=" * 100)
print("Ti-Zr-Y传统构造判别图绘制完成")
print("=" * 100)

print("\n保存：")
print(output_file)


print(
    "\n重要说明："
)

print(
    "Pearce & Cann (1973) A-D边界为经验性手绘边界；"
)

print(
    "本图边界为依据 Vermeesch (2006) Fig.6 "
    "再绘图数字化得到的视觉近似，"
)

print(
    "用于传统判别图与机器学习结果的定性比较，"
    "不用于计算精确分类准确率。"
)