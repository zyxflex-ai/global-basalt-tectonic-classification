# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 35_plot_shap_dependence_2x2.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
"""Documentation for this analysis component."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

PROJECT_DIR = Path(str(FINAL_DIR))
TRAIN_FILE = PROJECT_DIR / "03_model_data" / "Basalt_train_stable50_main_v4.csv"
TEST_FILE = PROJECT_DIR / "03_model_data" / "Basalt_test_stable50_main_v4.csv"
DIRECTION_SUMMARY_FILE = (
    PROJECT_DIR / "05_results" / "shap" / "shap_direction_class_specific_v4.csv"
)
DEPENDENCE_SUMMARY_FILE = (
    PROJECT_DIR
    / "05_results"
    / "shap"
    / "dependence"
    / "shap_dependence_key_features_v4.csv"
)
SHAP_CACHE_FILE = (
    PROJECT_DIR
    / "05_results"
    / "figures"
    / "shap_direction"
    / "stable50_independent_test_shap_values_v4.npz"
)
OUTPUT_DIR = PROJECT_DIR / "05_results" / "figures" / "shap_dependence"

CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
CLASS_TO_INT = {name: index for index, name in enumerate(CLASSES)}

FEATURES = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FE_TOTAL(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
    "V(PPM)",
    "CR(PPM)",
    "NI(PPM)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "BA(PPM)",
]

FROZEN_MODEL_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 8,
    "n_estimators": 1400,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

FIGURE_GROUPS = [
    (
        "弧相关构造环境",
        [
            ("CAB", "TIO2(WT%)"),
            ("IAB", "CAO(WT%)"),
            ("IOAB", "TIO2(WT%)"),
            ("BABB", "SIO2(WT%)"),
        ],
        "shap_dependence_arc_4classes_2x2_stable50_v4",
    ),
    (
        "非弧及板内构造环境",
        [
            ("MORB", "SR(PPM)"),
            ("OIB", "TIO2(WT%)"),
            ("OPB", "P2O5(WT%)"),
            ("CFB", "FE_TOTAL(WT%)"),
        ],
        "shap_dependence_nonarc_4classes_2x2_stable50_v4",
    ),
]

FEATURE_MATH_NAMES = {
    "SIO2(WT%)": r"$\mathrm{SiO_2}$",
    "TIO2(WT%)": r"$\mathrm{TiO_2}$",
    "CAO(WT%)": r"$\mathrm{CaO}$",
    "P2O5(WT%)": r"$\mathrm{P_2O_5}$",
    "SR(PPM)": r"$\mathrm{Sr}$",
    "FE_TOTAL(WT%)": "全铁",
}

FEATURE_PLAIN_NAMES = {
    "SIO2(WT%)": "SiO2",
    "TIO2(WT%)": "TiO2",
    "CAO(WT%)": "CaO",
    "P2O5(WT%)": "P2O5",
    "SR(PPM)": "Sr",
    "FE_TOTAL(WT%)": "全铁",
}

FEATURE_AXIS_LABELS = {
    "SIO2(WT%)": r"原始 $\mathrm{SiO_2}$ 含量 (wt.%)",
    "TIO2(WT%)": r"原始 $\mathrm{TiO_2}$ 含量 (wt.%)",
    "CAO(WT%)": r"原始 $\mathrm{CaO}$ 含量 (wt.%)",
    "P2O5(WT%)": r"原始 $\mathrm{P_2O_5}$ 含量 (wt.%)",
    "SR(PPM)": r"原始 $\mathrm{Sr}$ 含量 (ppm)",
    "FE_TOTAL(WT%)": "原始全铁含量 (wt.%)",
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
FIGSIZE = (7.2, 5.8)
DPI = 600
N_BINS = 15
LOWER_DISPLAY_PERCENTILE = 1.0
UPPER_DISPLAY_PERCENTILE = 99.0
SHAP_VALIDATION_ATOL = 2e-6
SHAP_VALIDATION_RTOL = 2e-6
DEPENDENCE_VALIDATION_ATOL = 2e-7

ALL_POINT_COLOR = "#7A7A7A"
TRUE_CLASS_COLOR = "#0072B2"
TREND_COLOR = "#D55E00"
ZERO_COLOR = "#222222"


@dataclass(frozen=True)
class PanelData:
    group_name: str
    class_name: str
    feature: str
    x: np.ndarray
    shap: np.ndarray
    true_target: np.ndarray
    display_mask: np.ndarray
    bins: pd.DataFrame
    n_total: int
    n_measured: int
    n_missing: int
    n_displayed: int
    n_true_measured: int
    x_plot_min: float
    x_plot_max: float


# ============================================================
# Analysis step.
# ============================================================


def select_chinese_font() -> str:
    candidates = [
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
    ]
    for font_name in candidates:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            return font_name
        except ValueError:
            continue
    raise RuntimeError("未找到可用中文字体，已停止出图。")


CHINESE_FONT = select_chinese_font()

plt.rcParams.update(
    {
        "font.family": CHINESE_FONT,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.7,
    }
)


# ============================================================
# Analysis step.
# ============================================================


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inputs() -> None:
    required = [
        TRAIN_FILE,
        TEST_FILE,
        DIRECTION_SUMMARY_FILE,
        DEPENDENCE_SUMMARY_FILE,
        SHAP_CACHE_FILE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少已冻结结果或既有 SHAP 产物；本脚本不会重新训练。\n"
            + "\n".join(missing)
        )


def load_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    test = pd.read_csv(TEST_FILE, encoding="utf-8-sig", low_memory=False)
    missing_columns = [column for column in ["LABEL", *FEATURES] if column not in test]
    if missing_columns:
        raise RuntimeError(f"独立测试集缺少字段：{missing_columns}")
    if test["LABEL"].isna().any():
        raise RuntimeError("独立测试集 LABEL 含缺失值。")
    unknown_labels = sorted(set(test["LABEL"].astype(str)) - set(CLASSES))
    if unknown_labels:
        raise RuntimeError(f"独立测试集含未知类别：{unknown_labels}")
    feature_values = test[FEATURES].to_numpy(dtype=np.float64, copy=True)
    return test, feature_values


def validate_cache_metadata(metadata_json: str) -> None:
    metadata = json.loads(metadata_json)
    if metadata.get("source_script") != "23_shap_direction_class_specific.py":
        raise RuntimeError("SHAP 缓存来源不是冻结的 23 号脚本。")
    cached_params = metadata.get("frozen_model_params")
    if cached_params != FROZEN_MODEL_PARAMS:
        raise RuntimeError("SHAP 缓存中的模型参数与 Stable-50 冻结定义不一致。")


def load_verified_cache(
    current_feature_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(SHAP_CACHE_FILE, allow_pickle=False) as cache:
        required_keys = {
            "shap_values",
            "feature_values",
            "feature_names",
            "class_names",
            "train_sha256",
            "test_sha256",
            "metadata_json",
        }
        missing_keys = required_keys - set(cache.files)
        if missing_keys:
            raise RuntimeError(f"SHAP 缓存缺少字段：{sorted(missing_keys)}")

        shap_values = np.asarray(cache["shap_values"], dtype=np.float32)
        cached_feature_values = np.asarray(cache["feature_values"], dtype=np.float64)
        cached_features = cache["feature_names"].astype(str).tolist()
        cached_classes = cache["class_names"].astype(str).tolist()
        cached_train_hash = str(cache["train_sha256"].item())
        cached_test_hash = str(cache["test_sha256"].item())
        metadata_json = str(cache["metadata_json"].item())

    expected_shape = (len(current_feature_values), len(FEATURES), len(CLASSES))
    if shap_values.shape != expected_shape:
        raise RuntimeError(
            f"SHAP 缓存 shape 错误：{shap_values.shape}，预期 {expected_shape}"
        )
    if cached_feature_values.shape != current_feature_values.shape:
        raise RuntimeError("SHAP 缓存原始特征矩阵 shape 与当前测试集不一致。")
    if cached_features != FEATURES or cached_classes != CLASSES:
        raise RuntimeError("SHAP 缓存的特征或类别顺序与冻结定义不一致。")
    if cached_train_hash != file_sha256(TRAIN_FILE):
        raise RuntimeError("训练集已变化，拒绝使用当前 SHAP 缓存。")
    if cached_test_hash != file_sha256(TEST_FILE):
        raise RuntimeError("独立测试集已变化，拒绝使用当前 SHAP 缓存。")
    if not np.allclose(
        cached_feature_values,
        current_feature_values,
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise RuntimeError("SHAP 缓存原始特征值与当前独立测试集逐项不一致。")
    if not np.isfinite(shap_values).all():
        raise RuntimeError("SHAP 缓存含 NaN 或无穷值。")

    validate_cache_metadata(metadata_json)
    return shap_values, cached_feature_values


def validate_against_direction_summary(shap_values: np.ndarray) -> float:
    summary = pd.read_csv(
        DIRECTION_SUMMARY_FILE, encoding="utf-8-sig", low_memory=False
    )
    required_columns = {"class", "feature", "mean_abs_shap"}
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise RuntimeError(f"23 号方向汇总表缺少字段：{sorted(missing_columns)}")

    summary["class"] = summary["class"].astype(str).str.strip()
    summary["feature"] = summary["feature"].astype(str).str.strip()
    lookup = summary.set_index(["class", "feature"])["mean_abs_shap"]

    differences: list[float] = []
    for class_id, class_name in enumerate(CLASSES):
        for feature_id, feature in enumerate(FEATURES):
            expected = float(lookup.loc[(class_name, feature)])
            actual = float(np.mean(np.abs(shap_values[:, feature_id, class_id])))
            differences.append(abs(actual - expected))
            if not np.isclose(
                actual,
                expected,
                atol=SHAP_VALIDATION_ATOL,
                rtol=SHAP_VALIDATION_RTOL,
            ):
                raise RuntimeError(
                    "逐样本 SHAP 与 23 号方向汇总表不一致："
                    f"{class_name} / {feature}; actual={actual:.9g}, "
                    f"reference={expected:.9g}"
                )
    return float(max(differences))


# ============================================================
# Analysis step.
# ============================================================


def make_quantile_bins(x: np.ndarray, shap_values: np.ndarray) -> pd.DataFrame:
    data = pd.DataFrame({"x": x, "shap": shap_values}).dropna()
    if len(data) < 30:
        raise RuntimeError("实测样本少于 30，无法生成稳定分位数趋势。")

    data["bin"] = pd.qcut(data["x"], q=N_BINS, duplicates="drop")
    bins = (
        data.groupby("bin", observed=True)
        .agg(
            x_median=("x", "median"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            shap_median=("shap", "median"),
            shap_mean=("shap", "mean"),
            shap_q25=("shap", lambda values: np.percentile(values, 25)),
            shap_q75=("shap", lambda values: np.percentile(values, 75)),
            n=("shap", "size"),
        )
        .reset_index(drop=True)
    )
    bins.insert(0, "bin", np.arange(1, len(bins) + 1, dtype=int))
    return bins


def load_dependence_reference() -> pd.DataFrame:
    reference = pd.read_csv(
        DEPENDENCE_SUMMARY_FILE, encoding="utf-8-sig", low_memory=False
    )
    required_columns = {
        "class",
        "feature",
        "bin",
        "x_median",
        "x_min",
        "x_max",
        "shap_median",
        "shap_mean",
        "shap_q25",
        "shap_q75",
        "n",
    }
    missing_columns = required_columns - set(reference.columns)
    if missing_columns:
        raise RuntimeError(f"25 号 dependence 产物缺少字段：{sorted(missing_columns)}")
    reference["class"] = reference["class"].astype(str).str.strip()
    reference["feature"] = reference["feature"].astype(str).str.strip()
    return reference


def validate_dependence_bins(
    rebuilt: pd.DataFrame,
    reference: pd.DataFrame,
    class_name: str,
    feature: str,
) -> tuple[pd.DataFrame, float]:
    existing = (
        reference.loc[
            reference["class"].eq(class_name) & reference["feature"].eq(feature)
        ]
        .sort_values("bin")
        .reset_index(drop=True)
    )
    if len(existing) != len(rebuilt):
        raise RuntimeError(
            f"25 号 dependence 分箱数不一致：{class_name} / {feature}"
        )

    numeric_columns = [
        "x_median",
        "x_min",
        "x_max",
        "shap_median",
        "shap_mean",
        "shap_q25",
        "shap_q75",
    ]
    maximum_difference = 0.0
    for column in numeric_columns:
        actual = rebuilt[column].to_numpy(dtype=float)
        expected = existing[column].to_numpy(dtype=float)
        maximum_difference = max(
            maximum_difference, float(np.max(np.abs(actual - expected)))
        )
        if not np.allclose(
            actual, expected, rtol=0, atol=DEPENDENCE_VALIDATION_ATOL
        ):
            raise RuntimeError(
                f"25 号 dependence 产物复核失败：{class_name} / {feature} / {column}"
            )

    if not np.array_equal(
        rebuilt["n"].to_numpy(dtype=int), existing["n"].to_numpy(dtype=int)
    ):
        raise RuntimeError(
            f"25 号 dependence 分箱样本数不一致：{class_name} / {feature}"
        )
    return existing, maximum_difference


def prepare_panels(
    test: pd.DataFrame,
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    dependence_reference: pd.DataFrame,
) -> tuple[dict[tuple[str, str], PanelData], float]:
    panels: dict[tuple[str, str], PanelData] = {}
    maximum_bin_difference = 0.0
    labels = test["LABEL"].astype(str).to_numpy()

    for group_name, pair_group, _ in FIGURE_GROUPS:
        for class_name, feature in pair_group:
            class_id = CLASS_TO_INT[class_name]
            feature_id = FEATURES.index(feature)
            x_all = np.asarray(feature_values[:, feature_id], dtype=float)
            shap_all = np.asarray(shap_values[:, feature_id, class_id], dtype=float)
            measured = np.isfinite(x_all) & np.isfinite(shap_all)
            x = x_all[measured]
            shap_selected = shap_all[measured]
            true_target = labels[measured] == class_name

            x_plot_min, x_plot_max = np.percentile(
                x, [LOWER_DISPLAY_PERCENTILE, UPPER_DISPLAY_PERCENTILE]
            )
            display_mask = (x >= x_plot_min) & (x <= x_plot_max)

            rebuilt_bins = make_quantile_bins(x, shap_selected)
            bins, difference = validate_dependence_bins(
                rebuilt=rebuilt_bins,
                reference=dependence_reference,
                class_name=class_name,
                feature=feature,
            )
            maximum_bin_difference = max(maximum_bin_difference, difference)

            panel = PanelData(
                group_name=group_name,
                class_name=class_name,
                feature=feature,
                x=x,
                shap=shap_selected,
                true_target=true_target,
                display_mask=display_mask,
                bins=bins,
                n_total=len(test),
                n_measured=int(measured.sum()),
                n_missing=int((~measured).sum()),
                n_displayed=int(display_mask.sum()),
                n_true_measured=int(true_target.sum()),
                x_plot_min=float(x_plot_min),
                x_plot_max=float(x_plot_max),
            )
            panels[(class_name, feature)] = panel

            display_name = FEATURE_PLAIN_NAMES[feature]
            print(
                f"[{class_name}–{display_name}] 总样本={panel.n_total:,}；"
                f"实测有效 n={panel.n_measured:,}；缺失/非有限={panel.n_missing:,}；"
                f"1%--99% 显示 n={panel.n_displayed:,}；"
                f"真实 {class_name} 实测 n={panel.n_true_measured:,}"
            )

    return panels, maximum_bin_difference


# ============================================================
# Analysis step.
# ============================================================


def collect_missing_glyph_warnings(
    caught: list[warnings.WarningMessage],
) -> list[str]:
    return sorted(
        {
            str(item.message)
            for item in caught
            if "Glyph" in str(item.message) and "missing from font" in str(item.message)
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(axis="both", labelsize=7.6, length=3, width=0.65, pad=2)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick_label.set_fontfamily("Times New Roman")


def draw_panel(axis: plt.Axes, panel: PanelData, panel_label: str) -> None:
    visible = panel.display_mask
    true_visible = visible & panel.true_target

    background = axis.scatter(
        panel.x[visible],
        panel.shap[visible],
        s=6.5,
        color=ALL_POINT_COLOR,
        alpha=0.18,
        edgecolors="none",
        zorder=2,
    )
    target = axis.scatter(
        panel.x[true_visible],
        panel.shap[true_visible],
        s=18,
        facecolors="none",
        edgecolors=TRUE_CLASS_COLOR,
        linewidths=0.55,
        alpha=0.82,
        zorder=3,
    )
    background.set_rasterized(True)
    target.set_rasterized(True)

    axis.plot(
        panel.bins["x_median"],
        panel.bins["shap_median"],
        color=TREND_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=TREND_COLOR,
        markeredgewidth=0.8,
        zorder=5,
    )
    axis.axhline(
        0,
        color=ZERO_COLOR,
        linewidth=0.9,
        linestyle=(0, (4, 2.5)),
        zorder=1,
    )

    x_span = panel.x_plot_max - panel.x_plot_min
    x_pad = 0.025 * x_span if x_span > 0 else 0.5
    axis.set_xlim(panel.x_plot_min - x_pad, panel.x_plot_max + x_pad)

    visible_shap = panel.shap[visible]
    y_min = min(float(np.min(visible_shap)), 0.0)
    y_max = max(float(np.max(visible_shap)), 0.0)
    y_span = y_max - y_min
    y_pad = 0.06 * y_span if y_span > 0 else 0.5
    axis.set_ylim(y_min - y_pad, y_max + y_pad)

    feature_name = FEATURE_MATH_NAMES[panel.feature]
    axis.set_title(
        rf"{panel_label}  $\mathrm{{{panel.class_name}}}$–{feature_name}",
        loc="left",
        fontsize=10.0,
        fontweight="bold",
        pad=6,
    )
    axis.set_xlabel(FEATURE_AXIS_LABELS[panel.feature], fontsize=8.8, labelpad=4)
    axis.set_ylabel(
        rf"$\mathrm{{{panel.class_name}}}$ 类 SHAP 值",
        fontsize=8.8,
        labelpad=4,
    )
    axis.text(
        0.98,
        1.025,
        f"实测 n={panel.n_measured:,}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#333333",
        clip_on=False,
        zorder=7,
    )
    style_axis(axis)


def save_grayscale_preview(source_png: Path) -> Path:
    grayscale_path = source_png.with_name(source_png.stem + "_grayscale.png")
    with Image.open(source_png) as image:
        grayscale = image.convert("L")
        grayscale.save(grayscale_path, dpi=(DPI, DPI), optimize=True)
    return grayscale_path


def draw_figure(
    pair_group: list[tuple[str, str]],
    output_stem: str,
    panels: dict[tuple[str, str], PanelData],
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.955,
        bottom=0.155,
        wspace=0.24,
        hspace=0.36,
    )

    for axis, pair, panel_label in zip(axes.flat, pair_group, PANEL_LABELS):
        draw_panel(axis, panels[pair], panel_label)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ALL_POINT_COLOR,
            markeredgecolor="none",
            markersize=4.8,
            alpha=0.45,
            label="实测独立测试样本",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=TRUE_CLASS_COLOR,
            markeredgewidth=0.8,
            markersize=5.8,
            label="真实属于该类别的样本",
        ),
        Line2D(
            [0],
            [0],
            color=TREND_COLOR,
            linewidth=1.8,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=TREND_COLOR,
            markersize=4.2,
            label="分位数箱中位 SHAP 趋势",
        ),
        Line2D(
            [0],
            [0],
            color=ZERO_COLOR,
            linewidth=0.9,
            linestyle=(0, (4, 2.5)),
            label="SHAP = 0",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=7.6,
        handlelength=2.1,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    with warnings.catch_warnings(record=True) as caught_draw:
        warnings.simplefilter("always")
        fig.canvas.draw()
    missing_glyphs = collect_missing_glyph_warnings(caught_draw)
    if missing_glyphs:
        plt.close(fig)
        raise RuntimeError("检测到字体缺字，已停止导出：\n" + "\n".join(missing_glyphs))

    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    with warnings.catch_warnings(record=True) as caught_save:
        warnings.simplefilter("always")
        fig.savefig(png_path, dpi=DPI)
        fig.savefig(pdf_path, dpi=DPI)
    missing_glyphs = collect_missing_glyph_warnings(caught_save)
    plt.close(fig)
    if missing_glyphs:
        raise RuntimeError("导出时检测到字体缺字：\n" + "\n".join(missing_glyphs))

    grayscale_path = save_grayscale_preview(png_path)
    return [png_path, pdf_path, grayscale_path]


def write_selected_bins(
    panels: dict[tuple[str, str], PanelData]
) -> Path:
    rows: list[pd.DataFrame] = []
    for group_name, pair_group, _ in FIGURE_GROUPS:
        for panel_index, pair in enumerate(pair_group):
            panel = panels[pair]
            data = panel.bins.drop(
                columns=["class", "feature"], errors="ignore"
            ).copy()
            data.insert(0, "feature", panel.feature)
            data.insert(0, "class", panel.class_name)
            data.insert(0, "panel", PANEL_LABELS[panel_index])
            data.insert(0, "figure_group", group_name)
            data["n_total"] = panel.n_total
            data["n_measured"] = panel.n_measured
            data["n_missing"] = panel.n_missing
            data["n_displayed_1_99pct"] = panel.n_displayed
            data["n_true_class_measured"] = panel.n_true_measured
            data["x_plot_min_1pct"] = panel.x_plot_min
            data["x_plot_max_99pct"] = panel.x_plot_max
            rows.append(data)

    output_path = OUTPUT_DIR / "shap_dependence_selected_bins_stable50_v4.csv"
    pd.concat(rows, ignore_index=True).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    return output_path


def write_caption_file(panels: dict[tuple[str, str], PanelData]) -> Path:
    arc_pairs = FIGURE_GROUPS[0][1]
    nonarc_pairs = FIGURE_GROUPS[1][1]

    def count_text(pair_group: list[tuple[str, str]]) -> str:
        return "；".join(
            f"{PANEL_LABELS[index]} {pair[0]}–"
            f"{FEATURE_PLAIN_NAMES[pair[1]]}: "
            f"n={panels[pair].n_measured:,}"
            for index, pair in enumerate(pair_group)
        )

    content = f"""图X 弧相关构造环境关键特征的类别特异 SHAP 依赖关系
（a）CAB–TiO2；（b）IAB–CaO；（c）IOAB–TiO2；（d）BABB–SiO2。
有效实测样本数：{count_text(arc_pairs)}。

图X 非弧及板内构造环境关键特征的类别特异 SHAP 依赖关系
（a）MORB–Sr；（b）OIB–TiO2；（c）OPB–P2O5；（d）CFB–全铁。
有效实测样本数：{count_text(nonarc_pairs)}。

统一图注：
横坐标为独立测试样本的原始元素实测值，纵坐标为该特征对对应类别模型输出的 SHAP 贡献；SHAP > 0 表示该特征在该取值下推动模型判为对应类别，SHAP < 0 表示使模型远离对应类别，虚线表示 SHAP = 0。灰色实心点为所有具有该特征实测值的独立测试样本，蓝色空心点为其中真实属于对应类别的样本，橙色折线为基于全部实测值的 15 个分位数箱中位 SHAP 趋势。缺失值不作插补且不绘制。为避免极端值压缩主体关系，散点横轴显示范围沿用既有 25 号分析脚本的 1%--99% 实测值分位范围；分箱趋势计算仍使用全部实测值。各子图坐标范围独立。SHAP 反映冻结 Stable-50 XGBoost 的模型预测贡献，不代表因果作用，也不直接定义新的地球化学判别阈值。
"""
    output_path = OUTPUT_DIR / "shap_dependence_captions_stable50_v4.txt"
    output_path.write_text(content, encoding="utf-8-sig")
    return output_path


def write_qa_report(
    panels: dict[tuple[str, str], PanelData],
    max_direction_difference: float,
    max_bin_difference: float,
    output_paths: list[Path],
) -> Path:
    lines = [
        "Stable-50 XGBoost SHAP dependence 2×2 图 QA 报告",
        "=" * 72,
        "模型操作: 未训练、未调参、未重新选择模型",
        f"逐样本 SHAP 来源: {SHAP_CACHE_FILE}",
        f"25 号 dependence 趋势来源: {DEPENDENCE_SUMMARY_FILE}",
        f"与 23 号 mean(|SHAP|) 汇总最大绝对差: {max_direction_difference:.8g}",
        f"与 25 号分箱产物最大绝对差: {max_bin_difference:.8g}",
        f"中文字体: {CHINESE_FONT}",
        "西文/数字: Times New Roman；数学下标: STIX mathtext",
        f"终稿尺寸: {FIGSIZE[0]} × {FIGSIZE[1]} in",
        f"PNG/栅格化散点分辨率: {DPI} dpi",
        "PDF 字体类型: TrueType (fonttype 42)",
        "PDF 策略: 散点栅格化；趋势、坐标轴和文字保持矢量",
        "缺字检查: PASS",
        "缺失值策略: 不插补、不绘制，仅绘实测值",
        "显示策略: 横轴 1%--99% 实测值；15 分位数趋势使用全部实测值",
        "参考线: 每个子图明确绘制 SHAP = 0 虚线",
        "",
        "面板样本数：",
    ]
    for group_name, pair_group, _ in FIGURE_GROUPS:
        lines.append(group_name)
        for index, pair in enumerate(pair_group):
            panel = panels[pair]
            lines.append(
                f"- {PANEL_LABELS[index]} {panel.class_name} / {panel.feature}: "
                f"总样本={panel.n_total:,}; 实测={panel.n_measured:,}; "
                f"缺失/非有限={panel.n_missing:,}; 显示={panel.n_displayed:,}; "
                f"真实目标类实测={panel.n_true_measured:,}; "
                f"x显示范围=[{panel.x_plot_min:.6g}, {panel.x_plot_max:.6g}]"
            )
    lines.extend(["", "输出文件："])
    lines.extend(f"- {path} ({path.stat().st_size:,} bytes)" for path in output_paths)

    output_path = OUTPUT_DIR / "shap_dependence_qa_stable50_v4.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return output_path


# ============================================================
# Analysis step.
# ============================================================


def main() -> None:
    print("=" * 92)
    print("Stable-50 XGBoost：关键特征 SHAP dependence 两张中文 2×2 投稿版图")
    print("=" * 92)
    print("模型状态：只读冻结结果；不训练、不调参、不重新选择模型")
    print(f"独立测试集：{TEST_FILE}")
    print(f"逐样本 SHAP 缓存：{SHAP_CACHE_FILE}")
    print(f"25 号 dependence 产物：{DEPENDENCE_SUMMARY_FILE}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"中文字体：{CHINESE_FONT}")

    require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test, current_feature_values = load_test_data()
    shap_values, feature_values = load_verified_cache(current_feature_values)
    max_direction_difference = validate_against_direction_summary(shap_values)
    print(
        "冻结 SHAP 核验通过；与 23 号方向汇总最大绝对差="
        f"{max_direction_difference:.8g}"
    )

    dependence_reference = load_dependence_reference()
    panels, max_bin_difference = prepare_panels(
        test=test,
        feature_values=feature_values,
        shap_values=shap_values,
        dependence_reference=dependence_reference,
    )
    print(
        "25 号 dependence 分箱产物复核通过；最大绝对差="
        f"{max_bin_difference:.8g}"
    )

    output_paths: list[Path] = []
    for _, pair_group, output_stem in FIGURE_GROUPS:
        figure_paths = draw_figure(
            pair_group=pair_group,
            output_stem=output_stem,
            panels=panels,
        )
        output_paths.extend(figure_paths)
        for path in figure_paths:
            print(f"已输出：{path}")

    bins_path = write_selected_bins(panels)
    caption_path = write_caption_file(panels)
    output_paths.extend([bins_path, caption_path])
    qa_path = write_qa_report(
        panels=panels,
        max_direction_difference=max_direction_difference,
        max_bin_difference=max_bin_difference,
        output_paths=output_paths,
    )
    print(f"分箱与样本数汇总：{bins_path}")
    print(f"图注：{caption_path}")
    print(f"QA 报告：{qa_path}")
    print("完成：两张图均只使用冻结 Stable-50 的独立测试集逐样本 SHAP。")


if __name__ == "__main__":
    main()
