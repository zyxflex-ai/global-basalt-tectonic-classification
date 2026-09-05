# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 34_plot_shap_direction_2x2.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
"""Documentation for this analysis component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import font_manager
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from xgboost import XGBClassifier
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
OUTPUT_DIR = PROJECT_DIR / "05_results" / "figures" / "shap_direction"
CACHE_FILE = OUTPUT_DIR / "stable50_independent_test_shap_values_v4.npz"

FIGURE_GROUPS = [
    (
        ["CAB", "IAB", "IOAB", "BABB"],
        "shap_direction_arc_4classes_2x2_stable50_v4",
    ),
    (
        ["MORB", "OIB", "OPB", "CFB"],
        "shap_direction_nonarc_4classes_2x2_stable50_v4",
    ),
]

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

FEATURE_NAME_MAP = {
    "SIO2(WT%)": r"$\mathrm{SiO_2}$",
    "TIO2(WT%)": r"$\mathrm{TiO_2}$",
    "AL2O3(WT%)": r"$\mathrm{Al_2O_3}$",
    "FE_TOTAL(WT%)": "全铁",
    "CAO(WT%)": r"$\mathrm{CaO}$",
    "MGO(WT%)": r"$\mathrm{MgO}$",
    "MNO(WT%)": r"$\mathrm{MnO}$",
    "K2O(WT%)": r"$\mathrm{K_2O}$",
    "NA2O(WT%)": r"$\mathrm{Na_2O}$",
    "P2O5(WT%)": r"$\mathrm{P_2O_5}$",
    "V(PPM)": r"$\mathrm{V}$",
    "CR(PPM)": r"$\mathrm{Cr}$",
    "NI(PPM)": r"$\mathrm{Ni}$",
    "RB(PPM)": r"$\mathrm{Rb}$",
    "SR(PPM)": r"$\mathrm{Sr}$",
    "Y(PPM)": r"$\mathrm{Y}$",
    "ZR(PPM)": r"$\mathrm{Zr}$",
    "NB(PPM)": r"$\mathrm{Nb}$",
    "BA(PPM)": r"$\mathrm{Ba}$",
}

TOP_N = 8
FIGSIZE = (7.2, 7.0)  # Analysis step.
DPI = 300
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
POINT_SIZE = 7.0
POINT_ALPHA = 0.58
PLOT_SEED = 42
SHAP_VALIDATION_ATOL = 2e-6
SHAP_VALIDATION_RTOL = 2e-6
FEATURE_CMAP = mpl.colormaps["viridis"]

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


# ============================================================
# Analysis step.
# ============================================================


def select_chinese_font() -> str:
    """Documentation for this analysis component."""
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
    raise RuntimeError(
        "未找到可用中文字体。请安装思源宋体/黑体、Noto CJK、宋体或黑体。"
    )


CHINESE_FONT = select_chinese_font()

plt.rcParams.update(
    {
        # Analysis step.
        "font.family": CHINESE_FONT,
        "axes.unicode_minus": False,
        # Analysis step.
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    required = [TRAIN_FILE, TEST_FILE, DIRECTION_SUMMARY_FILE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(missing))

    train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig", low_memory=False)
    test = pd.read_csv(TEST_FILE, encoding="utf-8-sig", low_memory=False)
    direction_summary = pd.read_csv(
        DIRECTION_SUMMARY_FILE, encoding="utf-8-sig", low_memory=False
    )

    for name, data in [("训练集", train), ("独立测试集", test)]:
        missing_columns = [column for column in ["LABEL", *FEATURES] if column not in data]
        if missing_columns:
            raise RuntimeError(f"{name}缺少字段：{missing_columns}")
        unknown_labels = sorted(set(data["LABEL"].dropna()) - set(CLASSES))
        if unknown_labels:
            raise RuntimeError(f"{name}含未知类别：{unknown_labels}")
        if data["LABEL"].isna().any():
            raise RuntimeError(f"{name}的 LABEL 含缺失值。")

    X_test_values = test[FEATURES].to_numpy(dtype=np.float64, copy=True)
    return train, test, direction_summary, X_test_values


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    missing_classes = [index for index in range(len(CLASSES)) if index not in counts]
    if missing_classes:
        raise RuntimeError(f"训练集缺少类别索引：{missing_classes}")

    n_samples = len(y)
    n_classes = len(CLASSES)
    class_weights = {
        class_id: n_samples / (n_classes * counts[class_id])
        for class_id in range(n_classes)
    }
    return np.asarray([class_weights[value] for value in y], dtype=float)


def normalize_multiclass_shap(raw_shap: object, n_samples: int) -> np.ndarray:
    """Documentation for this analysis component."""
    if isinstance(raw_shap, list):
        shap_values = np.stack(raw_shap, axis=-1)
    else:
        array = np.asarray(raw_shap)
        if array.ndim != 3:
            raise RuntimeError(f"无法识别 SHAP 维度：{array.shape}")

        if array.shape == (n_samples, len(FEATURES), len(CLASSES)):
            shap_values = array
        elif array.shape == (n_samples, len(CLASSES), len(FEATURES)):
            shap_values = np.transpose(array, (0, 2, 1))
        elif array.shape == (len(CLASSES), n_samples, len(FEATURES)):
            shap_values = np.transpose(array, (1, 2, 0))
        else:
            raise RuntimeError(f"无法识别 SHAP shape：{array.shape}")

    expected_shape = (n_samples, len(FEATURES), len(CLASSES))
    if shap_values.shape != expected_shape:
        raise RuntimeError(
            f"SHAP shape 不符合预期：{shap_values.shape}，预期 {expected_shape}"
        )
    if not np.isfinite(shap_values).all():
        raise RuntimeError("SHAP 数组含 NaN 或无穷值。")
    return np.asarray(shap_values, dtype=np.float32)


def compute_frozen_shap(
    train: pd.DataFrame, test: pd.DataFrame, X_test_values: np.ndarray
) -> np.ndarray:
    """Documentation for this analysis component."""
    y_train = train["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
    sample_weight = make_sample_weights(y_train)

    model = XGBClassifier(**FROZEN_MODEL_PARAMS)
    print("缓存不存在：按 23 号脚本的冻结参数重新拟合模型（不调参）……")
    model.fit(train[FEATURES].copy(), y_train, sample_weight=sample_weight)

    print("仅对独立测试集计算逐样本 SHAP……")
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(test[FEATURES].copy())
    return normalize_multiclass_shap(raw_shap, len(test))


def validate_against_direction_summary(
    shap_values: np.ndarray, direction_summary: pd.DataFrame
) -> float:
    required_columns = {"class", "feature", "mean_abs_shap", "class_rank"}
    missing_columns = required_columns - set(direction_summary.columns)
    if missing_columns:
        raise RuntimeError(f"方向汇总表缺少字段：{sorted(missing_columns)}")

    reference = direction_summary.copy()
    reference["class"] = reference["class"].astype(str).str.strip()
    reference["feature"] = reference["feature"].astype(str).str.strip()
    reference["mean_abs_shap"] = pd.to_numeric(
        reference["mean_abs_shap"], errors="raise"
    )
    if reference.duplicated(["class", "feature"]).any():
        raise RuntimeError("方向汇总表存在重复的 class-feature 组合。")

    expected_pairs = {(class_name, feature) for class_name in CLASSES for feature in FEATURES}
    actual_pairs = set(zip(reference["class"], reference["feature"]))
    if actual_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - actual_pairs)
        extra_pairs = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(
            "方向汇总表的类别/特征组合不完整。"
            f"\n缺少：{missing_pairs[:8]}\n多出：{extra_pairs[:8]}"
        )

    reference_lookup = reference.set_index(["class", "feature"])["mean_abs_shap"]
    actual_mean_abs = np.mean(np.abs(shap_values), axis=0)  # features × classes
    differences = []
    reference_values = []
    actual_values = []

    for class_id, class_name in enumerate(CLASSES):
        for feature_id, feature in enumerate(FEATURES):
            expected = float(reference_lookup.loc[(class_name, feature)])
            actual = float(actual_mean_abs[feature_id, class_id])
            reference_values.append(expected)
            actual_values.append(actual)
            differences.append(abs(expected - actual))

    reference_array = np.asarray(reference_values)
    actual_array = np.asarray(actual_values)
    max_difference = float(max(differences))
    if not np.allclose(
        actual_array,
        reference_array,
        atol=SHAP_VALIDATION_ATOL,
        rtol=SHAP_VALIDATION_RTOL,
    ):
        worst = int(np.argmax(np.abs(actual_array - reference_array)))
        raise RuntimeError(
            "重新得到的逐样本 SHAP 与 23 号脚本方向汇总表不一致，已停止出图。"
            f"\n最大绝对差：{max_difference:.8g}"
            f"\n最差项：actual={actual_array[worst]:.8g}, "
            f"reference={reference_array[worst]:.8g}"
        )
    return max_difference


def save_cache(shap_values: np.ndarray, X_test_values: np.ndarray) -> None:
    metadata = {
        "source_script": "23_shap_direction_class_specific.py",
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "direction_summary_file": str(DIRECTION_SUMMARY_FILE),
        "frozen_model_params": FROZEN_MODEL_PARAMS,
    }
    np.savez_compressed(
        CACHE_FILE,
        shap_values=np.asarray(shap_values, dtype=np.float32),
        feature_values=np.asarray(X_test_values, dtype=np.float64),
        feature_names=np.asarray(FEATURES, dtype="U32"),
        class_names=np.asarray(CLASSES, dtype="U8"),
        train_sha256=np.asarray(file_sha256(TRAIN_FILE)),
        test_sha256=np.asarray(file_sha256(TEST_FILE)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(f"已保存逐样本 SHAP 与对应原始特征值缓存：{CACHE_FILE}")


def load_and_validate_cache(X_test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CACHE_FILE, allow_pickle=False) as cache:
        required_keys = {
            "shap_values",
            "feature_values",
            "feature_names",
            "class_names",
            "train_sha256",
            "test_sha256",
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

    expected_shape = (X_test_values.shape[0], len(FEATURES), len(CLASSES))
    if shap_values.shape != expected_shape:
        raise RuntimeError(
            f"SHAP 缓存 shape 错误：{shap_values.shape}，预期 {expected_shape}"
        )
    if cached_feature_values.shape != X_test_values.shape:
        raise RuntimeError("SHAP 缓存中的原始特征矩阵 shape 与当前测试集不一致。")
    if cached_features != FEATURES or cached_classes != CLASSES:
        raise RuntimeError("SHAP 缓存中的特征或类别顺序与冻结定义不一致。")
    if cached_train_hash != file_sha256(TRAIN_FILE):
        raise RuntimeError("训练集已变化，缓存失效；确认后用 --recompute 重建。")
    if cached_test_hash != file_sha256(TEST_FILE):
        raise RuntimeError("独立测试集已变化，缓存失效；确认后用 --recompute 重建。")
    if not np.allclose(cached_feature_values, X_test_values, rtol=0, atol=0, equal_nan=True):
        raise RuntimeError("缓存原始特征值与当前独立测试集逐项不一致。")
    if not np.isfinite(shap_values).all():
        raise RuntimeError("SHAP 缓存含 NaN 或无穷值。")

    return shap_values, cached_feature_values


def obtain_verified_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    direction_summary: pd.DataFrame,
    X_test_values: np.ndarray,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if recompute or not CACHE_FILE.is_file():
        shap_values = compute_frozen_shap(train, test, X_test_values)
        max_difference = validate_against_direction_summary(shap_values, direction_summary)
        save_cache(shap_values, X_test_values)
    else:
        print(f"读取已计算的逐样本 SHAP 缓存：{CACHE_FILE}")

    # Analysis step.
    shap_values, feature_values = load_and_validate_cache(X_test_values)
    max_difference = validate_against_direction_summary(shap_values, direction_summary)
    return shap_values, feature_values, max_difference


# ============================================================
# Analysis step.
# ============================================================


def get_top_feature_indices(
    direction_summary: pd.DataFrame, class_name: str
) -> list[int]:
    subset = direction_summary.loc[
        direction_summary["class"].astype(str).str.strip().eq(class_name)
    ].copy()
    subset["feature"] = subset["feature"].astype(str).str.strip()
    subset["class_rank"] = pd.to_numeric(subset["class_rank"], errors="raise")
    top_features = (
        subset.sort_values("class_rank", ascending=True).head(TOP_N)["feature"].tolist()
    )
    if len(top_features) != TOP_N or len(set(top_features)) != TOP_N:
        raise RuntimeError(f"{class_name} 无法得到唯一的 Top {TOP_N} 特征。")
    unknown = [feature for feature in top_features if feature not in FEATURES]
    if unknown:
        raise RuntimeError(f"{class_name} Top 特征不在冻结特征表中：{unknown}")
    return [FEATURES.index(feature) for feature in top_features]


def collect_missing_glyph_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return sorted(
        {
            str(item.message)
            for item in caught
            if "Glyph" in str(item.message) and "missing from font" in str(item.message)
        }
    )


def draw_figure(
    class_group: list[str],
    output_stem: str,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    direction_summary: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained")
    display_names = [FEATURE_NAME_MAP[feature] for feature in FEATURES]

    for panel_index, (axis, class_name) in enumerate(zip(axes.flat, class_group)):
        class_id = CLASSES.index(class_name)
        top_indices = get_top_feature_indices(direction_summary, class_name)
        explanation = shap.Explanation(
            values=shap_values[:, :, class_id],
            data=feature_values,
            feature_names=display_names,
        )

        # Analysis step.
        np.random.seed(PLOT_SEED + class_id)
        shap.plots.beeswarm(
            explanation,
            max_display=TOP_N,
            order=np.asarray(top_indices, dtype=int),
            color=FEATURE_CMAP,
            axis_color="#333333",
            alpha=POINT_ALPHA,
            ax=axis,
            show=False,
            color_bar=False,
            s=POINT_SIZE,
            plot_size=None,
            group_remaining_features=False,
        )

        # Analysis step.
        for collection in axis.collections:
            if isinstance(collection, PathCollection):
                collection.set_rasterized(True)

        axis.axvline(0, color="#4A4A4A", linewidth=0.85, zorder=0)
        axis.set_title(
            f"{PANEL_LABELS[panel_index]}  {class_name}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            fontfamily="Times New Roman",
            pad=7,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.tick_params(axis="x", labelsize=7.6, length=3, width=0.7, pad=2)
        axis.tick_params(axis="y", labelsize=8.2, length=0, pad=3)
        for tick_label in axis.get_xticklabels():
            tick_label.set_fontfamily("Times New Roman")
        axis.grid(axis="x", color="#D8D8D8", linestyle="--", linewidth=0.45)
        axis.grid(axis="y", color="#E9E9E9", linestyle="-", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#666666")
        axis.spines["bottom"].set_linewidth(0.7)
        axis.margins(x=0.025)

    fig.supxlabel(
        "SHAP 值（负值：推离该类别；正值：推向该类别）",
        fontsize=9.8,
    )

    color_mappable = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_CMAP)
    color_mappable.set_array([])
    colorbar = fig.colorbar(
        color_mappable,
        ax=axes.ravel().tolist(),
        fraction=0.032,
        pad=0.018,
        aspect=34,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["低", "高"])
    colorbar.set_label("原始特征值（各特征内相对高低）", fontsize=8.8, labelpad=7)
    colorbar.ax.tick_params(labelsize=8.2, length=0, pad=2)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor("#777777")

    with warnings.catch_warnings(record=True) as caught_draw:
        warnings.simplefilter("always")
        fig.canvas.draw()
    missing_glyphs = collect_missing_glyph_warnings(caught_draw)
    if missing_glyphs:
        plt.close(fig)
        raise RuntimeError("检测到字体缺字，已停止导出：\n" + "\n".join(missing_glyphs))

    output_paths = [OUTPUT_DIR / f"{output_stem}.png", OUTPUT_DIR / f"{output_stem}.pdf"]
    with warnings.catch_warnings(record=True) as caught_save:
        warnings.simplefilter("always")
        fig.savefig(output_paths[0], dpi=DPI)
        fig.savefig(output_paths[1])
    missing_glyphs = collect_missing_glyph_warnings(caught_save)
    plt.close(fig)
    if missing_glyphs:
        raise RuntimeError("导出时检测到字体缺字：\n" + "\n".join(missing_glyphs))

    return output_paths, missing_glyphs


# ============================================================
# Analysis step.
# ============================================================


def write_caption_file() -> Path:
    caption_path = OUTPUT_DIR / "shap_direction_captions_stable50_v4.txt"
    content = """图X 弧相关构造环境的类别特异 SHAP 方向分布
（a）CAB；（b）IAB；（c）IOAB；（d）BABB。

图X 非弧及板内构造环境的类别特异 SHAP 方向分布
（a）MORB；（b）OIB；（c）OPB；（d）CFB。

统一图注：
各子图展示对应类别平均绝对 SHAP 值排名前 8 的地球化学特征。每个点代表独立测试集中的一个样本；横坐标为该特征对对应类别模型输出的 SHAP 贡献，正值推动模型判为该类别，负值使模型远离该类别。颜色表示同一特征原始实测值的相对高低（低值至高值），不同特征的颜色不代表相同的绝对数值；灰色点表示该原始特征值缺失。各子图横轴范围独立，仅用于展示对应类别内部的贡献分布。SHAP 反映模型预测贡献，不代表因果作用。
"""
    caption_path.write_text(content, encoding="utf-8-sig")
    return caption_path


def write_qa_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    max_difference: float,
    output_paths: list[Path],
) -> Path:
    report_path = OUTPUT_DIR / "shap_direction_qa_stable50_v4.txt"
    missing_count = int(np.isnan(feature_values).sum())
    report_lines = [
        "Stable-50 XGBoost SHAP 方向图 QA 报告",
        "=" * 64,
        f"训练集 shape: {train.shape}",
        f"独立测试集 shape: {test.shape}",
        f"逐样本 SHAP shape: {shap_values.shape}",
        f"对应原始特征值 shape: {feature_values.shape}",
        f"原始特征缺失单元格数: {missing_count}",
        f"与 23 号方向汇总表 mean(|SHAP|) 的最大绝对差: {max_difference:.8g}",
        f"中文字体: {CHINESE_FONT}",
        "数学下标: STIX mathtext",
        f"图尺寸: {FIGSIZE[0]} × {FIGSIZE[1]} in",
        f"PNG 分辨率: {DPI} dpi",
        "PDF 字体类型: TrueType (fonttype 42)",
        "点云 PDF 策略: 栅格化；坐标轴和文字保持矢量",
        "缺字检查: PASS",
        "颜色映射: viridis（低→高）；灰色表示原始值缺失",
        "SHAP 方向: 0 基线；负值推离类别，正值推向类别",
        "",
        "输出文件：",
    ]
    report_lines.extend(f"- {path} ({path.stat().st_size:,} bytes)" for path in output_paths)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")
    return report_path


# ============================================================
# Analysis step.
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 Stable-50 XGBoost 的 8 类 SHAP beeswarm 方向图。"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="忽略现有逐样本 SHAP 缓存，按同一冻结模型重新计算并覆盖缓存。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Stable-50 XGBoost：8 类 SHAP beeswarm 方向图")
    print("=" * 88)
    print(f"训练集：{TRAIN_FILE}")
    print(f"独立测试集：{TEST_FILE}")
    print(f"23 号方向汇总表：{DIRECTION_SUMMARY_FILE}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"中文字体：{CHINESE_FONT}")
    print("模型状态：冻结参数；不调参")

    train, test, direction_summary, X_test_values = load_inputs()
    shap_values, feature_values, max_difference = obtain_verified_values(
        train=train,
        test=test,
        direction_summary=direction_summary,
        X_test_values=X_test_values,
        recompute=args.recompute,
    )

    print(f"mean(|SHAP|) 复核通过；最大绝对差 = {max_difference:.8g}")
    output_paths: list[Path] = []
    for class_group, output_stem in FIGURE_GROUPS:
        paths, _ = draw_figure(
            class_group=class_group,
            output_stem=output_stem,
            shap_values=shap_values,
            feature_values=feature_values,
            direction_summary=direction_summary,
        )
        output_paths.extend(paths)
        print(f"已输出：{paths[0]}")
        print(f"已输出：{paths[1]}")

    caption_path = write_caption_file()
    output_paths.append(caption_path)
    qa_path = write_qa_report(
        train=train,
        test=test,
        shap_values=shap_values,
        feature_values=feature_values,
        max_difference=max_difference,
        output_paths=output_paths,
    )

    print(f"图注：{caption_path}")
    print(f"QA 报告：{qa_path}")
    print("完成：两张 2×2 图均使用独立测试集逐样本 SHAP 与对应原始特征值。")


if __name__ == "__main__":
    main()
