"""Create the publication figure for the split-strategy comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _project_paths import FINAL_DIR


INPUT_DIR = FINAL_DIR / "05_results" / "validation_bias"
OUTPUT_DIR = INPUT_DIR / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
}
METRIC_ORDER = ["accuracy", "balanced_accuracy", "macro_f1"]
CLASS_ORDER = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
SCHEME_LABELS = {
    "publication_grouped": "Publication-grouped",
    "sample_random": "Sample-random",
}
COLORS = {
    "publication_grouped": "#2C6EAA",
    "sample_random": "#D97732",
    "difference": "#B44B3E",
    "neutral": "#767676",
}
MARKERS = {"publication_grouped": "o", "sample_random": "^"}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    return float(values.median()), float(values.quantile(0.25)), float(values.quantile(0.75))


def main() -> None:
    metrics = pd.read_csv(INPUT_DIR / "random_vs_grouped_cv_metrics_v4.csv")
    class_f1 = pd.read_csv(INPUT_DIR / "random_vs_grouped_cv_class_f1_v4.csv")
    audit = pd.read_csv(INPUT_DIR / "random_vs_grouped_cv_split_audit_v4.csv")
    bootstrap = pd.read_csv(INPUT_DIR / "random_vs_grouped_cv_cluster_bootstrap_v4.csv")

    repeated = metrics[
        (metrics["level"] == "oof")
        & (metrics["weighting"] == "sample")
        & (metrics["repeat"] >= 0)
    ].copy()
    repeated_class = class_f1[class_f1["repeat"] >= 0].copy()
    repeated_audit = audit[audit["repeat"] >= 0].copy()

    # 183 mm by 143 mm, a double-column manuscript figure.
    fig = plt.figure(figsize=(7.2047, 5.6299), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.92,
        wspace=0.34,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a, repeated OOF performance. Points are repeated partitions, not independent studies.
    x_base = np.arange(len(METRIC_ORDER), dtype=float)
    offset = 0.17
    for scheme, direction in (("publication_grouped", -1), ("sample_random", 1)):
        for metric_index, metric in enumerate(METRIC_ORDER):
            values = repeated.loc[repeated["scheme"] == scheme, metric].to_numpy()
            x = x_base[metric_index] + direction * offset
            jitter = np.linspace(-0.035, 0.035, num=len(values))
            ax_a.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=17,
                color=COLORS[scheme],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                marker=MARKERS[scheme],
                zorder=2,
            )
            median, q25, q75 = median_iqr(pd.Series(values))
            ax_a.errorbar(
                x,
                median,
                yerr=[[median - q25], [q75 - median]],
                fmt=MARKERS[scheme],
                color=COLORS[scheme],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=5.0,
                capsize=3,
                linewidth=1.25,
                zorder=3,
            )
    ax_a.set_xticks(x_base, [METRIC_LABELS[name] for name in METRIC_ORDER])
    ax_a.set_ylabel("Out-of-fold score")
    ax_a.set_ylim(0.63, 0.95)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax_a.set_title("Repeated fivefold cross-validation", loc="left", pad=7)
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            markersize=5,
            color=COLORS[scheme],
            marker=MARKERS[scheme],
            label=SCHEME_LABELS[scheme],
        )
        for scheme in ("publication_grouped", "sample_random")
    ]
    ax_a.legend(handles=handles, loc="lower left", ncol=1, handletextpad=0.5)
    add_panel_label(ax_a, "a")

    # b, publication-cluster bootstrap interval for the primary fixed comparison.
    b_summary = (
        bootstrap.groupby("metric")["difference_random_minus_grouped"]
        .agg(
            median="median",
            lower=lambda values: values.quantile(0.025),
            upper=lambda values: values.quantile(0.975),
        )
        .reindex(METRIC_ORDER)
    )
    y_pos = np.arange(len(METRIC_ORDER))[::-1]
    ax_b.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    for y, metric in zip(y_pos, METRIC_ORDER):
        row = b_summary.loc[metric]
        ax_b.errorbar(
            row["median"],
            y,
            xerr=[[row["median"] - row["lower"]], [row["upper"] - row["median"]]],
            fmt="o",
            color=COLORS["difference"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=5.2,
            capsize=3,
            linewidth=1.4,
        )
        ax_b.text(
            row["upper"] + 0.006,
            y,
            f"{row['median']:.3f}",
            va="center",
            ha="left",
            fontsize=6.8,
        )
    ax_b.set_yticks(y_pos, [METRIC_LABELS[name] for name in METRIC_ORDER])
    ax_b.set_xlabel("Score inflation, sample-random minus grouped")
    ax_b.set_xlim(-0.01, 0.255)
    ax_b.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax_b.set_title("Publication-cluster bootstrap", loc="left", pad=7)
    add_panel_label(ax_b, "b")

    # c, class-specific inflation across repeated partitions.
    pivot = repeated_class.pivot_table(
        index=["repeat", "class"], columns="scheme", values="f1"
    ).reset_index()
    pivot["difference"] = pivot["sample_random"] - pivot["publication_grouped"]
    class_summary = (
        pivot.groupby("class")["difference"]
        .agg(
            median="median",
            lower=lambda values: values.quantile(0.25),
            upper=lambda values: values.quantile(0.75),
        )
        .reindex(CLASS_ORDER)
    )
    class_y = np.arange(len(CLASS_ORDER))[::-1]
    ax_c.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    for y, class_name in zip(class_y, CLASS_ORDER):
        row = class_summary.loc[class_name]
        ax_c.errorbar(
            row["median"],
            y,
            xerr=[[row["median"] - row["lower"]], [row["upper"] - row["median"]]],
            fmt="o",
            color=COLORS["difference"],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=4.8,
            capsize=2.5,
            linewidth=1.1,
        )
    ax_c.set_yticks(class_y, CLASS_ORDER)
    ax_c.set_xlabel("Class F1 inflation")
    ax_c.set_xlim(-0.02, 0.48)
    ax_c.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax_c.set_title("Class-specific effect of split strategy", loc="left", pad=7)
    add_panel_label(ax_c, "c")

    # d, the leakage mechanism measured directly at the fold level.
    for scheme, x in (("publication_grouped", 0), ("sample_random", 1)):
        values = (
            100
            * repeated_audit.loc[
                repeated_audit["scheme"] == scheme,
                "validation_sample_leakage_fraction",
            ].to_numpy()
        )
        jitter = np.linspace(-0.075, 0.075, num=len(values))
        ax_d.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=13,
            color=COLORS[scheme],
            alpha=0.6,
            edgecolor="none",
            marker=MARKERS[scheme],
        )
        median = float(np.median(values))
        ax_d.plot([x - 0.16, x + 0.16], [median, median], color="black", linewidth=1.25)
        ax_d.text(x, min(104, median + 4), f"{median:.1f}%", ha="center", va="bottom", fontsize=7)
    ax_d.set_xticks([0, 1], ["Publication-\ngrouped", "Sample-\nrandom"])
    ax_d.set_ylabel("Validation samples with publication overlap (%)")
    ax_d.set_ylim(-4, 108)
    ax_d.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax_d.set_title("Direct audit of source leakage", loc="left", pad=7)
    ax_d.text(
        -0.21,
        1.08,
        "d",
        transform=ax_d.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    output_stem = OUTPUT_DIR / "figure_random_vs_grouped_cv_v4"
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_rows = []
    for scheme in ("publication_grouped", "sample_random"):
        scheme_rows = repeated[repeated["scheme"] == scheme]
        for metric in METRIC_ORDER:
            median, q25, q75 = median_iqr(scheme_rows[metric])
            summary_rows.append(
                {
                    "scheme": scheme,
                    "metric": metric,
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "n_repeated_partitions": len(scheme_rows),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        INPUT_DIR / "random_vs_grouped_cv_repeated_summary_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
