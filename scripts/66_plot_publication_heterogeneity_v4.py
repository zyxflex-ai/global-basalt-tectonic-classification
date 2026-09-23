"""Plot publication-level transfer heterogeneity on the frozen holdout set."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "05_results" / "publication_heterogeneity"
OUTPUT_STEM = INPUT_DIR / "figure_publication_heterogeneity_v4"

CLASS_ORDER = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
COLORS = {
    "CAB": "#4477AA",
    "IAB": "#66CCEE",
    "IOAB": "#228833",
    "BABB": "#CCBB44",
    "MORB": "#EE6677",
    "OIB": "#AA3377",
    "OPB": "#999999",
    "CFB": "#332288",
}

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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def main() -> None:
    publication = pd.read_csv(INPUT_DIR / "publication_level_performance_v4.csv")
    publication_class = pd.read_csv(INPUT_DIR / "publication_class_performance_v4.csv")
    if len(publication) != 462:
        raise ValueError(f"Expected 462 publication groups, found {len(publication)}")
    if int(publication["n_samples"].sum()) != 10086:
        raise ValueError("Publication counts do not sum to the 10,086-sample holdout.")
    if (publication["n_samples"] <= 0).any():
        raise ValueError("Log-scaled publication sizes must all be positive.")
    if set(publication_class["TRUE_LABEL"]) != set(CLASS_ORDER):
        raise ValueError("Unexpected tectonic-class labels in plotting data.")

    fig = plt.figure(figsize=(7.20, 3.55), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    positions = np.arange(len(CLASS_ORDER))
    box_data = [
        publication_class.loc[
            publication_class["TRUE_LABEL"] == label, "accuracy"
        ].to_numpy()
        for label in CLASS_ORDER
    ]
    boxes = ax_a.boxplot(
        box_data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, label in zip(boxes["boxes"], CLASS_ORDER):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.82)
    for x_value, label in zip(positions, CLASS_ORDER):
        subset = publication_class.loc[
            publication_class["TRUE_LABEL"] == label
        ].sort_values(["accuracy", "n_samples"])
        jitter = np.linspace(-0.055, 0.055, len(subset))
        ax_a.scatter(
            np.full(len(subset), x_value) + jitter,
            subset["accuracy"],
            s=13,
            color=COLORS[label],
            alpha=0.30,
            edgecolors="none",
            rasterized=True,
        )
    ax_a.set_xticks(positions, CLASS_ORDER, rotation=32, ha="right", rotation_mode="anchor")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.set_ylabel("Accuracy within publication and class")
    ax_a.set_title("Transfer performance varies among source publications", loc="left", pad=7)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    panel_label(ax_a, "a")

    log_group_size = np.log10(publication["n_samples"].clip(lower=1))
    ax_b.scatter(
        log_group_size,
        publication["accuracy"],
        s=18,
        color="#6B7280",
        alpha=0.48,
        edgecolors="none",
        rasterized=True,
    )
    ax_b.axhline(
        publication["accuracy"].mean(),
        color="#B44B3E",
        linewidth=1.2,
        label="Publication-equal mean",
    )
    ax_b.set_xlim(-0.05, log_group_size.max() + 0.08)
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_xlabel("log10 samples per publication")
    ax_b.set_ylabel("Publication-level accuracy")
    ax_b.set_title("Heterogeneity is not explained by group size", loc="left", pad=7)
    ax_b.grid(color="#D9D9D9", linewidth=0.55)
    ax_b.text(
        0.04,
        0.08,
        "Spearman ρ = -0.09\n462 publications",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        color="#333333",
        fontsize=7.0,
    )
    ax_b.legend(loc="lower right")
    panel_label(ax_b, "b")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        f"{OUTPUT_STEM}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
