from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = ROOT / "figures"
TEMPORAL = ROOT / "05_results" / "temporal_v9"
SOURCE_DATA = ROOT / "05_results" / "figure_source_data"
CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
BLUE, TEAL, GOLD, RED, GREY, LIGHT = "#0F4D92", "#42949E", "#D29B27", "#B64342", "#767676", "#D8D8D8"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.5, "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="top")


def publication_bootstrap(data: pd.DataFrame, seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    values = data.groupby("CV_GROUP_V4", observed=True)["CORRECT"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_ablation() -> pd.DataFrame:
    results = ROOT / "05_results"
    test = pd.read_csv(ROOT / "03_model_data" / "Basalt_test_stable50_main_v4.csv", low_memory=False)
    train = pd.read_csv(ROOT / "03_model_data" / "Basalt_train_stable50_main_v4.csv", low_memory=False)
    combined = pd.concat([train, test], ignore_index=True)
    sets = combined.groupby("CV_GROUP_V4", observed=True)["DATABASE"].agg(lambda values: set(map(str, values)))
    shared = {group for group, databases in sets.items() if len(databases) > 1}
    mixed = pd.read_csv(results / "final_test" / "final_independent_test_predictions_v4.csv", low_memory=False)
    mixed = mixed[["CV_GROUP_V4", "TRUE_LABEL", "XGB_CORRECT"]].copy()
    mixed["DATABASE"] = test["DATABASE"].to_numpy()
    mixed["CORRECT"] = mixed["XGB_CORRECT"].astype(bool)
    mixed = mixed.loc[(mixed["TRUE_LABEL"] == "MORB") & (~mixed["CV_GROUP_V4"].isin(shared))]
    georoc = pd.read_csv(results / "external_validation" / "petdb_diagnostics" / "georoc_model_predictions_by_database_v5.csv", low_memory=False)
    georoc = georoc.loc[(georoc["LABEL"] == "MORB") & georoc["CV_GROUP_V4"].isin(set(test.loc[test["LABEL"] == "MORB", "CV_GROUP_V4"]))]
    rows = []
    seed = 20260921
    for model_name, frame in (("Mixed-source training", mixed), ("GEOROC-only training", georoc)):
        for database in ("GEOROC", "PetDB"):
            subset = frame.loc[frame["DATABASE"] == database]
            estimate, low, high = publication_bootstrap(subset, seed)
            seed += 1
            rows.append({
                "model": model_name, "test_database": database, "n_samples": len(subset),
                "n_publications": subset["CV_GROUP_V4"].nunique(),
                "publication_equal_recall": estimate, "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(TEMPORAL / "time_closed_temporal_overall_v9.csv")
    classes = pd.read_csv(TEMPORAL / "time_closed_temporal_classes_v9.csv")
    ablation_path = SOURCE_DATA / "figure6_morb_source_ablation_v9.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else build_ablation()
    overall.to_csv(SOURCE_DATA / "figure6_time_closed_temporal_overall_v9.csv", index=False)
    classes.to_csv(SOURCE_DATA / "figure6_time_closed_temporal_classes_v9.csv", index=False)
    ablation.to_csv(SOURCE_DATA / "figure6_morb_source_ablation_v9.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[0.9, 1.45])
    ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

    order = ["accuracy", "weighted_f1", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Weighted F1", "Balanced accuracy", "Macro-F1"]
    table = overall.set_index("metric").loc[order]
    y = np.arange(len(order))[::-1]
    ax_a.hlines(y, table["cluster_bootstrap_ci_low"], table["cluster_bootstrap_ci_high"], color=TEAL, linewidth=1.6)
    ax_a.scatter(table["estimate"], y, color=BLUE, s=24, zorder=3)
    ax_a.set_yticks(y, labels); ax_a.set_xlim(0.58, 0.90)
    ax_a.set_xlabel("Time-closed test performance"); ax_a.set_title("2011-2025 publications held out")
    ax_a.grid(axis="x", color=LIGHT, linewidth=0.5); panel_label(ax_a, "a")

    table = classes.set_index("class").loc[CLASSES]
    x = np.arange(len(CLASSES))
    ax_b.scatter(x - 0.10, table["sample_recall"], color=BLUE, s=22, label="Sample-weighted recall", zorder=3)
    publication = table["publication_equal_recall"].to_numpy(float)
    ax_b.errorbar(x + 0.10, publication, yerr=np.vstack([
        publication - table["publication_equal_recall_ci_low"].to_numpy(float),
        table["publication_equal_recall_ci_high"].to_numpy(float) - publication,
    ]), fmt="o", markersize=4, color=GOLD, ecolor=GOLD, elinewidth=1.2, capsize=2,
        label="Publication-equal recall (95% CI)", zorder=3)
    ax_b.set_xticks(x, CLASSES, rotation=35, ha="right", rotation_mode="anchor")
    ax_b.set_ylim(0, 1.03); ax_b.set_ylabel("Recall"); ax_b.set_title("Time-closed transfer by class")
    ax_b.grid(axis="y", color=LIGHT, linewidth=0.5); ax_b.legend(loc="lower left"); panel_label(ax_b, "b")

    positions = {"GEOROC": 1.0, "PetDB": 0.0}
    offsets = {"Mixed-source training": 0.10, "GEOROC-only training": -0.10}
    colors = {"Mixed-source training": BLUE, "GEOROC-only training": RED}
    markers = {"Mixed-source training": "o", "GEOROC-only training": "s"}
    for _, row in ablation.iterrows():
        yi = positions[row["test_database"]] + offsets[row["model"]]
        ax_c.hlines(yi, row["ci_low"], row["ci_high"], color=colors[row["model"]], linewidth=1.8)
        ax_c.scatter(row["publication_equal_recall"], yi, color=colors[row["model"]], marker=markers[row["model"]], s=34, zorder=3)
        ax_c.text(min(row["ci_high"] + 0.025, 0.96), yi, f"n={int(row['n_publications'])} publications", ha="left", va="center", fontsize=8, color=GREY)
    ax_c.set_yticks([1.0, 0.0], ["GEOROC MORB test", "PetDB MORB test"])
    ax_c.set_xlim(0, 1); ax_c.set_ylim(-0.45, 1.45)
    ax_c.set_xlabel("Publication-equal MORB recall (95% publication-bootstrap CI)")
    ax_c.set_title("MORB transfer depends on source representation in training")
    ax_c.grid(axis="x", color=LIGHT, linewidth=0.5)
    ax_c.legend(handles=[
        plt.Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Mixed-source training"),
        plt.Line2D([], [], color=RED, marker="s", linestyle="none", label="GEOROC-only training"),
    ], loc="lower right"); panel_label(ax_c, "c")

    stem = OUT / "figure6_time_closed_temporal_source_robustness_v9"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".eps"), format="eps", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Built {stem} in SVG, PDF, EPS, TIFF and PNG", flush=True)


if __name__ == "__main__":
    main()
