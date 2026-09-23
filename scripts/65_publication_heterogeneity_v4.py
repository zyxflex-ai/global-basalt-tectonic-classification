from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "03_model_data" / "Basalt_test_stable50_main_v4.csv"
PRED_PATH = ROOT / "05_results" / "final_test" / "final_independent_test_predictions_v4.csv"
OUT_DIR = ROOT / "05_results" / "publication_heterogeneity"

CLASS_ORDER = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]


def percentile_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    pred = pd.read_csv(PRED_PATH, low_memory=False)

    metadata = test[
        ["GROUP_ID", "CV_GROUP_V4", "DATABASE", "PAPER_YEAR", "N_STABLE50"]
    ].copy()
    merged = pred.merge(
        metadata,
        on="GROUP_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_META"),
    )
    if merged["DATABASE"].isna().any():
        raise RuntimeError("Prediction rows could not all be matched to holdout metadata.")
    if not (merged["CV_GROUP_V4"] == merged["CV_GROUP_V4_META"]).all():
        raise RuntimeError("CV_GROUP_V4 mismatch between predictions and holdout metadata.")
    merged["XGB_CORRECT"] = merged["XGB_CORRECT"].astype(str).str.lower().eq("true")

    publication_class = (
        merged.groupby(["CV_GROUP_V4", "TRUE_LABEL"], observed=True)
        .agg(
            n_samples=("XGB_CORRECT", "size"),
            accuracy=("XGB_CORRECT", "mean"),
            mean_max_probability=("XGB_MAX_PROB", "mean"),
            mean_entropy=("XGB_ENTROPY", "mean"),
            mean_element_coverage=("N_STABLE50", "mean"),
            publication_year=("PAPER_YEAR", "median"),
            database=("DATABASE", lambda x: "+".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    publication = (
        merged.groupby("CV_GROUP_V4", observed=True)
        .agg(
            n_samples=("XGB_CORRECT", "size"),
            n_classes=("TRUE_LABEL", "nunique"),
            accuracy=("XGB_CORRECT", "mean"),
            mean_max_probability=("XGB_MAX_PROB", "mean"),
            mean_entropy=("XGB_ENTROPY", "mean"),
            mean_element_coverage=("N_STABLE50", "mean"),
            publication_year=("PAPER_YEAR", "median"),
            database=("DATABASE", lambda x: "+".join(sorted(set(map(str, x))))),
            majority_class=("TRUE_LABEL", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )

    rng = np.random.default_rng(20260920)
    summary_rows: list[dict[str, float | int | str]] = []
    for label in CLASS_ORDER:
        subset = publication_class.loc[publication_class["TRUE_LABEL"] == label]
        ci_low, ci_high = percentile_ci(subset["accuracy"].to_numpy(), rng)
        sample_subset = merged.loc[merged["TRUE_LABEL"] == label]
        summary_rows.append(
            {
                "tectonic_class": label,
                "n_publications": int(len(subset)),
                "n_samples": int(subset["n_samples"].sum()),
                "publication_equal_mean_accuracy": float(subset["accuracy"].mean()),
                "publication_equal_mean_accuracy_ci_low": ci_low,
                "publication_equal_mean_accuracy_ci_high": ci_high,
                "publication_median_accuracy": float(subset["accuracy"].median()),
                "publication_accuracy_q25": float(subset["accuracy"].quantile(0.25)),
                "publication_accuracy_q75": float(subset["accuracy"].quantile(0.75)),
                "sample_weighted_accuracy": float(sample_subset["XGB_CORRECT"].mean()),
            }
        )
    class_summary = pd.DataFrame(summary_rows)

    overall_ci = percentile_ci(publication["accuracy"].to_numpy(), rng)
    correlations = []
    for variable in ["n_samples", "mean_element_coverage", "publication_year"]:
        valid = publication[[variable, "accuracy"]].dropna()
        rho = valid[variable].rank(method="average").corr(
            valid["accuracy"].rank(method="average")
        )
        correlations.append(
            {
                "variable": variable,
                "n_publications": int(len(valid)),
                "spearman_rho": float(rho),
            }
        )

    overall_summary = pd.DataFrame(
        [
            {
                "n_publications": int(len(publication)),
                "n_samples": int(publication["n_samples"].sum()),
                "publication_equal_mean_accuracy": float(publication["accuracy"].mean()),
                "publication_equal_mean_accuracy_ci_low": overall_ci[0],
                "publication_equal_mean_accuracy_ci_high": overall_ci[1],
                "publication_median_accuracy": float(publication["accuracy"].median()),
                "publication_accuracy_q25": float(publication["accuracy"].quantile(0.25)),
                "publication_accuracy_q75": float(publication["accuracy"].quantile(0.75)),
                "fraction_publications_below_0_5": float((publication["accuracy"] < 0.5).mean()),
                "fraction_publications_zero_accuracy": float((publication["accuracy"] == 0).mean()),
                "fraction_publications_perfect_accuracy": float((publication["accuracy"] == 1).mean()),
                "sample_weighted_accuracy": float(merged["XGB_CORRECT"].mean()),
            }
        ]
    )
    database_summary = (
        publication.groupby("database", observed=True)
        .agg(
            n_publications=("CV_GROUP_V4", "size"),
            n_samples=("n_samples", "sum"),
            publication_equal_mean_accuracy=("accuracy", "mean"),
            publication_median_accuracy=("accuracy", "median"),
        )
        .reset_index()
    )

    publication.to_csv(OUT_DIR / "publication_level_performance_v4.csv", index=False)
    publication_class.to_csv(OUT_DIR / "publication_class_performance_v4.csv", index=False)
    class_summary.to_csv(OUT_DIR / "publication_class_summary_v4.csv", index=False)
    overall_summary.to_csv(OUT_DIR / "publication_overall_summary_v4.csv", index=False)
    pd.DataFrame(correlations).to_csv(
        OUT_DIR / "publication_accuracy_correlations_v4.csv", index=False
    )
    database_summary.to_csv(OUT_DIR / "publication_database_summary_v4.csv", index=False)

    print(overall_summary.to_string(index=False))
    print("\nClass-level publication-equal summary")
    print(class_summary.to_string(index=False))
    print("\nSpearman diagnostics")
    print(pd.DataFrame(correlations).to_string(index=False))
    print("\nDatabase summary")
    print(database_summary.to_string(index=False))


if __name__ == "__main__":
    main()
