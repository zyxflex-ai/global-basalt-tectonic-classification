"""Support-matched control for the PetDB training ablation.

The original ablation retained 295 GEOROC MORB rows from 20 publications after
PetDB training rows were removed. This script compares that source-specific
restriction with repeated controls that retain the same number of MORB rows and
publications, sampled from the complete database-exclusive training pool. All
non-MORB training rows and both frozen database-exclusive test cohorts are held
fixed. The analysis isolates loss of MORB training support more closely, but it
does not eliminate geographic or analytical-program confounding.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_DEPS = Path(__file__).resolve().parents[1] / "modeldeps"
sys.path.insert(0, str(MODEL_DEPS))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


FINAL = Path(__file__).resolve().parents[1]
TRAIN_PATH = FINAL / "03_model_data" / "Basalt_train_stable50_main_v4.csv"
TEST_PATH = FINAL / "03_model_data" / "Basalt_test_stable50_main_v4.csv"
OUT = FINAL / "05_results" / "chemical_geology_sensitivities"

CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASSES)}
FEATURES = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FE_TOTAL(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "BA(PPM)",
]
N_REPEATS = 20


def build_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=len(CLASSES), n_estimators=1400,
        learning_rate=0.02, max_depth=6, min_child_weight=1, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0,
        tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )


def sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    if set(counts.index) != set(range(len(CLASSES))):
        raise RuntimeError("Training data must contain all eight classes.")
    weights = {i: len(y) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))}
    return np.asarray([weights[int(value)] for value in y], dtype=float)


def fit_and_score(train: pd.DataFrame, cohorts: dict[str, pd.DataFrame], seed: int) -> list[dict[str, object]]:
    y = train["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
    model = build_model(seed)
    model.fit(train[FEATURES], y, sample_weight=sample_weights(y))
    rows = []
    for cohort_name, cohort in cohorts.items():
        pred = model.predict(cohort[FEATURES]).astype(int)
        correct = pred == CLASS_TO_INT["MORB"]
        per_publication = pd.DataFrame(
            {"CV_GROUP_V4": cohort["CV_GROUP_V4"].to_numpy(), "correct": correct}
        ).groupby("CV_GROUP_V4", observed=True)["correct"].mean()
        rows.append(
            {
                "cohort": cohort_name,
                "n_test_samples": len(cohort),
                "n_test_publications": cohort["CV_GROUP_V4"].nunique(),
                "sample_recall": correct.mean(),
                "publication_equal_recall": per_publication.mean(),
            }
        )
    return rows


def support_matched_morb(pool: pd.DataFrame, n_publications: int, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = pool["CV_GROUP_V4"].drop_duplicates().to_numpy()
    for _ in range(10000):
        selected = rng.choice(groups, size=n_publications, replace=False)
        candidate = pool.loc[pool["CV_GROUP_V4"].isin(selected)]
        if len(candidate) < n_samples:
            continue
        one_each = (
            candidate.groupby("CV_GROUP_V4", group_keys=False, observed=True)
            .sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        )
        remaining = candidate.drop(index=one_each.index)
        extra = remaining.sample(
            n=n_samples - n_publications,
            replace=False,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        result = pd.concat([one_each, extra], ignore_index=False)
        if len(result) == n_samples and result["CV_GROUP_V4"].nunique() == n_publications:
            return result.copy()
    raise RuntimeError("Could not draw a support-matched MORB subset.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    combined = pd.concat([train, test], ignore_index=True)
    database_sets = combined.groupby("CV_GROUP_V4", observed=True)["DATABASE"].agg(lambda x: set(map(str, x)))
    shared_groups = {group for group, sources in database_sets.items() if len(sources) > 1}

    train_exclusive = train.loc[~train["CV_GROUP_V4"].isin(shared_groups)].copy()
    test_exclusive = test.loc[~test["CV_GROUP_V4"].isin(shared_groups)].copy()
    cohorts = {
        "GEOROC_frozen_test_MORB": test_exclusive.loc[(test_exclusive["DATABASE"] == "GEOROC") & (test_exclusive["LABEL"] == "MORB")].copy(),
        "PetDB_frozen_test_MORB": test_exclusive.loc[(test_exclusive["DATABASE"] == "PetDB") & (test_exclusive["LABEL"] == "MORB")].copy(),
    }
    non_morb = train_exclusive.loc[train_exclusive["LABEL"] != "MORB"].copy()
    morb_pool = train_exclusive.loc[train_exclusive["LABEL"] == "MORB"].copy()
    georoc_morb = morb_pool.loc[morb_pool["DATABASE"] == "GEOROC"].copy()
    target_samples = len(georoc_morb)
    target_publications = georoc_morb["CV_GROUP_V4"].nunique()
    if (target_samples, target_publications) != (267, 19):
        raise RuntimeError(f"Unexpected GEOROC MORB support: {(target_samples, target_publications)}")

    rows: list[dict[str, object]] = []
    for item in fit_and_score(train_exclusive, cohorts, seed=42):
        rows.append({"training_design": "full_exclusive_support", "repeat": 0, **item})
    georoc_train = pd.concat([non_morb, georoc_morb], ignore_index=True)
    for item in fit_and_score(georoc_train, cohorts, seed=42):
        rows.append({"training_design": "georoc_only_support", "repeat": 0, **item})

    rng = np.random.default_rng(20260926)
    for repeat in range(1, N_REPEATS + 1):
        retained = support_matched_morb(morb_pool, target_publications, target_samples, rng)
        matched_train = pd.concat([non_morb, retained], ignore_index=True)
        for item in fit_and_score(matched_train, cohorts, seed=4200 + repeat):
            rows.append(
                {
                    "training_design": "support_matched_random_MORB",
                    "repeat": repeat,
                    "retained_morb_petdb_fraction": (retained["DATABASE"] == "PetDB").mean(),
                    **item,
                }
            )
        print(f"completed support-matched repeat {repeat}/{N_REPEATS}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "petdb_support_matched_control_runs.csv", index=False)
    summary_rows = []
    for (design, cohort), x in results.groupby(["training_design", "cohort"], observed=True):
        values = x["publication_equal_recall"]
        summary_rows.append(
            {
                "training_design": design,
                "cohort": cohort,
                "n_runs": len(x),
                "publication_equal_recall_mean": values.mean(),
                "publication_equal_recall_median": values.median(),
                "publication_equal_recall_min": values.min(),
                "publication_equal_recall_max": values.max(),
                "publication_equal_recall_q025": values.quantile(0.025),
                "publication_equal_recall_q975": values.quantile(0.975),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "petdb_support_matched_control_summary.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
