"""Compare sample-random and publication-grouped cross-validation.

The comparison holds the dataset, Stable-50 features, XGBoost hyperparameters,
class weighting, and number of folds constant. Only the fold-assignment unit is
changed. Outputs are written for manuscript reporting and figure generation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from _project_paths import FINAL_DIR


CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASSES)}

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

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(CLASSES),
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
    "verbosity": 0,
}


def sample_class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    if np.any(counts == 0):
        missing = [CLASSES[i] for i, count in enumerate(counts) if count == 0]
        raise RuntimeError(f"Training fold lacks classes: {missing}")
    class_weights = len(y) / (len(CLASSES) * counts)
    return class_weights[y]


def group_equal_weights(groups: np.ndarray) -> np.ndarray:
    counts = pd.Series(groups).value_counts()
    return np.asarray([1.0 / counts[group] for group in groups], dtype=float)


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=np.arange(len(CLASSES)),
            average="macro",
            sample_weight=sample_weight,
            zero_division=0,
        ),
    }


def split_audit(
    groups: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> dict[str, float | int]:
    train_groups = set(groups[train_idx])
    valid_groups = groups[valid_idx]
    overlap_groups = train_groups.intersection(valid_groups)
    leaked_sample_mask = np.fromiter(
        (group in train_groups for group in valid_groups),
        dtype=bool,
        count=len(valid_groups),
    )
    return {
        "n_train_groups": len(train_groups),
        "n_valid_groups": len(set(valid_groups)),
        "n_overlapping_groups": len(overlap_groups),
        "validation_sample_leakage_fraction": float(leaked_sample_mask.mean()),
    }


def fit_oof(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    scheme: str,
    repeat: int,
    seed: int,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    X = df[FEATURES]
    y = df["Y"].to_numpy(dtype=int)
    groups = df["CV_GROUP_V4"].astype(str).to_numpy()
    predictions = np.full(len(df), -1, dtype=int)
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splits):
        start = time.time()
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            sample_weight=sample_class_weights(y[train_idx]),
        )
        fold_pred = model.predict(X.iloc[valid_idx]).astype(int)
        predictions[valid_idx] = fold_pred

        fold_metric = metric_row(y[valid_idx], fold_pred)
        fold_rows.append(
            {
                "scheme": scheme,
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                "level": "fold",
                "weighting": "sample",
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "seconds": time.time() - start,
                **fold_metric,
            }
        )
        audit_rows.append(
            {
                "scheme": scheme,
                "repeat": repeat,
                "seed": seed,
                "fold": fold,
                **split_audit(groups, train_idx, valid_idx),
            }
        )
        print(
            f"{scheme} repeat={repeat:02d} fold={fold} "
            f"Macro-F1={fold_metric['macro_f1']:.4f} "
            f"elapsed={fold_rows[-1]['seconds']:.1f}s",
            flush=True,
        )

    if np.any(predictions < 0):
        raise RuntimeError(f"Incomplete OOF predictions for {scheme}, repeat {repeat}")
    return predictions, fold_rows, audit_rows


def aggregate_rows(
    df: pd.DataFrame,
    predictions: np.ndarray,
    scheme: str,
    repeat: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    y = df["Y"].to_numpy(dtype=int)
    groups = df["CV_GROUP_V4"].astype(str).to_numpy()
    rows: list[dict] = []
    for weighting, weights in (
        ("sample", None),
        ("publication_equal", group_equal_weights(groups)),
    ):
        rows.append(
            {
                "scheme": scheme,
                "repeat": repeat,
                "seed": seed,
                "fold": -1,
                "level": "oof",
                "weighting": weighting,
                "n_train": np.nan,
                "n_valid": len(df),
                "seconds": np.nan,
                **metric_row(y, predictions, sample_weight=weights),
            }
        )

    class_f1 = f1_score(
        y,
        predictions,
        labels=np.arange(len(CLASSES)),
        average=None,
        zero_division=0,
    )
    class_rows = [
        {
            "scheme": scheme,
            "repeat": repeat,
            "seed": seed,
            "class": class_name,
            "f1": float(class_f1[class_idx]),
            "support": int(np.sum(y == class_idx)),
        }
        for class_idx, class_name in enumerate(CLASSES)
    ]
    return rows, class_rows


def primary_splits(
    df: pd.DataFrame,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    y = df["Y"].to_numpy(dtype=int)
    grouped = []
    inner_fold = df["INNER_FOLD"].to_numpy(dtype=int)
    for fold in range(5):
        grouped.append((np.flatnonzero(inner_fold != fold), np.flatnonzero(inner_fold == fold)))
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random = list(random_cv.split(np.zeros(len(df)), y))
    return {"publication_grouped": grouped, "sample_random": random}


def repeated_splits(
    df: pd.DataFrame, seed: int
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    y = df["Y"].to_numpy(dtype=int)
    groups = df["CV_GROUP_V4"].astype(str).to_numpy()
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grouped_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return {
        "publication_grouped": list(grouped_cv.split(np.zeros(len(df)), y, groups)),
        "sample_random": list(random_cv.split(np.zeros(len(df)), y)),
    }


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    grouped_pred: np.ndarray,
    random_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    y = df["Y"].to_numpy(dtype=int)
    groups = df["CV_GROUP_V4"].astype(str).to_numpy()
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        multiplicities = np.bincount(sampled, minlength=len(unique_groups)).astype(float)
        weights = multiplicities[group_inverse]
        grouped_metrics = metric_row(y, grouped_pred, sample_weight=weights)
        random_metrics = metric_row(y, random_pred, sample_weight=weights)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "iteration": iteration,
                    "metric": metric,
                    "sample_random": random_metrics[metric],
                    "publication_grouped": grouped_metrics[metric],
                    "difference_random_minus_grouped": (
                        random_metrics[metric] - grouped_metrics[metric]
                    ),
                }
            )
        if (iteration + 1) % 500 == 0:
            print(f"Cluster bootstrap {iteration + 1}/{iterations}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    input_file = FINAL_DIR / "03_model_data" / "Basalt_train_stable50_main_v4.csv"
    output_dir = FINAL_DIR / "05_results" / "validation_bias"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
    missing = [name for name in FEATURES if name not in df.columns]
    if missing:
        raise RuntimeError(f"Missing Stable-50 features: {missing}")
    if df[FEATURES].notna().sum(axis=1).min() < 10:
        raise RuntimeError("Stable-50 input includes a sample below the frozen 10/19 rule")
    df["Y"] = df["LABEL"].map(CLASS_TO_INT)
    if df["Y"].isna().any():
        raise RuntimeError("Unknown class labels found")
    df["Y"] = df["Y"].astype(int)
    if df["CV_GROUP_V4"].isna().any():
        raise RuntimeError("Missing publication group identifiers found")

    print(
        f"Input n={len(df)}, publication groups={df['CV_GROUP_V4'].nunique()}, "
        f"features={len(FEATURES)}",
        flush=True,
    )

    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    audit_rows: list[dict] = []
    primary_predictions: dict[str, np.ndarray] = {}

    for scheme, splits in primary_splits(df).items():
        pred, fold_rows, audit = fit_oof(df, splits, scheme, -1, 42)
        agg_rows, per_class = aggregate_rows(df, pred, scheme, -1, 42)
        metric_rows.extend(fold_rows + agg_rows)
        class_rows.extend(per_class)
        audit_rows.extend(audit)
        primary_predictions[scheme] = pred

    for repeat in range(args.repeats):
        seed = 1000 + repeat
        for scheme, splits in repeated_splits(df, seed).items():
            pred, fold_rows, audit = fit_oof(df, splits, scheme, repeat, seed)
            agg_rows, per_class = aggregate_rows(df, pred, scheme, repeat, seed)
            metric_rows.extend(fold_rows + agg_rows)
            class_rows.extend(per_class)
            audit_rows.extend(audit)

    bootstrap = paired_cluster_bootstrap(
        df,
        primary_predictions["publication_grouped"],
        primary_predictions["sample_random"],
        args.bootstrap,
        seed=20260904,
    )

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "random_vs_grouped_cv_metrics_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(class_rows).to_csv(
        output_dir / "random_vs_grouped_cv_class_f1_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "random_vs_grouped_cv_split_audit_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        output_dir / "random_vs_grouped_cv_cluster_bootstrap_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_output = df[
        ["GROUP_ID", "CV_GROUP_V4", "CITATION", "SAMPLE NAME", "LABEL", "INNER_FOLD"]
    ].copy()
    for scheme, pred in primary_predictions.items():
        prediction_output[f"PRED_{scheme.upper()}"] = [CLASSES[value] for value in pred]
        prediction_output[f"CORRECT_{scheme.upper()}"] = (
            prediction_output["LABEL"] == prediction_output[f"PRED_{scheme.upper()}"]
        )
    prediction_output.to_csv(
        output_dir / "random_vs_grouped_cv_primary_predictions_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "input_file": str(input_file),
        "n_samples": len(df),
        "n_publication_groups": int(df["CV_GROUP_V4"].nunique()),
        "n_features": len(FEATURES),
        "n_folds": 5,
        "n_repeats": args.repeats,
        "bootstrap_iterations": args.bootstrap,
        "xgb_parameters": XGB_PARAMS,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    (output_dir / "random_vs_grouped_cv_run_manifest_v4.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
