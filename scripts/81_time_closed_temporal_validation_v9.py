from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "02_merged" / "Basalt_8classes_paperid_v4.csv"
OUT_DIR = ROOT / "05_results" / "temporal_v9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASSES)}
CANDIDATE_FEATURES = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FE_TOTAL(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "SC(PPM)", "V(PPM)", "CR(PPM)", "CO(PPM)", "NI(PPM)", "RB(PPM)",
    "SR(PPM)", "Y(PPM)", "ZR(PPM)", "NB(PPM)", "BA(PPM)", "LA(PPM)",
    "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)", "GD(PPM)",
    "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "TM(PPM)", "YB(PPM)",
    "LU(PPM)", "HF(PPM)", "TA(PPM)", "PB(PPM)", "TH(PPM)", "U(PPM)",
]


def select_features(frame: pd.DataFrame) -> list[str]:
    rates = frame[CANDIDATE_FEATURES].isna().mean()
    return [feature for feature in CANDIDATE_FEATURES if rates[feature] <= 0.50]


def apply_completeness(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, int]:
    threshold = math.ceil(len(features) / 2)
    work = frame.copy()
    work["N_TEMPORAL_FEATURES"] = work[features].notna().sum(axis=1)
    return work.loc[work["N_TEMPORAL_FEATURES"] >= threshold].copy(), threshold


def sample_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    missing = sorted(set(range(len(CLASSES))) - set(counts.index))
    if missing:
        raise RuntimeError(f"Training fold lacks classes: {[CLASSES[i] for i in missing]}")
    weights = {i: len(y) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))}
    return np.asarray([weights[int(value)] for value in y], dtype=float)


def make_model(params: dict[str, float | int], n_estimators: int, early_stopping: bool) -> XGBClassifier:
    kwargs = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=n_estimators,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    if early_stopping:
        kwargs["early_stopping_rounds"] = 75
    return XGBClassifier(**kwargs)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average="weighted", zero_division=0)),
    }


def tune(early: pd.DataFrame) -> tuple[dict[str, float | int], int, pd.DataFrame, pd.DataFrame]:
    grid = {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }
    parameter_sets = list(ParameterSampler(grid, n_iter=12, random_state=42))
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=180)
    splits = list(splitter.split(early, early["LABEL"], early["CV_GROUP_V4"]))
    config_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for config_id, params in enumerate(parameter_sets, start=1):
        current: list[dict[str, object]] = []
        for fold, (train_index, valid_index) in enumerate(splits):
            train_raw = early.iloc[train_index].copy()
            valid_raw = early.iloc[valid_index].copy()
            if set(train_raw["CV_GROUP_V4"]) & set(valid_raw["CV_GROUP_V4"]):
                raise RuntimeError("Publication leakage in temporal-development cross-validation")
            features = select_features(train_raw)
            train, threshold = apply_completeness(train_raw, features)
            valid, _ = apply_completeness(valid_raw, features)
            y_train = train["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
            y_valid = valid["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
            model = make_model(params, n_estimators=1800, early_stopping=True)
            model.fit(
                train[features], y_train,
                sample_weight=sample_weights(y_train),
                eval_set=[(valid[features], y_valid)],
                verbose=False,
            )
            pred = model.predict(valid[features]).astype(int)
            row = {
                "config_id": config_id,
                "fold": fold,
                "n_train": len(train),
                "n_valid": len(valid),
                "n_train_publications": train["CV_GROUP_V4"].nunique(),
                "n_valid_publications": valid["CV_GROUP_V4"].nunique(),
                "n_features": len(features),
                "completeness_threshold": threshold,
                "selected_features": "|".join(features),
                "best_iteration": int(model.best_iteration),
                **metrics(y_valid, pred),
            }
            current.append(row)
            fold_rows.append({**params, **row})
        current_df = pd.DataFrame(current)
        config_rows.append({
            "config_id": config_id,
            **params,
            "mean_macro_f1": current_df["macro_f1"].mean(),
            "sd_macro_f1": current_df["macro_f1"].std(ddof=1),
            "mean_balanced_accuracy": current_df["balanced_accuracy"].mean(),
            "mean_accuracy": current_df["accuracy"].mean(),
            "median_best_iteration": int(np.median(current_df["best_iteration"])) + 1,
        })
        pd.DataFrame(config_rows).to_csv(OUT_DIR / "time_closed_tuning_summary_checkpoint_v9.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT_DIR / "time_closed_tuning_folds_checkpoint_v9.csv", index=False)
        print(f"config {config_id:02d}/12 macro-F1={config_rows[-1]['mean_macro_f1']:.4f}", flush=True)
    configs = pd.DataFrame(config_rows).sort_values(
        ["mean_macro_f1", "mean_balanced_accuracy", "sd_macro_f1"],
        ascending=[False, False, True],
    )
    folds = pd.DataFrame(fold_rows)
    winner = configs.iloc[0]
    param_names = list(grid)
    best_params = {name: winner[name].item() if hasattr(winner[name], "item") else winner[name] for name in param_names}
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    for name in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        best_params[name] = float(best_params[name])
    return best_params, int(winner["median_best_iteration"]), configs, folds


def cluster_bootstrap(predictions: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    group_indices = {
        group: values.index.to_numpy()
        for group, values in predictions.groupby("CV_GROUP_V4", observed=True)
    }
    groups = np.asarray(list(group_indices), dtype=object)
    rng = np.random.default_rng(20260922)
    rows = []
    for _ in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        subset = predictions.loc[indices]
        rows.append(metrics(subset["TRUE_INT"].to_numpy(), subset["PRED_INT"].to_numpy()))
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["PAPER_YEAR_NUM"] = pd.to_numeric(data["PAPER_YEAR"], errors="coerce")
    dated = data.loc[data["PAPER_YEAR_NUM"].notna()].copy()
    early = dated.loc[dated["PAPER_YEAR_NUM"] <= 2010].copy()
    late_raw = dated.loc[dated["PAPER_YEAR_NUM"] >= 2011].copy()
    if set(early["CV_GROUP_V4"]) & set(late_raw["CV_GROUP_V4"]):
        raise RuntimeError("Publication overlap across temporal boundary")
    if set(early["LABEL"]) != set(CLASSES) or set(late_raw["LABEL"]) != set(CLASSES):
        raise RuntimeError("All classes must be represented in both periods")

    best_params, n_estimators, configs, folds = tune(early)
    final_features = select_features(early)
    train, completeness_threshold = apply_completeness(early, final_features)
    test, _ = apply_completeness(late_raw, final_features)
    y_train = train["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
    y_test = test["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
    model = make_model(best_params, n_estimators=n_estimators, early_stopping=False)
    model.fit(train[final_features], y_train, sample_weight=sample_weights(y_train))
    probabilities = model.predict_proba(test[final_features])
    pred = np.argmax(probabilities, axis=1).astype(int)

    columns = [
        "UNIQUE_ID", "GROUP_ID", "CV_GROUP_V4", "DATABASE", "source_file", "LOCATION",
        "PAPER_YEAR_NUM", "PAPER_DOI", "CITATION", "SAMPLE NAME", "LABEL", "N_TEMPORAL_FEATURES",
    ]
    predictions = test[columns].copy().rename(columns={"LABEL": "TRUE_LABEL", "PAPER_YEAR_NUM": "PAPER_YEAR"})
    predictions["TRUE_INT"] = y_test
    predictions["PRED_INT"] = pred
    predictions["PRED_LABEL"] = [CLASSES[value] for value in pred]
    predictions["CORRECT"] = predictions["TRUE_INT"] == predictions["PRED_INT"]
    predictions["MAX_PROB"] = probabilities.max(axis=1)
    for i, label in enumerate(CLASSES):
        predictions[f"PROB_{label}"] = probabilities[:, i]
    predictions = predictions.reset_index(drop=True)

    point = metrics(y_test, pred)
    bootstrap = cluster_bootstrap(predictions)
    overall = pd.DataFrame([
        {
            "metric": metric,
            "estimate": estimate,
            "cluster_bootstrap_ci_low": bootstrap[metric].quantile(0.025),
            "cluster_bootstrap_ci_high": bootstrap[metric].quantile(0.975),
        }
        for metric, estimate in point.items()
    ])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=np.arange(len(CLASSES)), zero_division=0
    )
    class_rows = []
    for i, label in enumerate(CLASSES):
        subset = predictions.loc[predictions["TRUE_LABEL"] == label]
        publication_recall = subset.groupby("CV_GROUP_V4", observed=True)["CORRECT"].mean()
        rng = np.random.default_rng(20260922 + i)
        draws = rng.choice(
            publication_recall.to_numpy(dtype=float),
            size=(5000, len(publication_recall)),
            replace=True,
        ).mean(axis=1)
        class_rows.append({
            "class": label,
            "n_samples": int(support[i]),
            "n_publications": int(subset["CV_GROUP_V4"].nunique()),
            "precision": float(precision[i]),
            "sample_recall": float(recall[i]),
            "f1": float(f1[i]),
            "publication_equal_recall": float(publication_recall.mean()),
            "publication_equal_recall_ci_low": float(np.quantile(draws, 0.025)),
            "publication_equal_recall_ci_high": float(np.quantile(draws, 0.975)),
        })
    class_table = pd.DataFrame(class_rows)

    configs.to_csv(OUT_DIR / "time_closed_tuning_summary_v9.csv", index=False)
    folds.to_csv(OUT_DIR / "time_closed_tuning_folds_v9.csv", index=False)
    predictions.to_csv(OUT_DIR / "time_closed_temporal_predictions_v9.csv", index=False)
    overall.to_csv(OUT_DIR / "time_closed_temporal_overall_v9.csv", index=False)
    class_table.to_csv(OUT_DIR / "time_closed_temporal_classes_v9.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "time_closed_temporal_bootstrap_v9.csv", index=False)
    pd.DataFrame({"feature": final_features, "training_period_missing_rate": early[final_features].isna().mean().values}).to_csv(
        OUT_DIR / "time_closed_selected_features_v9.csv", index=False
    )
    manifest = {
        "analysis": "strict time-closed temporal validation",
        "cutoff": "train publication year <= 2010; test publication year 2011-2025",
        "candidate_feature_universe": CANDIDATE_FEATURES,
        "feature_rule": "missingness <= 0.50 using only the relevant training data",
        "selected_features": final_features,
        "completeness_rule": "at least ceil(n_selected_features / 2) measured predictors",
        "completeness_threshold": completeness_threshold,
        "tuning": "12 random configurations; fourfold stratified publication-group CV within <=2010 data only",
        "best_params": best_params,
        "n_estimators": n_estimators,
        "train_samples_before_completeness": int(len(early)),
        "train_samples": int(len(train)),
        "train_publications": int(train["CV_GROUP_V4"].nunique()),
        "test_samples_before_completeness": int(len(late_raw)),
        "test_samples": int(len(test)),
        "test_publications": int(test["CV_GROUP_V4"].nunique()),
        "excluded_missing_year": int(data["PAPER_YEAR_NUM"].isna().sum()),
        "metrics": point,
        "software": {"pandas": pd.__version__},
    }
    (OUT_DIR / "time_closed_temporal_manifest_v9.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
