from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "04_split" / "Basalt_train_innercv_v4.csv"
OUT_DIR = ROOT / "05_results" / "sensitivity"
CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASSES)}
FEATURES = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FE_TOTAL(WT%)", "CAO(WT%)",
    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    "V(PPM)", "CR(PPM)", "NI(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "BA(PPM)",
]
PARAMS = dict(
    learning_rate=0.02, max_depth=6, min_child_weight=1, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0,
)


def weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    mapping = {i: len(y) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))}
    return np.asarray([mapping[int(v)] for v in y])


def model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=8, n_estimators=1400,
        tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=-1,
        **PARAMS,
    )


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=np.arange(8), average="macro", zero_division=0),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["N_STABLE50"] = data[FEATURES].notna().sum(axis=1)
    data = data.loc[data["N_STABLE50"] >= 10].copy()
    fold_rows = []
    prediction_rows = []
    for fold in sorted(data["INNER_FOLD"].unique()):
        train = data.loc[data["INNER_FOLD"] != fold].copy()
        valid = data.loc[data["INNER_FOLD"] == fold].copy()
        if set(train["CV_GROUP_V4"]) & set(valid["CV_GROUP_V4"]):
            raise RuntimeError("Publication leakage")
        y_train = train["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
        y_valid = valid["LABEL"].map(CLASS_TO_INT).to_numpy(dtype=int)
        for method in ("native_missing", "fold_median_imputed"):
            if method == "native_missing":
                x_train = train[FEATURES]
                x_valid = valid[FEATURES]
            else:
                imputer = SimpleImputer(strategy="median")
                x_train = imputer.fit_transform(train[FEATURES])
                x_valid = imputer.transform(valid[FEATURES])
            fitted = model()
            fitted.fit(x_train, y_train, sample_weight=weights(y_train))
            pred = fitted.predict(x_valid).astype(int)
            fold_rows.append({"fold": int(fold), "method": method, "n_train": len(train), "n_valid": len(valid), **score(y_valid, pred)})
            part = valid[["UNIQUE_ID", "CV_GROUP_V4", "LABEL"]].copy()
            part["fold"] = int(fold)
            part["method"] = method
            part["TRUE_INT"] = y_valid
            part["PRED_INT"] = pred
            prediction_rows.append(part)
        print(f"fold {fold} complete", flush=True)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = folds.groupby("method", observed=True).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), sd_balanced_accuracy=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"),
    ).reset_index()
    paired = folds.pivot(index="fold", columns="method", values=["accuracy", "balanced_accuracy", "macro_f1"])
    difference_rows = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        diff = paired[(metric, "native_missing")] - paired[(metric, "fold_median_imputed")]
        difference_rows.append({
            "metric": metric,
            "mean_native_minus_imputed": diff.mean(),
            "min_fold_difference": diff.min(),
            "max_fold_difference": diff.max(),
        })
    folds.to_csv(OUT_DIR / "stable50_native_vs_fold_median_folds_v9.csv", index=False)
    summary.to_csv(OUT_DIR / "stable50_native_vs_fold_median_summary_v9.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(OUT_DIR / "stable50_native_vs_fold_median_differences_v9.csv", index=False)
    predictions.to_csv(OUT_DIR / "stable50_native_vs_fold_median_predictions_v9.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(difference_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
