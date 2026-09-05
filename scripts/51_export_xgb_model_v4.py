"""Train and export the frozen Stable-50 XGBoost classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "03_model_data" / "Basalt_train_stable50_main_v4.csv"
TEST_FILE = ROOT / "03_model_data" / "Basalt_test_stable50_main_v4.csv"
DEFAULT_OUTPUT_DIR = ROOT / "models"

CLASSES = ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"]
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
PARAMETERS = {
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
EXPECTED_HOLDOUT = {
    "accuracy": 0.8517747372595678,
    "balanced_accuracy": 0.7373851152961828,
    "macro_f1": 0.7386430582045277,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Every class must be present in the training data")
    class_weights = len(labels) / (len(CLASSES) * counts)
    return class_weights[labels]


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing_columns = sorted(set(FEATURES + ["LABEL"]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {path.name}: {missing_columns}")
    measured = data[FEATURES].notna().sum(axis=1)
    if (measured < 10).any():
        raise ValueError(f"{path.name} contains rows below the Stable-50 10/19 rule")
    mapping = {label: index for index, label in enumerate(CLASSES)}
    labels = data["LABEL"].map(mapping)
    if labels.isna().any():
        unknown = sorted(data.loc[labels.isna(), "LABEL"].astype(str).unique())
        raise ValueError(f"Unknown labels in {path.name}: {unknown}")
    return data, labels.to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-holdout-check",
        action="store_true",
        help="Export the model without verifying frozen holdout metrics.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train, y_train = load_dataset(TRAIN_FILE)
    model = XGBClassifier(**PARAMETERS)
    started = time.perf_counter()
    model.fit(train[FEATURES], y_train, sample_weight=sample_weights(y_train))
    fit_seconds = time.perf_counter() - started

    model_file = output_dir / "xgboost_stable50_v4.ubj"
    metadata_file = output_dir / "xgboost_stable50_v4.metadata.json"
    model.save_model(model_file)

    observed_holdout = None
    if not args.skip_holdout_check:
        test, y_test = load_dataset(TEST_FILE)
        predictions = model.predict(test[FEATURES]).astype(int)
        observed_holdout = {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        for metric, expected in EXPECTED_HOLDOUT.items():
            if abs(observed_holdout[metric] - expected) > 1e-9:
                raise RuntimeError(
                    f"Frozen holdout {metric} changed: "
                    f"observed={observed_holdout[metric]:.12f}, expected={expected:.12f}"
                )

    metadata = {
        "model_name": "Frozen Stable-50 XGBoost tectonic-setting classifier",
        "artifact": model_file.name,
        "model_sha256": sha256(model_file),
        "training_data": str(TRAIN_FILE.relative_to(ROOT)).replace("\\", "/"),
        "training_data_sha256": sha256(TRAIN_FILE),
        "training_rows": int(len(train)),
        "class_order": CLASSES,
        "features_in_order": FEATURES,
        "minimum_measured_features": 10,
        "missing_value_representation": "blank CSV cell parsed as NaN",
        "parameters": PARAMETERS,
        "software": {
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "holdout_metrics": observed_holdout,
        "intended_use": "Research evaluation of the eight-class global-basalt workflow.",
        "limitations": [
            "Predictions are valid only for rows satisfying the Stable-50 10-of-19 measurement rule.",
            "The model is not a substitute for geological interpretation or field context.",
            "Performance outside the represented publication and geochemical domains is not established.",
        ],
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {model_file}")
    print(f"Wrote {metadata_file}")
    print(f"Model fitting time on this host: {fit_seconds:.1f} s")
    if observed_holdout:
        print(f"Verified holdout macro-F1: {observed_holdout['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
