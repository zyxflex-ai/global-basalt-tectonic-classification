"""Apply the frozen Stable-50 XGBoost model to a compatible CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "xgboost_stable50_v4.ubj"
DEFAULT_METADATA = ROOT / "models" / "xgboost_stable50_v4.metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV containing the 19 features.")
    parser.add_argument("--output", required=True, type=Path, help="Destination prediction CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of leading rows for a quick test.")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    features = metadata["features_in_order"]
    classes = metadata["class_order"]
    minimum = int(metadata["minimum_measured_features"])

    data = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        data = data.head(args.limit).copy()
    missing_columns = sorted(set(features) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required feature columns: {missing_columns}")

    matrix = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    measured = matrix.notna().sum(axis=1)
    invalid_count = int((measured < minimum).sum())
    if invalid_count:
        raise ValueError(
            f"{invalid_count} rows violate the Stable-50 rule of at least {minimum} measured features"
        )

    model = XGBClassifier()
    model.load_model(args.model)
    probabilities = model.predict_proba(matrix)
    predicted = probabilities.argmax(axis=1)

    identifier_columns = [
        name
        for name in ["UNIQUE_ID", "GROUP_ID", "CV_GROUP_V4", "SAMPLE NAME", "LABEL"]
        if name in data.columns
    ]
    output = data[identifier_columns].copy()
    output["N_MEASURED_STABLE50"] = measured.to_numpy()
    output["PREDICTED_LABEL"] = [classes[index] for index in predicted]
    output["MAX_PROBABILITY"] = probabilities.max(axis=1)
    for index, label in enumerate(classes):
        output[f"PROB_{label}"] = probabilities[:, index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Predicted {len(output):,} rows")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
