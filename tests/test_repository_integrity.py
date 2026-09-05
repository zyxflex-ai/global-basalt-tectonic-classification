"""Fast, non-destructive integrity checks for the frozen manuscript release."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    merged = pd.read_csv(
        ROOT / "02_merged" / "Basalt_8classes_paperid_v4.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    train = pd.read_csv(
        ROOT / "03_model_data" / "Basalt_train_stable50_main_v4.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    test = pd.read_csv(
        ROOT / "03_model_data" / "Basalt_test_stable50_main_v4.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    inner = pd.read_csv(
        ROOT / "04_split" / "Basalt_train_innercv_v4.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )

    require(len(merged) == 48_403, "Unexpected merged sample count")
    require(merged["CV_GROUP_V4"].nunique() == 2_430, "Unexpected publication-group count")
    require(len(train) == 35_865, "Unexpected Stable-50 development count")
    require(len(test) == 10_086, "Unexpected Stable-50 holdout count")
    require(train["CV_GROUP_V4"].nunique() == 1_846, "Unexpected development-group count")
    require(test["CV_GROUP_V4"].nunique() == 462, "Unexpected holdout-group count")
    require(set(train["PAPER_ID"]).isdisjoint(test["PAPER_ID"]), "PAPER_ID leakage detected")
    require(set(train["CV_GROUP_V4"]).isdisjoint(test["CV_GROUP_V4"]), "CV_GROUP leakage detected")
    require(inner["INNER_FOLD"].between(0, 4).all(), "Invalid inner-fold assignment")
    require(inner.groupby("CV_GROUP_V4")["INNER_FOLD"].nunique().max() == 1, "A group crosses inner folds")

    publications = pd.read_csv(ROOT / "references" / "source_publications.csv")
    require(len(publications) == 2_430, "Attribution table must contain one row per publication group")
    require(publications["PAPER_ID"].is_unique, "Duplicate PAPER_ID in attribution table")
    require(
        publications["CV_GROUP_V4"].str.contains(";").sum() == 0,
        "A publication identity maps to multiple CV groups",
    )

    petdb_records = pd.read_csv(ROOT / "references" / "petdb_source_records.csv")
    require(len(petdb_records) == 303, "Unexpected PetDB publication-record count")
    require(petdb_records["PAPER_RECORD_ID"].is_unique, "Duplicate PetDB publication record")
    require(petdb_records["N_RETAINED_SAMPLES"].sum() == 4_734, "Unexpected retained PetDB count")
    require(
        petdb_records["N_RETAINED_SAMPLES"].eq(petdb_records["N_UNIQUE_SAMPLE_URLS"]).all(),
        "A retained PetDB sample URL is duplicated",
    )
    require(
        petdb_records["PETDB_CITATION_URL"].str.startswith(
            "https://www.earthchem.org/petdb/citation/"
        ).all(),
        "Invalid PetDB citation URL",
    )

    expected_classes = {
        "CAB": 865,
        "IAB": 2706,
        "IOAB": 546,
        "BABB": 1225,
        "MORB": 5316,
        "OIB": 18311,
        "OPB": 1860,
        "CFB": 17574,
    }
    require(merged["LABEL"].value_counts().to_dict() == expected_classes, "Class counts changed")

    metrics = pd.read_csv(
        ROOT / "05_results" / "final_test" / "final_independent_test_metrics_v4.csv"
    )
    xgb = metrics[(metrics["model"] == "XGBoost") & (metrics["level"] == "overall")].iloc[0]
    require(abs(xgb["macro_f1"] - 0.7386430582) < 1e-9, "XGBoost macro-F1 changed")
    require(abs(xgb["balanced_accuracy"] - 0.7373851153) < 1e-9, "Balanced accuracy changed")

    model_file = ROOT / "models" / "xgboost_stable50_v4.ubj"
    metadata_file = ROOT / "models" / "xgboost_stable50_v4.metadata.json"
    require(model_file.is_file(), "Frozen XGBoost model is missing")
    require(metadata_file.is_file(), "Frozen model metadata are missing")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    require(metadata["artifact"] == model_file.name, "Model metadata artifact name changed")
    require(len(metadata["features_in_order"]) == 19, "Frozen model feature count changed")
    require(metadata["class_order"] == ["CAB", "IAB", "IOAB", "BABB", "MORB", "OIB", "OPB", "CFB"], "Class order changed")

    predictions = pd.read_csv(
        ROOT / "05_results" / "final_test" / "final_independent_test_predictions_v4.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    require(len(predictions) == 10_086, "Holdout prediction row count changed")

    with np.load(
        ROOT / "05_results" / "figures" / "shap_direction" / "stable50_independent_test_shap_values_v4.npz",
        allow_pickle=False,
    ) as shap_cache:
        require(shap_cache["shap_values"].shape == (10_086, 19, 8), "SHAP cache shape changed")
        require(shap_cache["feature_values"].shape == (10_086, 19), "SHAP feature-value shape changed")

    required_guides = [
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "USER_GUIDE.md",
        ROOT / "docs" / "SUBMISSION_READINESS.md",
    ]
    require(all(path.is_file() for path in required_guides), "A required repository guide is missing")

    print("Repository integrity checks passed.")
    print("48,403 samples; 2,430 publication groups; no train-holdout group leakage.")


if __name__ == "__main__":
    main()
