from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "02_merged" / "Basalt_8classes_paperid_v4.csv"
OUT_DIR = ROOT / "05_results" / "audit_v9"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalized(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "[blank]"
    return " ".join(str(value).strip().split())


def top_location(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "[unknown]"
    text = str(value).strip()
    for separator in ("/", ";", ",", "|"):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    return " ".join(text.upper().split())


def count_matrix(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return data.groupby(group_columns, dropna=False, observed=True).agg(
        n_samples=("UNIQUE_ID", "size"),
        n_publications=("CV_GROUP_V4", "nunique"),
    ).reset_index()


def main() -> None:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["PAPER_YEAR_NUM"] = pd.to_numeric(data["PAPER_YEAR"], errors="coerce")
    data["TIME_PERIOD"] = pd.cut(
        data["PAPER_YEAR_NUM"], bins=[-float("inf"), 2010, 2025],
        labels=["through_2010", "2011_2025"],
    ).astype("object").fillna("unknown_year")
    data["RAW_TECTONIC_SETTING"] = data["TECTONIC SETTING"].map(normalized)
    data["IOAB_REGION_AUDIT"] = data["IOAB_REGION"].map(normalized)
    data["BABB_REGION_AUDIT"] = data["BABB_REGION"].map(normalized)
    data["COARSE_LOCATION_TOKEN"] = data["LOCATION"].map(top_location)

    count_matrix(data, ["DATABASE", "source_file", "LABEL"]).to_csv(
        OUT_DIR / "source_file_to_final_label_counts_v9.csv", index=False
    )
    count_matrix(data, ["RAW_TECTONIC_SETTING", "LABEL"]).to_csv(
        OUT_DIR / "raw_tectonic_setting_to_final_label_counts_v9.csv", index=False
    )
    count_matrix(data, ["LABEL", "IOAB_REGION_AUDIT", "BABB_REGION_AUDIT"]).to_csv(
        OUT_DIR / "regional_label_fields_counts_v9.csv", index=False
    )
    count_matrix(data, ["DATABASE", "LABEL", "TIME_PERIOD"]).to_csv(
        OUT_DIR / "database_class_time_support_v9.csv", index=False
    )
    count_matrix(data, ["DATABASE", "LABEL", "TIME_PERIOD", "COARSE_LOCATION_TOKEN"]).to_csv(
        OUT_DIR / "database_class_time_location_support_v9.csv", index=False
    )

    publication_labels = data.groupby("CV_GROUP_V4", observed=True).agg(
        n_samples=("UNIQUE_ID", "size"),
        n_labels=("LABEL", "nunique"),
        labels=("LABEL", lambda values: "|".join(sorted(set(map(str, values))))),
        databases=("DATABASE", lambda values: "|".join(sorted(set(map(str, values))))),
        years=("PAPER_YEAR_NUM", lambda values: "|".join(map(str, sorted(set(values.dropna().astype(int)))))),
    ).reset_index()
    publication_labels.to_csv(OUT_DIR / "publication_group_label_audit_v9.csv", index=False)

    overview = pd.DataFrame([
        {"item": "samples", "value": len(data)},
        {"item": "publication_groups", "value": data["CV_GROUP_V4"].nunique()},
        {"item": "multi_label_publication_groups", "value": int((publication_labels["n_labels"] > 1).sum())},
        {"item": "blank_original_tectonic_setting_samples", "value": int((data["RAW_TECTONIC_SETTING"] == "[blank]").sum())},
        {"item": "dated_samples", "value": int(data["PAPER_YEAR_NUM"].notna().sum())},
        {"item": "undated_samples", "value": int(data["PAPER_YEAR_NUM"].isna().sum())},
    ])
    overview.to_csv(OUT_DIR / "label_traceability_overview_v9.csv", index=False)
    print(overview.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
