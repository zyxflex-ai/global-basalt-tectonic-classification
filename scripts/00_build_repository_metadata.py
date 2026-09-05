"""Generate source-publication attribution and column metadata files."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "02_merged" / "Basalt_8classes_paperid_v4.csv"
SOURCE_OUTPUT = ROOT / "references" / "source_publications.csv"
DICTIONARY_OUTPUT = ROOT / "docs" / "data_dictionary.csv"


def describe_column(name: str) -> tuple[str, str]:
    if name.endswith("(WT%)"):
        return "Major-element concentration", "weight percent"
    if name.endswith("(PPM)"):
        return "Trace-element concentration", "parts per million"
    descriptions = {
        "Year": ("Publication or record year", "year"),
        "CITATION": ("Original source-publication citation", "text"),
        "SAMPLE NAME": ("Source sample name", "text"),
        "UNIQUE_ID": ("Stable record identifier", "text"),
        "GROUP_ID": ("Sample-group identifier", "text"),
        "LOCATION": ("Reported locality", "text"),
        "LATITUDE (MAX.)": ("Reported maximum latitude", "decimal degrees"),
        "LONGITUDE (MAX.)": ("Reported maximum longitude", "decimal degrees"),
        "TECTONIC SETTING": ("Source tectonic-setting description", "text"),
        "GEOLOGICAL AGE": ("Source geological-age description", "text"),
        "TYPE OF MATERIAL": ("Analysed material type", "text"),
        "source_file": ("Local provenance label for the source extract", "text"),
        "LABEL": ("Harmonized tectonic class", "categorical"),
        "IOAB_REGION": ("Regional audit field for intra-oceanic arcs", "text"),
        "BABB_REGION": ("Regional audit field for back-arc basins", "text"),
        "DATABASE": ("Source synthesis database", "GEOROC or PetDB"),
        "PAPER_YEAR": ("Normalized source-publication year", "year"),
        "PAPER_FIRST_AUTHOR": ("Normalized first-author name", "text"),
        "PAPER_MATCH_KEY": ("Normalized cross-database matching key", "text"),
        "PAPER_DOI": ("Normalized source-publication DOI", "DOI"),
        "PAPER_RECORD_ID": ("Database-specific publication record identifier", "text"),
        "OLD_SET_V3": ("Previous frozen-split membership used for audit", "categorical"),
        "PAPER_ID": ("Verified source-publication identity", "text"),
        "PAPER_ID_STATUS": ("Publication-identity verification status", "categorical"),
        "CV_GROUP_V4": ("Publication-aware leakage-control group", "text"),
        "CV_GROUP_V4_BASIS": ("Evidence used to construct the group", "categorical"),
        "INNER_FOLD": ("Frozen grouped development fold", "integer 0-4"),
        "N_STABLE50": ("Measured values among 19 Stable-50 predictors", "integer 0-19"),
    }
    return descriptions.get(name, ("Source metadata field", "text"))


def main() -> None:
    data = pd.read_csv(MERGED, encoding="utf-8-sig", low_memory=False)
    publication_columns = [
        "DATABASE",
        "PAPER_ID",
        "CV_GROUP_V4",
        "PAPER_DOI",
        "CITATION",
        "PAPER_FIRST_AUTHOR",
        "PAPER_YEAR",
    ]
    source_rows = data[publication_columns].fillna("")

    def join_unique(values: pd.Series) -> str:
        return "; ".join(sorted({str(value).strip() for value in values if str(value).strip()}))

    publications = (
        source_rows.groupby("PAPER_ID", dropna=False, as_index=False)
        .agg(
            DATABASE=("DATABASE", join_unique),
            CV_GROUP_V4=("CV_GROUP_V4", join_unique),
            PAPER_DOI=("PAPER_DOI", join_unique),
            CITATION=("CITATION", join_unique),
            PAPER_FIRST_AUTHOR=("PAPER_FIRST_AUTHOR", join_unique),
            PAPER_YEAR=("PAPER_YEAR", join_unique),
            N_SAMPLES=("PAPER_ID", "size"),
        )
        .sort_values(["PAPER_YEAR", "PAPER_FIRST_AUTHOR", "PAPER_ID"])
    )
    publications.to_csv(SOURCE_OUTPUT, index=False, encoding="utf-8-sig")

    with DICTIONARY_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "description", "unit_or_coding"])
        for name in data.columns:
            description, unit = describe_column(name)
            writer.writerow([name, description, unit])

    print(f"Wrote {SOURCE_OUTPUT} ({len(publications):,} publication records)")
    print(f"Wrote {DICTIONARY_OUTPUT} ({len(data.columns):,} columns)")


if __name__ == "__main__":
    main()
