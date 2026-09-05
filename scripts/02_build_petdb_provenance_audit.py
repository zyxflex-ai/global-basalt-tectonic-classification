"""Verify retained PetDB rows against the original portal CSV export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "02_merged" / "Basalt_8classes_paperid_v4.csv"
OUTPUT = ROOT / "references" / "petdb_source_records.csv"


def join_unique(values: pd.Series) -> str:
    return "; ".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-petdb", required=True, type=Path, help="Original PetDB portal CSV export.")
    parser.add_argument("--download-date", required=True, help="Portal download date in YYYY-MM-DD format.")
    args = parser.parse_args()

    merged = pd.read_csv(MERGED, encoding="utf-8-sig", low_memory=False)
    raw = pd.read_csv(args.raw_petdb, encoding="utf-8-sig", low_memory=False)
    required_raw = {"Sample URL", "Citation", "Citation URL"}
    missing_raw = sorted(required_raw - set(raw.columns))
    if missing_raw:
        raise ValueError(f"Raw PetDB export is missing columns: {missing_raw}")

    retained = merged.loc[
        merged["DATABASE"].eq("PetDB"),
        ["UNIQUE_ID", "PAPER_RECORD_ID", "PAPER_ID", "CV_GROUP_V4", "CITATION"],
    ].copy()
    if retained.empty:
        raise RuntimeError("No PetDB rows were found in the merged release")

    lookup = raw[["Sample URL", "Citation", "Citation URL"]].drop_duplicates()
    matched = retained.merge(
        lookup,
        left_on=["UNIQUE_ID", "CITATION"],
        right_on=["Sample URL", "Citation"],
        how="left",
        validate="many_to_one",
    )
    missing_matches = int(matched["Citation URL"].isna().sum())
    if missing_matches:
        raise RuntimeError(f"{missing_matches} retained PetDB rows lack an exact URL/citation match")

    citation_url_counts = matched.groupby("PAPER_RECORD_ID")["Citation URL"].nunique()
    if not citation_url_counts.eq(1).all():
        raise RuntimeError("A PetDB publication record maps to multiple citation URLs")

    raw_citation_counts = raw.groupby("Citation URL").size()
    records = (
        matched.groupby("PAPER_RECORD_ID", as_index=False)
        .agg(
            PAPER_ID=("PAPER_ID", join_unique),
            CV_GROUP_V4=("CV_GROUP_V4", join_unique),
            CITATION=("CITATION", join_unique),
            PETDB_CITATION_URL=("Citation URL", join_unique),
            N_RETAINED_SAMPLES=("UNIQUE_ID", "size"),
            N_UNIQUE_SAMPLE_URLS=("UNIQUE_ID", "nunique"),
        )
        .sort_values(["PAPER_ID", "PAPER_RECORD_ID"])
    )
    records["N_ROWS_IN_ORIGINAL_EXPORT_FOR_CITATION"] = records["PETDB_CITATION_URL"].map(
        raw_citation_counts
    )
    records["SOURCE_EXPORT"] = args.raw_petdb.name
    records["DOWNLOAD_DATE"] = args.download_date
    records["MATCH_STATUS"] = "EXACT_SAMPLE_URL_AND_CITATION"
    records.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Verified {len(matched):,} retained PetDB rows")
    print(f"Verified {len(records):,} PetDB publication records")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
