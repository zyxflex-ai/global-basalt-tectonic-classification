"""Build explicit database provenance and leakage-aware paper identities.

This script preserves the v2/v3 inputs and writes a new v4 dataset plus audit
tables.  PAPER_ID only merges cross-database citation records when sample-level
evidence supports the match.  CV_GROUP_V4 additionally purges every unresolved
cross-database author-year candidate into one conservative validation group so
that a possible duplicate publication cannot straddle train and test.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from _project_paths import FINAL_DIR


INPUT_FILE = FINAL_DIR / "02_merged" / "Basalt_8classes_merged_v2.csv"
OLD_MANIFEST_FILE = FINAL_DIR / "04_split" / "split_manifest_v3.csv"
OUTPUT_FILE = FINAL_DIR / "02_merged" / "Basalt_8classes_paperid_v4.csv"
AUDIT_DIR = FINAL_DIR / "05_results" / "audit"
CANDIDATE_FILE = AUDIT_DIR / "paper_match_candidates_v4.csv"
MANUAL_DECISION_FILE = AUDIT_DIR / "paper_match_manual_decisions_v4.csv"
CONFLICT_FILE = AUDIT_DIR / "paper_label_conflicts_v4.csv"
SUMMARY_FILE = AUDIT_DIR / "paper_identity_audit_summary_v4.txt"

STABLE_FEATURES = [
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


def ascii_upper(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.upper().strip()


def compact(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", ascii_upper(value))


def normalize_database(value: object) -> str:
    text = ascii_upper(value)
    if not text:
        return "GEOROC"
    if text == "PETDB":
        return "PetDB"
    if text == "GEOROC":
        return "GEOROC"
    raise ValueError(f"Unexpected DATABASE value: {value!r}")


def citation_year(citation: object) -> str:
    match = re.search(r"\((19\d{2}|20\d{2})\)", str(citation))
    return match.group(1) if match else "UNKNOWN"


def first_author(citation: object, database: str) -> str:
    text = ascii_upper(citation)
    if database == "PetDB":
        surname = text.split(",", maxsplit=1)[0]
    else:
        text = re.sub(r"^\[\d+\]\s*", "", text)
        surname = text.split(maxsplit=1)[0] if text else ""
    normalized = compact(surname)
    return normalized or "UNKNOWN"


def georoc_citation_id(citation: object) -> str | None:
    match = re.match(r"^\[(\d+)\]", str(citation).strip())
    return match.group(1) if match else None


def citation_doi(citation: object) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", ascii_upper(citation))
    return match.group(0).rstrip(".,;)") if match else None


def citation_record_id(citation: object, database: str) -> str:
    if database == "GEOROC":
        source_id = georoc_citation_id(citation)
        if source_id:
            return f"GEOROC|CITATION_ID|{source_id}"
    digest = hashlib.sha1(compact(citation).encode("utf-8")).hexdigest()[:16]
    return f"{database.upper()}|CITATION_SHA1|{digest}"


def normalize_sample_name(value: object) -> str:
    text = ascii_upper(value)
    text = re.sub(r"^SAMP\.?\s*", "", text)
    return compact(text)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            keep, merge = sorted([root_left, root_right])
            self.parent[merge] = keep


def corroborated_sample_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    shared_names = sorted(set(left["_SAMPLE_NORM"]) & set(right["_SAMPLE_NORM"]))
    shared_names = [name for name in shared_names if len(name) >= 3]
    corroborated = 0

    for sample_name in shared_names:
        left_rows = left[left["_SAMPLE_NORM"] == sample_name]
        right_rows = right[right["_SAMPLE_NORM"] == sample_name]
        found = False

        for _, left_row in left_rows.iterrows():
            if found:
                break
            for _, right_row in right_rows.iterrows():
                chemical_matches = 0
                for feature in STABLE_FEATURES:
                    left_value = pd.to_numeric(left_row[feature], errors="coerce")
                    right_value = pd.to_numeric(right_row[feature], errors="coerce")
                    if pd.isna(left_value) or pd.isna(right_value):
                        continue
                    tolerance = max(1e-6, 1e-4 * max(abs(left_value), abs(right_value), 1.0))
                    if abs(left_value - right_value) <= tolerance:
                        chemical_matches += 1

                lat_left = pd.to_numeric(left_row["LATITUDE (MAX.)"], errors="coerce")
                lat_right = pd.to_numeric(right_row["LATITUDE (MAX.)"], errors="coerce")
                lon_left = pd.to_numeric(left_row["LONGITUDE (MAX.)"], errors="coerce")
                lon_right = pd.to_numeric(right_row["LONGITUDE (MAX.)"], errors="coerce")
                coordinates_match = (
                    pd.notna(lat_left)
                    and pd.notna(lat_right)
                    and pd.notna(lon_left)
                    and pd.notna(lon_right)
                    and abs(lat_left - lat_right) <= 0.05
                    and abs(lon_left - lon_right) <= 0.05
                )

                if chemical_matches >= 3 or (coordinates_match and chemical_matches >= 2):
                    found = True
                    break

        if found:
            corroborated += 1

    return corroborated


def joined_unique(series: pd.Series) -> str:
    return " || ".join(sorted({str(value) for value in series if pd.notna(value)}))


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
    required = {
        "DATABASE",
        "CITATION",
        "GROUP_ID",
        "SAMPLE NAME",
        "LABEL",
        "LATITUDE (MAX.)",
        "LONGITUDE (MAX.)",
        *STABLE_FEATURES,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    data["DATABASE"] = data["DATABASE"].map(normalize_database)
    data["PAPER_YEAR"] = data["CITATION"].map(citation_year)
    data["PAPER_FIRST_AUTHOR"] = [
        first_author(citation, database)
        for citation, database in zip(data["CITATION"], data["DATABASE"])
    ]
    data["PAPER_MATCH_KEY"] = (
        data["PAPER_FIRST_AUTHOR"] + "|" + data["PAPER_YEAR"]
    )
    data["PAPER_DOI"] = data["CITATION"].map(citation_doi)
    data["PAPER_RECORD_ID"] = [
        citation_record_id(citation, database)
        for citation, database in zip(data["CITATION"], data["DATABASE"])
    ]
    data["_SAMPLE_NORM"] = data["SAMPLE NAME"].map(normalize_sample_name)

    manual_decisions: dict[tuple[str, str], dict[str, str]] = {}
    if MANUAL_DECISION_FILE.exists():
        reviewed = pd.read_csv(MANUAL_DECISION_FILE, encoding="utf-8-sig").fillna("")
        required_review_columns = {
            "GEOROC_RECORD_ID",
            "PETDB_RECORD_ID",
            "REVIEW_DECISION",
        }
        missing_review_columns = sorted(required_review_columns - set(reviewed.columns))
        if missing_review_columns:
            raise RuntimeError(
                f"Missing manual-decision columns: {missing_review_columns}"
            )
        allowed_decisions = {
            "VERIFIED_SAME_PUBLICATION",
            "VERIFIED_DIFFERENT_PUBLICATION",
        }
        for _, row in reviewed.iterrows():
            decision = str(row["REVIEW_DECISION"]).strip()
            if decision not in allowed_decisions:
                raise RuntimeError(f"Unexpected REVIEW_DECISION: {decision!r}")
            key = (
                str(row["GEOROC_RECORD_ID"]).strip(),
                str(row["PETDB_RECORD_ID"]).strip(),
            )
            if key in manual_decisions:
                raise RuntimeError(f"Duplicate manual-decision pair: {key}")
            manual_decisions[key] = {
                "decision": decision,
                "basis": str(row.get("VERIFICATION_BASIS", "")).strip(),
            }

    if OLD_MANIFEST_FILE.exists():
        old_manifest = pd.read_csv(
            OLD_MANIFEST_FILE,
            encoding="utf-8-sig",
            low_memory=False,
            usecols=["GROUP_ID", "SET"],
        )
        old_set = old_manifest.drop_duplicates("GROUP_ID").set_index("GROUP_ID")["SET"]
        data["OLD_SET_V3"] = data["GROUP_ID"].map(old_set).fillna("UNKNOWN")
    else:
        data["OLD_SET_V3"] = "UNKNOWN"

    record_ids = sorted(data["PAPER_RECORD_ID"].unique())
    union_find = UnionFind(record_ids)
    candidate_rows: list[dict[str, object]] = []
    auto_confirmed_records: set[str] = set()
    verified_same_records: set[str] = set()
    verified_different_records: set[str] = set()
    review_records: set[str] = set()

    for match_key, key_rows in data.groupby("PAPER_MATCH_KEY", sort=True):
        georoc_ids = sorted(
            key_rows.loc[key_rows["DATABASE"] == "GEOROC", "PAPER_RECORD_ID"].unique()
        )
        petdb_ids = sorted(
            key_rows.loc[key_rows["DATABASE"] == "PetDB", "PAPER_RECORD_ID"].unique()
        )
        if not georoc_ids or not petdb_ids:
            continue

        for georoc_id in georoc_ids:
            georoc_rows = data[data["PAPER_RECORD_ID"] == georoc_id]
            for petdb_id in petdb_ids:
                petdb_rows = data[data["PAPER_RECORD_ID"] == petdb_id]
                georoc_samples = {
                    value for value in georoc_rows["_SAMPLE_NORM"] if len(value) >= 3
                }
                petdb_samples = {
                    value for value in petdb_rows["_SAMPLE_NORM"] if len(value) >= 3
                }
                sample_overlap = sorted(georoc_samples & petdb_samples)
                corroborated = corroborated_sample_count(georoc_rows, petdb_rows)

                georoc_dois = {value for value in georoc_rows["PAPER_DOI"] if pd.notna(value)}
                petdb_dois = {value for value in petdb_rows["PAPER_DOI"] if pd.notna(value)}
                doi_match = bool(georoc_dois & petdb_dois)

                auto_confirm = doi_match or (
                    corroborated >= 1
                    and (
                        len(sample_overlap) >= 2
                        or any(len(name) >= 5 for name in sample_overlap)
                    )
                )

                pair_key = (georoc_id, petdb_id)
                manual_review = manual_decisions.get(pair_key)
                verification_basis = ""
                if manual_review is not None:
                    decision = manual_review["decision"]
                    verification_basis = manual_review["basis"]
                    if decision == "VERIFIED_SAME_PUBLICATION":
                        union_find.union(georoc_id, petdb_id)
                        verified_same_records.update([georoc_id, petdb_id])
                    else:
                        verified_different_records.update([georoc_id, petdb_id])
                elif auto_confirm:
                    union_find.union(georoc_id, petdb_id)
                    auto_confirmed_records.update([georoc_id, petdb_id])
                    decision = "AUTO_CONFIRMED_SAMPLE_OR_DOI_EVIDENCE"
                else:
                    review_records.update([georoc_id, petdb_id])
                    decision = "MANUAL_REVIEW_REQUIRED"

                old_sets = sorted(
                    set(georoc_rows["OLD_SET_V3"]) | set(petdb_rows["OLD_SET_V3"])
                )
                candidate_rows.append(
                    {
                        "PAPER_MATCH_KEY": match_key,
                        "GEOROC_RECORD_ID": georoc_id,
                        "PETDB_RECORD_ID": petdb_id,
                        "GEOROC_CITATION": joined_unique(georoc_rows["CITATION"]),
                        "PETDB_CITATION": joined_unique(petdb_rows["CITATION"]),
                        "GEOROC_LABELS": joined_unique(georoc_rows["LABEL"]),
                        "PETDB_LABELS": joined_unique(petdb_rows["LABEL"]),
                        "GEOROC_SAMPLES": len(georoc_rows),
                        "PETDB_SAMPLES": len(petdb_rows),
                        "OVERLAPPING_SAMPLE_NAMES": len(sample_overlap),
                        "CORROBORATED_SAMPLE_NAMES": corroborated,
                        "SAMPLE_NAME_EXAMPLES": "|".join(sample_overlap[:10]),
                        "DOI_MATCH": doi_match,
                        "OLD_SPLIT_SETS": "|".join(old_sets),
                        "CROSSED_V3_TRAIN_TEST": {"TRAIN", "TEST"}.issubset(old_sets),
                        "MATCH_DECISION": decision,
                        "VERIFICATION_BASIS": verification_basis,
                    }
                )

    generated_pairs = {
        (row["GEOROC_RECORD_ID"], row["PETDB_RECORD_ID"])
        for row in candidate_rows
    }
    unused_decisions = sorted(set(manual_decisions) - generated_pairs)
    if unused_decisions:
        raise RuntimeError(f"Manual decisions do not match generated candidates: {unused_decisions}")

    components: dict[str, list[str]] = {}
    for record_id in record_ids:
        root = union_find.find(record_id)
        components.setdefault(root, []).append(record_id)

    paper_id_by_record: dict[str, str] = {}
    for members in components.values():
        if len(members) == 1:
            paper_id = f"PAPER|{members[0]}"
        else:
            digest = hashlib.sha1("||".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
            paper_id = f"PAPER|MERGED|{digest}"
        for member in members:
            paper_id_by_record[member] = paper_id

    data["PAPER_ID"] = data["PAPER_RECORD_ID"].map(paper_id_by_record)

    def paper_status(record_id: str) -> str:
        if record_id in verified_same_records:
            return "VERIFIED_CROSS_DATABASE_SAME"
        if record_id in verified_different_records:
            return "VERIFIED_CROSS_DATABASE_DISTINCT"
        if record_id in auto_confirmed_records:
            return "AUTO_CONFIRMED_CROSS_DATABASE"
        if record_id in review_records:
            return "CROSS_DATABASE_REVIEW_NEEDED"
        return "SOURCE_RECORD_ONLY"

    data["PAPER_ID_STATUS"] = data["PAPER_RECORD_ID"].map(paper_status)
    # Confirmed matches share PAPER_ID. Only still-unresolved candidate pairs
    # receive an additional conservative CV grouping; verified-different pairs
    # remain independent even when their first-author surname and year coincide.
    cv_union = UnionFind(record_ids)
    unresolved_pairs: list[tuple[str, str]] = []
    for row in candidate_rows:
        left = str(row["GEOROC_RECORD_ID"])
        right = str(row["PETDB_RECORD_ID"])
        decision = str(row["MATCH_DECISION"])
        if decision == "MANUAL_REVIEW_REQUIRED":
            cv_union.union(left, right)
            unresolved_pairs.append((left, right))

    unresolved_roots = {cv_union.find(left) for left, _ in unresolved_pairs}
    cv_group_by_record: dict[str, str] = {}
    cv_basis_by_record: dict[str, str] = {}
    for record_id in record_ids:
        root = cv_union.find(record_id)
        if root in unresolved_roots:
            members = sorted(value for value in record_ids if cv_union.find(value) == root)
            digest = hashlib.sha1("||".join(members).encode("utf-8")).hexdigest()[:16]
            cv_group_by_record[record_id] = f"PURGED|CANDIDATE|{digest}"
            cv_basis_by_record[record_id] = "UNRESOLVED_CROSS_DATABASE_PURGE"
        else:
            cv_group_by_record[record_id] = paper_id_by_record[record_id]
            cv_basis_by_record[record_id] = "PAPER_ID"

    data["CV_GROUP_V4"] = data["PAPER_RECORD_ID"].map(cv_group_by_record)
    data["CV_GROUP_V4_BASIS"] = data["PAPER_RECORD_ID"].map(cv_basis_by_record)

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["CROSSED_V3_TRAIN_TEST", "MATCH_DECISION", "PAPER_MATCH_KEY"],
        ascending=[False, True, True],
    )
    candidates.to_csv(CANDIDATE_FILE, index=False, encoding="utf-8-sig")

    conflict_rows: list[dict[str, object]] = []
    for group_id, group_rows in data.groupby("CV_GROUP_V4", sort=True):
        labels = sorted(group_rows["LABEL"].unique())
        if len(labels) <= 1:
            continue
        conflict_rows.append(
            {
                "CV_GROUP_V4": group_id,
                "N_LABELS": len(labels),
                "LABELS": "|".join(labels),
                "N_SAMPLES": len(group_rows),
                "N_PAPER_RECORDS": group_rows["PAPER_RECORD_ID"].nunique(),
                "DATABASES": joined_unique(group_rows["DATABASE"]),
                "CITATIONS": joined_unique(group_rows["CITATION"]),
                "OLD_SPLIT_SETS": joined_unique(group_rows["OLD_SET_V3"]),
            }
        )
    conflicts = pd.DataFrame(conflict_rows).sort_values(
        ["N_SAMPLES", "CV_GROUP_V4"], ascending=[False, True]
    )
    conflicts.to_csv(CONFLICT_FILE, index=False, encoding="utf-8-sig")

    output = data.drop(columns=["_SAMPLE_NORM"])
    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    database_counts = output["DATABASE"].value_counts().to_dict()
    summary_lines = [
        "Paper identity audit v4",
        f"Input rows: {len(output)}",
        f"GEOROC rows: {database_counts.get('GEOROC', 0)}",
        f"PetDB rows: {database_counts.get('PetDB', 0)}",
        f"Source citation records: {output['PAPER_RECORD_ID'].nunique()}",
        f"Evidence-based PAPER_ID values: {output['PAPER_ID'].nunique()}",
        f"Leakage-aware CV_GROUP_V4 values: {output['CV_GROUP_V4'].nunique()}",
        f"Cross-database candidate pairs: {len(candidates)}",
        "Auto-confirmed candidate pairs: "
        f"{int((candidates['MATCH_DECISION'] == 'AUTO_CONFIRMED_SAMPLE_OR_DOI_EVIDENCE').sum())}",
        "Verified-same candidate pairs: "
        f"{int((candidates['MATCH_DECISION'] == 'VERIFIED_SAME_PUBLICATION').sum())}",
        "Verified-different candidate pairs: "
        f"{int((candidates['MATCH_DECISION'] == 'VERIFIED_DIFFERENT_PUBLICATION').sum())}",
        "Unresolved manual-review candidate pairs: "
        f"{int((candidates['MATCH_DECISION'] == 'MANUAL_REVIEW_REQUIRED').sum())}",
        "Candidate pairs crossing the old v3 split: "
        f"{int(candidates['CROSSED_V3_TRAIN_TEST'].sum())}",
        f"Multi-label validation groups: {len(conflicts)}",
        "",
        "Notes:",
        "- Blank DATABASE values were filled as GEOROC per author confirmation.",
        "- PAPER_ID merges require DOI or corroborated overlapping sample evidence.",
        "- Verified-different same-surname/year records remain separate.",
        "- CV_GROUP_V4 conservatively groups only unresolved cross-database candidates.",
    ]
    SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {CANDIDATE_FILE}")
    print(f"Saved: {CONFLICT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
