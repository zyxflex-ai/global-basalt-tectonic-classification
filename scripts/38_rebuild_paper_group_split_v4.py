"""Rebuild the frozen holdout and inner folds with CV_GROUP_V4.

The v4 split is deliberately conservative.  It uses the leakage-aware group
created by 37_build_paper_identity_v4.py and preserves all previous v3 files.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from _project_paths import FINAL_DIR


INPUT_FILE = FINAL_DIR / "02_merged" / "Basalt_8classes_paperid_v4.csv"
SPLIT_DIR = FINAL_DIR / "04_split"
MODEL_DIR = FINAL_DIR / "03_model_data"
AUDIT_DIR = FINAL_DIR / "05_results" / "audit"

TRAIN_FILE = SPLIT_DIR / "Basalt_train_v4.csv"
TEST_FILE = SPLIT_DIR / "Basalt_test_holdout_v4.csv"
MANIFEST_FILE = SPLIT_DIR / "split_manifest_v4.csv"
INNER_FILE = SPLIT_DIR / "Basalt_train_innercv_v4.csv"
TRAIN_MAIN_FILE = MODEL_DIR / "Basalt_train_stable50_main_v4.csv"
TEST_MAIN_FILE = MODEL_DIR / "Basalt_test_stable50_main_v4.csv"
EXCLUDED_TRAIN_FILE = MODEL_DIR / "excluded_train_low_completeness_v4.csv"
EXCLUDED_TEST_FILE = MODEL_DIR / "excluded_test_low_completeness_v4.csv"
SUMMARY_FILE = AUDIT_DIR / "split_audit_summary_v4.txt"

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


def assert_all_classes(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(CLASSES) - set(frame["LABEL"]))
    if missing:
        raise RuntimeError(f"{name} is missing classes: {missing}")


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)
    group_column = "CV_GROUP_V4"
    required = {group_column, "LABEL", "GROUP_ID", "CITATION", "SAMPLE NAME", *FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    selected_fold = 0
    train_index = test_index = None
    for fold, (fold_train, fold_test) in enumerate(
        outer.split(data, data["LABEL"], data[group_column])
    ):
        if fold == selected_fold:
            train_index, test_index = fold_train, fold_test
            break
    if train_index is None or test_index is None:
        raise RuntimeError("Failed to create outer split")

    train = data.iloc[train_index].copy()
    test = data.iloc[test_index].copy()
    outer_overlap = set(train[group_column]) & set(test[group_column])
    if outer_overlap:
        raise RuntimeError(f"Outer group leakage detected: {len(outer_overlap)} groups")
    assert_all_classes(train, "Outer train")
    assert_all_classes(test, "Outer test")

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=180)
    train["INNER_FOLD"] = -1
    for fold, (_, validation_index) in enumerate(
        inner.split(train, train["LABEL"], train[group_column])
    ):
        train.loc[train.index[validation_index], "INNER_FOLD"] = fold
    if (train["INNER_FOLD"] < 0).any():
        raise RuntimeError("Some training rows did not receive an inner fold")
    cross_fold_groups = (
        train.groupby(group_column)["INNER_FOLD"].nunique().gt(1).sum()
    )
    if cross_fold_groups:
        raise RuntimeError(f"Inner-fold group leakage detected: {cross_fold_groups} groups")

    for frame in (train, test):
        frame["N_STABLE50"] = frame[FEATURES].notna().sum(axis=1)

    train_main = train[train["N_STABLE50"] >= 10].copy()
    test_main = test[test["N_STABLE50"] >= 10].copy()
    excluded_train = train[train["N_STABLE50"] < 10].copy()
    excluded_test = test[test["N_STABLE50"] < 10].copy()
    assert_all_classes(train_main, "Stable-50 train")
    assert_all_classes(test_main, "Stable-50 test")

    stable_overlap = set(train_main[group_column]) & set(test_main[group_column])
    if stable_overlap:
        raise RuntimeError(f"Stable-50 group leakage detected: {len(stable_overlap)} groups")

    train.drop(columns=["N_STABLE50"]).to_csv(
        TRAIN_FILE, index=False, encoding="utf-8-sig"
    )
    test.drop(columns=["N_STABLE50"]).to_csv(
        TEST_FILE, index=False, encoding="utf-8-sig"
    )
    train.to_csv(INNER_FILE, index=False, encoding="utf-8-sig")
    train_main.to_csv(TRAIN_MAIN_FILE, index=False, encoding="utf-8-sig")
    test_main.to_csv(TEST_MAIN_FILE, index=False, encoding="utf-8-sig")
    excluded_train.to_csv(EXCLUDED_TRAIN_FILE, index=False, encoding="utf-8-sig")
    excluded_test.to_csv(EXCLUDED_TEST_FILE, index=False, encoding="utf-8-sig")

    test_ids = set(test["GROUP_ID"])
    manifest_columns = [
        "GROUP_ID",
        "PAPER_RECORD_ID",
        "PAPER_ID",
        "PAPER_ID_STATUS",
        "CV_GROUP_V4",
        "CV_GROUP_V4_BASIS",
        "DATABASE",
        "CITATION",
        "SAMPLE NAME",
        "LABEL",
    ]
    manifest = data[manifest_columns].copy()
    manifest["SET"] = manifest["GROUP_ID"].map(
        lambda value: "TEST" if value in test_ids else "TRAIN"
    )
    manifest.to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")

    candidate_file = AUDIT_DIR / "paper_match_candidates_v4.csv"
    candidates = pd.read_csv(candidate_file, encoding="utf-8-sig")
    unresolved_candidates = int(
        (candidates["MATCH_DECISION"] == "MANUAL_REVIEW_REQUIRED").sum()
    )
    record_sets = (
        manifest.groupby("PAPER_RECORD_ID")["SET"]
        .agg(lambda values: set(values))
        .to_dict()
    )
    candidate_crossings = 0
    for _, row in candidates.iterrows():
        sets = set(record_sets.get(row["GEOROC_RECORD_ID"], set()))
        sets |= set(record_sets.get(row["PETDB_RECORD_ID"], set()))
        candidate_crossings += int({"TRAIN", "TEST"}.issubset(sets))
    if candidate_crossings:
        raise RuntimeError(
            f"Cross-database candidate leakage remains in v4: {candidate_crossings} pairs"
        )

    train_counts = train_main["LABEL"].value_counts().reindex(CLASSES, fill_value=0)
    test_counts = test_main["LABEL"].value_counts().reindex(CLASSES, fill_value=0)
    count_lines = [
        f"{label}: train={int(train_counts[label])}, test={int(test_counts[label])}"
        for label in CLASSES
    ]
    summary_lines = [
        "Leakage-aware split audit v4",
        f"All rows: {len(data)}",
        f"Outer train rows: {len(train)}",
        f"Outer test rows: {len(test)}",
        f"Outer test fraction: {len(test) / len(data):.6f}",
        f"Outer train groups: {train[group_column].nunique()}",
        f"Outer test groups: {test[group_column].nunique()}",
        f"Outer group overlap: {len(outer_overlap)}",
        f"Inner-fold cross-group overlap: {cross_fold_groups}",
        f"Stable-50 train rows: {len(train_main)}",
        f"Stable-50 test rows: {len(test_main)}",
        f"Stable-50 train groups: {train_main[group_column].nunique()}",
        f"Stable-50 test groups: {test_main[group_column].nunique()}",
        f"Stable-50 group overlap: {len(stable_overlap)}",
        f"Cross-database candidate pairs crossing v4 split: {candidate_crossings}",
        "",
        "Stable-50 class counts:",
        *count_lines,
        "",
        f"Unresolved citation candidate pairs: {unresolved_candidates}",
        "Status: final citation-identity grouping."
        if unresolved_candidates == 0
        else "Status: provisional until citation-candidate review is complete.",
    ]
    SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved: {TRAIN_FILE}")
    print(f"Saved: {TEST_FILE}")
    print(f"Saved: {MANIFEST_FILE}")
    print(f"Saved: {INNER_FILE}")
    print(f"Saved: {TRAIN_MAIN_FILE}")
    print(f"Saved: {TEST_MAIN_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
