from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_results" / "audit_v9"
TOKEN = re.compile(r"^\s*(<=|>=|<|>)\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?\s*$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source CSV exports for comparator-prefixed numeric tokens.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing the original source CSV exports.")
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for path in sorted(raw_root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, dtype=str, low_memory=False, encoding_errors="replace")
        except Exception as first_exc:
            try:
                frame = pd.read_csv(
                    path, dtype=str, engine="python", on_bad_lines="warn",
                    encoding_errors="replace",
                )
            except Exception as second_exc:
                failures.append({
                    "file": str(path.relative_to(raw_root)),
                    "error": f"c_engine={first_exc!r}; python_engine={second_exc!r}",
                })
                continue
        for column in frame.columns:
            values = frame[column].dropna().astype(str)
            matched = values[values.str.match(TOKEN)]
            if matched.empty:
                continue
            operators = matched.str.extract(r"^\s*(<=|>=|<|>)", expand=False).value_counts()
            rows.append({
                "file": str(path.relative_to(raw_root)),
                "column": str(column),
                "n_censored_tokens": int(len(matched)),
                "n_lt": int(operators.get("<", 0)),
                "n_le": int(operators.get("<=", 0)),
                "n_gt": int(operators.get(">", 0)),
                "n_ge": int(operators.get(">=", 0)),
                "example_tokens": "|".join(matched.drop_duplicates().head(8).tolist()),
            })
    audit = pd.DataFrame(rows).sort_values(["n_censored_tokens", "file", "column"], ascending=[False, True, True]) if rows else pd.DataFrame(
        columns=["file", "column", "n_censored_tokens", "n_lt", "n_le", "n_gt", "n_ge", "example_tokens"]
    )
    audit.to_csv(OUT_DIR / "raw_censored_numeric_token_audit_v9.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "raw_censored_numeric_token_failures_v9.csv", index=False)
    summary = pd.DataFrame([
        {"item": "csv_files_scanned", "value": len(list(raw_root.rglob("*.csv")))},
        {"item": "files_with_censored_numeric_tokens", "value": audit["file"].nunique() if not audit.empty else 0},
        {"item": "columns_with_censored_numeric_tokens", "value": len(audit)},
        {"item": "censored_numeric_tokens", "value": int(audit["n_censored_tokens"].sum()) if not audit.empty else 0},
        {"item": "read_failures", "value": len(failures)},
    ])
    summary.to_csv(OUT_DIR / "raw_censored_numeric_token_summary_v9.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
