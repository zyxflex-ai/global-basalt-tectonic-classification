"""Generate or verify SHA-256 checksums for critical release artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checksums.sha256"
TARGETS = [
    "02_merged/Basalt_8classes_paperid_v4.csv",
    "03_model_data/Basalt_train_stable50_main_v4.csv",
    "03_model_data/Basalt_test_stable50_main_v4.csv",
    "04_split/Basalt_train_innercv_v4.csv",
    "04_split/split_manifest_v4.csv",
    "05_results/final_test/final_independent_test_metrics_v4.csv",
    "05_results/final_test/final_independent_test_predictions_v4.csv",
    "05_results/validation_bias/random_vs_grouped_cv_metrics_v4.csv",
    "05_results/figures/shap_direction/stable50_independent_test_shap_values_v4.npz",
    "models/xgboost_stable50_v4.ubj",
    "models/xgboost_stable50_v4.metadata.json",
    "references/petdb_source_records.csv",
    "references/source_publications.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_entries() -> dict[str, str]:
    entries = {}
    for relative in TARGETS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[relative] = sha256(path)
    return entries


def read_manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check without modifying the manifest.")
    args = parser.parse_args()
    observed = current_entries()

    if args.verify:
        expected = read_manifest()
        if expected != observed:
            missing = sorted(set(observed) - set(expected))
            extra = sorted(set(expected) - set(observed))
            changed = sorted(key for key in observed.keys() & expected.keys() if observed[key] != expected[key])
            raise RuntimeError(
                f"Checksum verification failed; missing={missing}, extra={extra}, changed={changed}"
            )
        print(f"Verified {len(observed)} critical release artifacts.")
        return

    content = "".join(f"{digest}  {relative}\n" for relative, digest in observed.items())
    MANIFEST.write_text(content, encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST} with {len(observed)} entries.")


if __name__ == "__main__":
    main()
