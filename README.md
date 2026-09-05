# Publication-aware tectonic classification of global basalts

This repository contains the data, code, result tables, and figure assets for
the manuscript **“Publication-Grouped Validation Reveals Optimistic Bias in
Machine-Learning Tectonic Classification of Global Basalts.”**

The central computational result is that sample-random cross-validation can
substantially overestimate performance when samples from the same source
publication occur in both training and validation sets. The repository therefore
uses source publication as the grouping unit throughout model development and
reserves a frozen holdout containing previously unseen publication groups.

## Main results

- 48,403 harmonized basalt records from GEOROC and PetDB/EarthChem.
- 2,430 verified publication groups.
- Sample-random cross-validation macro-F1: 0.8846.
- Publication-grouped cross-validation macro-F1: 0.6975.
- Median macro-F1 inflation: 0.1873 (publication-cluster bootstrap).
- Frozen-holdout XGBoost macro-F1: 0.7386.
- Frozen-holdout XGBoost balanced accuracy: 0.7374.

## Repository contents

```text
.
|-- 02_merged/                 Harmonized records with publication identities
|-- 03_model_data/             Stable-50 development and frozen-holdout data
|-- 04_split/                  Inner-CV folds and frozen split manifest
|-- 05_results/                Audits, metrics, source tables, and summaries
|-- figures/                   Submission figures in PNG, SVG, and PDF
|-- scripts/                   Analysis and figure-generation scripts
|-- docs/                      Data dictionary and reproducibility notes
|-- references/                Source-publication attribution table
|-- tests/                     Fast integrity and leakage checks
|-- CITATION.cff               Repository citation metadata
|-- DATA_LICENSE.md            Third-party data and derived-data terms
|-- LICENSE                    MIT licence for original software
|-- requirements.txt           Tested Python dependencies
```

The numbered directories are retained because they are part of the frozen
analysis paths used for the manuscript. The scripts resolve the repository root
relative to `scripts/_project_paths.py`; no author-specific absolute path is
required.

## Installation

The analysis was run on Windows with Python 3.13.9. Create a fresh environment
and install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux and macOS users can activate the environment with
`source .venv/bin/activate`. The repository integrity test is platform
independent; the historical numbered analysis scripts were executed on Windows.

## Quick verification

Run the non-destructive integrity test before any model fitting:

```bash
python tests/test_repository_integrity.py
```

The test verifies sample counts, class counts, publication-group isolation,
inner-fold isolation, and the principal manuscript metrics.

## Reproducing the analyses

The main execution order is documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). A machine-readable command
list can be printed with:

```bash
python scripts/run_pipeline.py --list
```

To run a named stage, for example the validation-bias experiment:

```bash
python scripts/run_pipeline.py --stage validation-bias
```

Full model tuning and SHAP analysis are computationally intensive. Existing
result tables and publication figures are included so that reviewers can inspect
the reported evidence without rerunning every expensive step.

## Data provenance and licensing

The compiled records were downloaded from
[GEOROC](https://georoc.eu/) and [PetDB/EarthChem](https://earthchem.org/petdb)
between 7 and 11 August 2026. GEOROC-derived data must be distributed under
CC BY-SA 4.0 with attribution to GEOROC and the original data sources. EarthChem
materials remain subject to the licence and citation terms attached to their
source records. See [`DATA_LICENSE.md`](DATA_LICENSE.md) before reuse.

The file `references/source_publications.csv` maps every retained publication
group to its database, citation, DOI when available, and sample count. This is
provided to support attribution to the original data producers.

## Citation

Please cite the associated manuscript and this repository. Citation metadata are
provided in [`CITATION.cff`](CITATION.cff). A versioned archival DOI will be
added after the first public release is deposited in Zenodo.

## Contact

Yuxuan Zhang  
School of Earth Sciences and Engineering, China University of
Petroleum-Beijing at Karamay, Karamay 834000, Xinjiang, China  
Email: 3380827355@qq.com  
ORCID: https://orcid.org/0009-0004-0582-8349

