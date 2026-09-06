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
|-- examples/                  Tested result-inspection and prediction examples
|-- models/                    Frozen XGBoost artifact and machine-readable metadata
|-- docs/                      Tutorials, user guide, data dictionary, and reproducibility notes
|-- references/                Source-publication attribution table
|-- tests/                     Fast integrity and leakage checks
|-- checksums.sha256           SHA-256 manifest for critical release artifacts
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

The original analysis was run on Windows with Python 3.13.9. The release audit
and frozen-model export were independently repeated with Python 3.12.14 and
reproduced the reported holdout metrics exactly. In Windows PowerShell, create
a fresh environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Linux or macOS, replace the activation command with
`source .venv/bin/activate`. The integrity and inspection examples are platform
independent; the historical numbered analysis scripts were executed on Windows.

Recommended resources are at least 8 GB RAM and 2 GB free disk space for the
provided data, environment, and generated outputs. The full tuning and SHAP
workflow benefits from a multicore CPU and may take several hours.

## Tutorials and user documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md): verify the release, inspect the
  reported values, and run a 20-row prediction example.
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md): input schema, command options,
  outputs, expected behaviour, compute requirements, and troubleshooting.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): complete frozen workflow
  and expected manuscript results.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md): principal variables,
  units, coding, and missing-value rules.
- [`docs/SUBMISSION_READINESS.md`](docs/SUBMISSION_READINESS.md): completed
  checks and the remaining public-release actions.
- [`docs/PETDB_LICENSE_AUDIT.md`](docs/PETDB_LICENSE_AUDIT.md): PetDB provenance,
  attribution, licence boundaries, and release conditions.

## Quick verification

Run the non-destructive integrity test before any model fitting:

```bash
python tests/test_repository_integrity.py
```

The test verifies sample counts, class counts, publication-group isolation,
inner-fold isolation, source-attribution cardinality, the frozen model artifact,
and the principal manuscript metrics.

To inspect the key results without refitting a model:

```powershell
python examples/inspect_key_results.py
```

Verify the critical large files and model artifact against the release manifest:

```powershell
python scripts/01_generate_checksums.py --verify
```

The repository also includes a directly loadable frozen XGBoost model. A tested
inference command is provided in the quick-start tutorial.

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

Full model tuning and SHAP analysis are computationally intensive. The frozen
model, sample-level holdout predictions, signed SHAP cache, result tables, and
publication figures are included so reviewers can inspect or regenerate the
reported evidence without rerunning every expensive step.

## Data provenance and licensing

The compiled records were downloaded from
[GEOROC](https://georoc.eu/) and [PetDB/EarthChem](https://earthchem.org/petdb)
between 7 and 11 August 2026. GEOROC-derived data must be distributed under
CC BY-SA 4.0 with attribution to GEOROC and the original data sources. EarthChem
materials remain subject to the applicable EarthChem and source-record terms;
this repository does not assign a new uniform licence to PetDB-derived values.
See [`DATA_LICENSE.md`](DATA_LICENSE.md) before reuse.

The file `references/source_publications.csv` maps every retained publication
group to its database, citation, DOI when available, and sample count. This is
provided to support attribution to the original data producers. The companion
file `references/petdb_source_records.csv` maps all 303 retained PetDB
publication records to their PetDB citation URLs and verified download
provenance.

## Citation

Please cite the associated manuscript and this repository. Citation metadata are
provided in [`CITATION.cff`](CITATION.cff). A versioned archival DOI will be
added after the first public release is deposited in Zenodo.

## Contact

Yuxuan Zhang  
School of Earth Sciences and Engineering, China University of
Petroleum-Beijing at Karamay, Karamay 834000, Xinjiang, China  
Email: 2025015169@st.cupk.edu.cn  
ORCID: https://orcid.org/0009-0004-0582-8349
