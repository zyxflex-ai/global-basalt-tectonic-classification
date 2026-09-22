# Reproducibility guide

## Scope

The repository separates four reproducibility levels:

1. **Integrity audit** checks the frozen data and reported results in seconds.
2. **Validation-bias experiment** reproduces the comparison between
   sample-random and publication-grouped cross-validation.
3. **Model workflow** reproduces tuning, out-of-fold diagnostics, the frozen
   holdout evaluation, publication-cluster bootstrap intervals, and sensitivity
   analyses.
4. **Interpretation and figures** reproduces SHAP summaries and publication
   figures.
5. **Time-closed validation and label audit** repeats all data-dependent model
   choices within publications dated 2010 or earlier, evaluates 2011-2025
   publications once, and exports machine-readable label and support audits.

The original analysis was executed with Python 3.13.9 on Windows. The release
audit and frozen-model export were repeated with Python 3.12.14 and reproduced
the holdout metrics exactly. Random seeds and fixed fold assignments are stored
in the scripts and CSV files.

## Frozen datasets

| File | Purpose | Rows |
|---|---|---:|
| `02_merged/Basalt_8classes_paperid_v4.csv` | Harmonized compilation with verified publication identity | 48,403 |
| `04_split/Basalt_train_innercv_v4.csv` | Development data with fixed inner folds | 38,094 |
| `03_model_data/Basalt_train_stable50_main_v4.csv` | Stable-50 development analysis | 35,865 |
| `03_model_data/Basalt_test_stable50_main_v4.csv` | Frozen unseen-publication holdout | 10,086 |
| `04_split/split_manifest_v4.csv` | Frozen split membership and audit fields | 48,403 |

## Fast audit

```bash
python tests/test_repository_integrity.py
```

This command does not modify any file.

## Recommended execution order

The complete order can be printed with `python scripts/run_pipeline.py --list`.
The principal stages are:

### Publication identity and partitioning

- `37_build_paper_identity_v4.py` documents the cross-database publication
  reconciliation logic. Rebuilding this first stage requires the original
  database downloads and the previous audit manifest; these source downloads
  are not duplicated in the repository.
- `38_rebuild_paper_group_split_v4.py` rebuilds the frozen grouped partitions
  from `02_merged/Basalt_8classes_paperid_v4.csv`.

### Validation-bias experiment

```bash
python scripts/49_compare_random_vs_grouped_cv_v4.py
python scripts/50_plot_random_vs_grouped_cv_v4.py
```

### Model development and evaluation

Run scripts `12` through `30` in numeric order. The holdout is first accessed
by script `19`, after the development settings have been fixed. Scripts `20`
and `21` perform publication-cluster bootstrap analyses.

### Interpretation and figures

Run scripts `22` through `25` for SHAP calculations and robustness checks, then
scripts `31` through `36` for the principal diagnostic figures. Existing figure
source tables and final figures are included.

### Fully time-closed validation and v9 audits

```bash
python scripts/run_pipeline.py --stage temporal-v9
python scripts/run_pipeline.py --stage audit-v9
```

Script `81` selects features within each early-period tuning fold, compares 12
XGBoost configurations by fourfold stratified publication-group validation,
refits on publications dated through 2010, and evaluates publications dated
2011-2025. Script `82` directly compares native XGBoost missing-value handling
with training-fold median imputation on identical grouped folds. Scripts `83`
exports the recoverable label mapping, database-class-time-location support, and
multi-label publication groups. Optional script `84` audits comparator-prefixed
tokens when the original raw-download directory is supplied with `--raw-root`;
the raw third-party downloads are not redistributed here. Script `85`
recalculates frozen-holdout metrics after collapsing selected adjacent
operational labels without refitting the model.

## Expected principal results

| Evaluation | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Sample-random CV | 0.9153 | 0.8879 | 0.8846 |
| Publication-grouped CV | 0.8202 | 0.6905 | 0.6975 |
| Frozen-holdout XGBoost | 0.8518 | 0.7374 | 0.7386 |
| Frozen-holdout random forest | 0.8386 | 0.6571 | 0.6899 |
| Frozen-holdout RBF-SVM | 0.8256 | 0.6556 | 0.6716 |
| Time-closed 2011-2025 XGBoost | 0.8314 | 0.7050 | 0.6962 |

Small floating-point differences may occur across operating systems and CPU
libraries. The frozen CSV result tables are the values reported in the
manuscript.

## Computational notes

- The full pipeline may require several hours depending on CPU and memory.
- XGBoost tuning evaluates 30 candidate configurations with grouped fivefold CV.
- Publication-cluster uncertainty uses 2,000 resamples; paired model comparisons
  use 5,000 resamples.
- SHAP arrays and sample-level predictions can be regenerated but are not all
  tracked when they are unused intermediate files. The sample-level predictions
  and SHAP cache required by the manuscript figures are included.
- The paired model-comparison tables report effect differences and percentile
  intervals. The revised manuscript does not interpret uncentered bootstrap
  tail proportions as null-hypothesis P values.
