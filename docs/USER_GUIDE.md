# User guide

## Purpose and intended use

This repository supports evaluation and reproduction of an eight-class basalt
tectonic-setting study. Its principal methodological purpose is to compare
sample-random validation with publication-grouped validation and to provide a
frozen, publication-disjoint evaluation workflow. The trained classifier is a
research artifact, not a substitute for geological interpretation.

## Typical use cases

1. Audit the frozen split and headline metrics without model fitting.
2. Inspect the reported validation-bias and holdout values.
3. Apply the frozen XGBoost model to a compatible geochemical table.
4. Reproduce one analysis stage or the full manuscript workflow.
5. Regenerate manuscript figures from the included source data and SHAP cache.

## Input data

### Frozen files

| Path | Role | Rows |
|---|---|---:|
| `02_merged/Basalt_8classes_paperid_v4.csv` | Harmonized compilation and publication identities | 48,403 |
| `03_model_data/Basalt_train_stable50_main_v4.csv` | Frozen development data | 35,865 |
| `03_model_data/Basalt_test_stable50_main_v4.csv` | Unseen-publication holdout | 10,086 |
| `04_split/Basalt_train_innercv_v4.csv` | Fixed grouped folds | 38,094 |
| `04_split/split_manifest_v4.csv` | Frozen membership audit | 48,403 |

### Predictor schema

Inference requires these columns in this exact spelling. Values are numeric;
blank cells represent missing measurements.

```text
SIO2(WT%), TIO2(WT%), AL2O3(WT%), FE_TOTAL(WT%), CAO(WT%),
MGO(WT%), MNO(WT%), K2O(WT%), NA2O(WT%), P2O5(WT%),
V(PPM), CR(PPM), NI(PPM), RB(PPM), SR(PPM), Y(PPM),
ZR(PPM), NB(PPM), BA(PPM)
```

Each row must contain at least 10 measured values among the 19 predictors. The
class order is `CAB`, `IAB`, `IOAB`, `BABB`, `MORB`, `OIB`, `OPB`, `CFB`.
Definitions for all 66 source columns are in `docs/data_dictionary.csv`.

## Commands and options

### Integrity audit

```powershell
python tests/test_repository_integrity.py
```

Inputs: the frozen data, attribution table, result table, and model metadata.
Output: pass/fail messages only. The command does not modify files.

For byte-level validation of critical artifacts, run:

```powershell
python scripts/01_generate_checksums.py --verify
```

### Result inspection

```powershell
python examples/inspect_key_results.py
```

Inputs: frozen validation-bias and holdout metric tables. Output: six rounded
values printed to the console. No files are created.

### Frozen-model prediction

```powershell
python examples/predict_with_frozen_model.py --input INPUT.csv --output PREDICTIONS.csv [--limit N] [--model MODEL.ubj] [--metadata METADATA.json]
```

- `--input`: CSV containing all 19 predictor columns; additional columns are allowed.
- `--output`: destination CSV; parent directories are created automatically.
- `--limit`: optional positive integer for testing only the first `N` rows.
- `--model`: optional alternative XGBoost UBJ artifact.
- `--metadata`: optional matching metadata JSON.

Expected output fields are available identifiers, `N_MEASURED_STABLE50`,
`PREDICTED_LABEL`, `MAX_PROBABILITY`, and `PROB_<CLASS>` for all eight classes.
The command stops instead of silently predicting if columns are missing, values
cannot be parsed, or a row violates the 10-of-19 rule.

### Stage runner

```powershell
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --stage validation-bias
python scripts/run_pipeline.py --stage model-export
```

Valid stages are `validation-bias`, `development`, `holdout`, `model-export`,
`interpretation`, `sensitivity`, `figures`, and `all`. The runner executes the
numbered scripts in a fixed order and stops on the first error.

## Outputs and figure mapping

- `05_results/validation_bias/`: Figure 2 source tables and split audit.
- `05_results/final_test/final_independent_test_predictions_v4.csv`: Figure 3 source predictions.
- `05_results/shap/`: Figures 4, 5, and 8 source tables.
- `05_results/figures/shap_direction/stable50_independent_test_shap_values_v4.npz`: Figures 6 and 7 sample-level SHAP source.
- `03_model_data/Basalt_test_stable50_main_v4.csv`: Figure 9 plotted records.
- `figures/`: final PNG, SVG, and PDF assets plus synchronized legends.

The complete evidence mapping is in `figures/figure_contract.md`.

## Expected behaviour and runtime

- Integrity and result inspection complete in seconds on a typical laptop.
- Frozen-model inference on the supplied holdout should complete within minutes.
- Model export refits one 1,400-tree XGBoost model and verifies the holdout.
- Full grouped tuning, repeated validation, bootstrap resampling, and SHAP
  recomputation may require several hours. Runtime varies with CPU and libraries.
- The original numbered pipeline was tested on Windows with Python 3.13.9. The
  repository audit, model export, and quick examples were repeated with Python
  3.12.14. Quick checks and examples use platform-independent paths.

Recommended minimum resources are 8 GB RAM and 2 GB free disk space. A
multicore CPU is recommended for full model fitting; no GPU is required.

## Limitations

- Raw database downloads are not duplicated; their original providers remain
  the authoritative source.
- The frozen model has not established transfer performance outside the
  represented publication and geochemical domains.
- SHAP values describe model attribution and not causal geochemical controls.
- Classical Ti-Zr-Y boundaries are used only for qualitative visualization.
- Third-party database records retain their original attribution and licence terms.

## Troubleshooting

- `ModuleNotFoundError`: activate the environment and reinstall `requirements.txt`.
- Missing feature error: match column spelling to the predictor schema above.
- Stable-50 rule error: provide at least 10 measured predictors per row.
- Metric mismatch: verify package versions and run the integrity test before refitting.
- Windows execution-policy error during environment activation: use a terminal
  where locally created virtual-environment scripts are permitted, or invoke
  `.venv\Scripts\python.exe` directly.
