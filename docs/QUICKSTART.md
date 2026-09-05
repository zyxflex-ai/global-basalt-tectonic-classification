# Quick start

This tutorial verifies the frozen release, inspects the central results, and
applies the trained classifier to 20 rows. It does not refit or tune a model.

## 1. Create the environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Verify frozen data and artifacts

```powershell
python tests/test_repository_integrity.py
```

Expected final lines:

```text
Repository integrity checks passed.
48,403 samples; 2,430 publication groups; no train-holdout group leakage.
```

This is a read-only check. It verifies row counts, class counts, publication
isolation, inner-fold isolation, attribution metadata, the model artifact, and
the headline holdout metrics.

Verify byte-level integrity of the critical data and model files:

```powershell
python scripts/01_generate_checksums.py --verify
```

## 3. Inspect the key results

```powershell
python examples/inspect_key_results.py
```

Expected rounded values:

```text
Publication-grouped OOF macro-F1 : 0.6975
Sample-random OOF macro-F1       : 0.8846
Apparent inflation                : 0.1870
XGBoost holdout accuracy          : 0.8518
XGBoost holdout balanced accuracy : 0.7374
XGBoost holdout macro-F1          : 0.7386
```

## 4. Apply the frozen model

The following command uses the first 20 rows of the supplied frozen holdout as
a format-compatible example:

```powershell
python examples/predict_with_frozen_model.py --input 03_model_data/Basalt_test_stable50_main_v4.csv --output quickstart_predictions.csv --limit 20
```

The output contains available identifiers, the measured-feature count,
`PREDICTED_LABEL`, `MAX_PROBABILITY`, and one probability column for each of the
eight classes. The command stops with a clear error if a required feature is
absent or if any row has fewer than 10 measured values among the 19 predictors.

Delete `quickstart_predictions.csv` after inspection if it is not needed; the
file is generated output and is not part of the frozen release.

## 5. Discover the full workflow

```powershell
python scripts/run_pipeline.py --list
```

Do not run `--stage all` for a quick check. The complete tuning, bootstrap, and
SHAP workflow is computationally intensive. See `docs/REPRODUCIBILITY.md` before
recomputing a stage.
