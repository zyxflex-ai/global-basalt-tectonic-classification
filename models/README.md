# Frozen model artifact

`xgboost_stable50_v4.ubj` is the XGBoost classifier fitted to the frozen
35,865-row development set only. The publication-group holdout was not used for
fitting or hyperparameter selection.

UBJSON is used because it is directly supported by XGBoost and is more compact
than the equivalent JSON model. `xgboost_stable50_v4.metadata.json` records the
exact ordered feature list,
class order, training parameters, package versions, input and model checksums,
measured-feature rule, verification metrics, and intended-use limitations.

Rebuild and verify the artifact with:

```powershell
python scripts/51_export_xgb_model_v4.py
```

Apply it to a compatible CSV file using
`examples/predict_with_frozen_model.py`. See `docs/QUICKSTART.md` and
`docs/USER_GUIDE.md` for tested commands and input/output definitions.
