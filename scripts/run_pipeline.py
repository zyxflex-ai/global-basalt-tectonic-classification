"""Run or list the frozen manuscript analysis stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = {
    "validation-bias": [
        "49_compare_random_vs_grouped_cv_v4.py",
        "50_plot_random_vs_grouped_cv_v4.py",
    ],
    "development": [
        "12_baseline_models_innercv_v4.py",
        "13_xgb_oof_diagnosis_v4.py",
        "14_xgb_tuning_innercv_v4.py",
        "15_xgb_tuned_oof_diagnosis_v4.py",
        "16_xgb_oof_calibration_v4.py",
        "17_rf_tuning_innercv_v4.py",
        "18_svm_tuning_innercv_v4.py",
    ],
    "holdout": [
        "19_final_independent_test_v4.py",
        "20_final_test_bootstrap_ci_v4.py",
        "21_paired_bootstrap_model_compare_v4.py",
    ],
    "model-export": [
        "51_export_xgb_model_v4.py",
    ],
    "interpretation": [
        "22_shap_global_independent_test_v4.py",
        "23_shap_direction_class_specific_v4.py",
        "24_shap_trueclass_measured_robustness_v4.py",
        "25_shap_dependence_key_features_v4.py",
    ],
    "sensitivity": [
        "26_xgb_completeness_sensitivity_v4.py",
        "27_xgb_feature_threshold_sensitivity_v4.py",
        "28_xgb_full40_feature_sensitivity_v4.py",
        "29_xgb_full40_missingness_diagnosis_v4.py",
        "30_final_full40_independent_test_v4.py",
    ],
    "figures": [
        "31_plot_final_confusion_matrix_vertical_labels_v4.py",
        "32_plot_shap_global_importance_v4.py",
        "33_plot_shap_8classes_2x4_v4.py",
        "34_plot_shap_direction_2x2_v4.py",
        "35_plot_shap_dependence_2x2_v4.py",
        "36_plot_ti_zr_y_discrimination_v4.py",
    ],
}


def ordered_scripts(stage: str) -> list[str]:
    if stage == "all":
        order = [
            "validation-bias",
            "development",
            "holdout",
            "model-export",
            "interpretation",
            "sensitivity",
            "figures",
        ]
        return [name for item in order for name in STAGES[item]]
    return STAGES[stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for stage, names in STAGES.items():
            print(f"[{stage}]")
            for name in names:
                print(f"  {name}")
        return
    if not args.stage:
        parser.error("provide --stage or --list")

    for name in ordered_scripts(args.stage):
        path = SCRIPT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
