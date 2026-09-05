# AUTO-GENERATED V4 COPY. SOURCE: 18_svm_tuning_innercv.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 18_svm_tuning_innercv.py
#
# RBF-SVM
# Stable-50 + >=10/19
# Analysis step.
#
# Analysis step.
# Analysis step.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from _project_paths import FINAL_DIR

warnings.filterwarnings("ignore")


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    '\\04_split\\Basalt_train_innercv_v4.csv'
)

result_dir = (
    str(FINAL_DIR) + 
    r"\05_results\tuning"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

output_file = os.path.join(
    result_dir,
    "svm_tuning_innercv_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)

print("开发集：", df.shape)


# ============================================================
# Analysis step.
# ============================================================

classes = [
    "CAB",
    "IAB",
    "IOAB",
    "BABB",
    "MORB",
    "OIB",
    "OPB",
    "CFB"
]

class_to_int = {
    c: i
    for i, c in enumerate(classes)
}

df["Y"] = (
    df["LABEL"]
    .map(class_to_int)
    .astype(int)
)


# ============================================================
# Analysis step.
# ============================================================

candidate_features = [
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

    "SC(PPM)",
    "V(PPM)",
    "CR(PPM)",
    "CO(PPM)",
    "NI(PPM)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "BA(PPM)",

    "LA(PPM)",
    "CE(PPM)",
    "PR(PPM)",
    "ND(PPM)",
    "SM(PPM)",
    "EU(PPM)",
    "GD(PPM)",
    "TB(PPM)",
    "DY(PPM)",
    "HO(PPM)",
    "ER(PPM)",
    "TM(PPM)",
    "YB(PPM)",
    "LU(PPM)",

    "HF(PPM)",
    "TA(PPM)",
    "PB(PPM)",
    "TH(PPM)",
    "U(PPM)"
]


# ============================================================
# Analysis step.
# ============================================================

stable50_features = [
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
    "BA(PPM)"
]


# ============================================================
# Analysis step.
# ============================================================

def make_sample_weights(y):

    y = np.asarray(y)

    n = len(y)
    k = len(classes)

    counts = (
        pd.Series(y)
        .value_counts()
    )

    weights = {
        c: n / (k * counts[c])
        for c in range(k)
    }

    return np.array(
        [
            weights[v]
            for v in y
        ],
        dtype=float
    )


# ============================================================
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================

param_grid = {

    "C": [
        0.5,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0
    ],

    "gamma": [
        "scale",
        0.01,
        0.03,
        0.1,
        0.3
    ]
}


parameter_sets = list(
    ParameterSampler(
        param_grid,
        n_iter=12,
        random_state=42
    )
)


print(
    "参数组合数：",
    len(parameter_sets)
)


# ============================================================
# Analysis step.
# ============================================================

all_results = []


for config_id, params in enumerate(
    parameter_sets,
    start=1
):

    print("\n")
    print("=" * 80)
    print(
        f"SVM PARAMETER SET "
        f"{config_id}/"
        f"{len(parameter_sets)}"
    )
    print("=" * 80)

    print(params)


    fold_results = []


    for fold in range(5):

        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        train_raw = (
            df[
                df["INNER_FOLD"] != fold
            ]
            .copy()
        )

        valid_raw = (
            df[
                df["INNER_FOLD"] == fold
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        overlap = (
            set(train_raw["CV_GROUP_V4"])
            &
            set(valid_raw["CV_GROUP_V4"])
        )

        if len(overlap) != 0:

            raise RuntimeError(
                f"Fold {fold} CV_GROUP_V4泄漏"
            )


        # ----------------------------------------------------
        # Stable-50
        # Analysis step.
        # ----------------------------------------------------

        missing_rate = (
            train_raw[
                candidate_features
            ]
            .isna()
            .mean()
        )

        selected = [
            c
            for c in candidate_features
            if missing_rate[c] <= 0.50
        ]


        if set(selected) != set(
            stable50_features
        ):

            raise RuntimeError(
                f"Fold {fold} "
                "Stable-50特征异常"
            )


        features = stable50_features


        # ----------------------------------------------------
        # >=10/19
        # ----------------------------------------------------

        train_raw["N_FEATURE"] = (
            train_raw[features]
            .notna()
            .sum(axis=1)
        )

        valid_raw["N_FEATURE"] = (
            valid_raw[features]
            .notna()
            .sum(axis=1)
        )


        train_df = (
            train_raw[
                train_raw["N_FEATURE"] >= 10
            ]
            .copy()
        )

        valid_df = (
            valid_raw[
                valid_raw["N_FEATURE"] >= 10
            ]
            .copy()
        )


        X_train = (
            train_df[
                features
            ]
        )

        X_valid = (
            valid_df[
                features
            ]
        )

        y_train = (
            train_df["Y"]
            .to_numpy()
        )

        y_valid = (
            valid_df["Y"]
            .to_numpy()
        )


        sample_weight = (
            make_sample_weights(
                y_train
            )
        )


        # ----------------------------------------------------
        # RBF-SVM
        #
        # Median imputation
        # -> StandardScaler
        # -> SVC
        #
        # Analysis step.
        # ----------------------------------------------------

        model = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "svm",
                    SVC(
                        kernel="rbf",

                        C=params["C"],

                        gamma=params["gamma"],

                        probability=False,

                        cache_size=4000
                    )
                )
            ]
        )


        model.fit(
            X_train,
            y_train,
            svm__sample_weight=
                sample_weight
        )


        pred = (
            model
            .predict(
                X_valid
            )
            .astype(int)
        )


        macro_f1 = f1_score(
            y_valid,
            pred,
            average="macro",
            zero_division=0
        )

        bal_acc = (
            balanced_accuracy_score(
                y_valid,
                pred
            )
        )

        acc = accuracy_score(
            y_valid,
            pred
        )


        fold_results.append({

            "fold": fold,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                bal_acc,

            "accuracy":
                acc
        })


        print(
            f"Fold {fold}: "
            f"Macro-F1="
            f"{macro_f1:.4f}, "
            f"BalAcc="
            f"{bal_acc:.4f}"
        )


    # ========================================================
    # Analysis step.
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )


    result = {

        "config_id":
            config_id,

        "C":
            params["C"],

        "gamma":
            params["gamma"],

        "mean_macro_f1":
            fold_df[
                "macro_f1"
            ].mean(),

        "sd_macro_f1":
            fold_df[
                "macro_f1"
            ].std(ddof=1),

        "mean_balanced_accuracy":
            fold_df[
                "balanced_accuracy"
            ].mean(),

        "sd_balanced_accuracy":
            fold_df[
                "balanced_accuracy"
            ].std(ddof=1),

        "mean_accuracy":
            fold_df[
                "accuracy"
            ].mean()
    }


    all_results.append(
        result
    )


    print(
        "\nMean Macro-F1 =",
        round(
            result[
                "mean_macro_f1"
            ],
            4
        ),
        "±",
        round(
            result[
                "sd_macro_f1"
            ],
            4
        )
    )


    # Analysis step.
    pd.DataFrame(
        all_results
    ).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# Analysis step.
# ============================================================

result_df = pd.DataFrame(
    all_results
)


result_df = (
    result_df
    .sort_values(

        by=[
            "mean_macro_f1",
            "mean_balanced_accuracy",
            "sd_macro_f1"
        ],

        ascending=[
            False,
            False,
            True
        ]
    )
)


result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. TOP 10
# ============================================================

print("\n\n")
print("=" * 100)
print("SVM 调参完成：TOP 10")
print("=" * 100)


show_cols = [

    "config_id",

    "C",

    "gamma",

    "mean_macro_f1",

    "sd_macro_f1",

    "mean_balanced_accuracy",

    "mean_accuracy"
]


print(
    result_df[
        show_cols
    ]
    .head(10)
    .round(4)
    .to_string(
        index=False
    )
)


print("\n保存：")
print(output_file)

print(
    "\n独立测试集未读取。"
)