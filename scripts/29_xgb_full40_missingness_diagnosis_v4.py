# AUTO-GENERATED V4 COPY. SOURCE: 29_xgb_full40_missingness_diagnosis.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 29_xgb_full40_missingness_diagnosis.py
#
# Analysis step.
#
# Analysis step.
# 1 Full40_native
# 2 Full40_imputed
# 3 Missingness_only
#
# Analysis step.
#
# Analysis step.
# ============================================================


import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
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
    r"\05_results\sensitivity"
)


os.makedirs(
    result_dir,
    exist_ok=True
)


output_file = os.path.join(
    result_dir,
    "xgb_full40_missingness_diagnosis_v4.csv"
)



# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)


print("="*90)
print("Full-40缺失模式来源诊断")
print("="*90)

print(
    "开发集:",
    df.shape
)



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
    c:i
    for i,c in enumerate(classes)
}


df["Y"] = (
    df["LABEL"]
    .map(class_to_int)
    .astype(int)
)



# ============================================================
# Stable-50
# Analysis step.
# ============================================================


stable50 = [

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
# Full40
# ============================================================


full40 = [

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


def make_sample_weights(y):

    y = np.asarray(y)

    counts = (
        pd.Series(y)
        .value_counts()
    )

    n = len(y)
    k = 8


    weights = {

        c:
        n/(k*counts[c])

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
# ============================================================


def build_model():

    return XGBClassifier(

        objective="multi:softprob",

        num_class=8,

        n_estimators=1400,

        learning_rate=0.02,

        max_depth=6,

        min_child_weight=1,

        subsample=0.8,

        colsample_bytree=0.7,

        reg_alpha=0.0,

        reg_lambda=1.0,


        tree_method="hist",

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1

    )



# ============================================================
# Analysis step.
# ============================================================


schemes = [

    "Full40_native",

    "Full40_imputed",

    "Missingness_only"

]



records = []



# ============================================================
# Analysis step.
# ============================================================


for fold in range(5):


    print("\n")
    print("="*90)
    print(
        f"Fold {fold}"
    )
    print("="*90)



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



    overlap = (
        set(train_raw["CV_GROUP_V4"])
        &
        set(valid_raw["CV_GROUP_V4"])
    )


    if len(overlap)>0:

        raise RuntimeError(
            "CV_GROUP_V4泄漏"
        )



    # --------------------------------------------------------
    # Analysis step.
    # Stable50 >=10/19
    # --------------------------------------------------------


    train_keep = (
        train_raw[stable50]
        .notna()
        .sum(axis=1)
        >=10
    )


    valid_keep = (
        valid_raw[stable50]
        .notna()
        .sum(axis=1)
        >=10
    )


    train_df = (
        train_raw[
            train_keep
        ]
    )


    valid_df = (
        valid_raw[
            valid_keep
        ]
    )


    print(
        "Train:",
        len(train_df)
    )

    print(
        "Valid:",
        len(valid_df)
    )



    y_train = (
        train_df["Y"]
        .values
    )


    y_valid = (
        valid_df["Y"]
        .values
    )


    sample_weight = (
        make_sample_weights(
            y_train
        )
    )



    for scheme in schemes:


        print(
            "\n运行:",
            scheme
        )



        # ====================================================
        # Full40 native
        # ====================================================

        if scheme=="Full40_native":


            X_train = (
                train_df[full40]
            )


            X_valid = (
                valid_df[full40]
            )



        # ====================================================
        # Full40 median
        # ====================================================

        elif scheme=="Full40_imputed":


            imputer = SimpleImputer(
                strategy="median"
            )


            X_train = (
                imputer
                .fit_transform(
                    train_df[full40]
                )
            )


            X_valid = (
                imputer
                .transform(
                    valid_df[full40]
                )
            )



        # ====================================================
        # Missing only
        # ====================================================

        elif scheme=="Missingness_only":


            X_train = (
                train_df[full40]
                .isna()
                .astype(int)
            )


            X_valid = (
                valid_df[full40]
                .isna()
                .astype(int)
            )



        model = build_model()


        model.fit(

            X_train,

            y_train,

            sample_weight=sample_weight

        )



        pred = (
            model.predict(
                X_valid
            )
            .astype(int)
        )



        acc = accuracy_score(
            y_valid,
            pred
        )


        bal = balanced_accuracy_score(
            y_valid,
            pred
        )


        macro = f1_score(

            y_valid,

            pred,

            average="macro",

            labels=np.arange(8),

            zero_division=0

        )



        print(

            f"{scheme}: "

            f"Acc={acc:.4f}, "

            f"BalAcc={bal:.4f}, "

            f"Macro-F1={macro:.4f}"

        )



        records.append({

            "fold":fold,

            "scheme":scheme,

            "accuracy":acc,

            "balanced_accuracy":bal,

            "macro_f1":macro,

            "n_train":len(train_df),

            "n_valid":len(valid_df)

        })



# ============================================================
# Analysis step.
# ============================================================


result = pd.DataFrame(
    records
)


result.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)



# ============================================================
# Analysis step.
# ============================================================


summary = (

    result

    .groupby(
        "scheme"
    )

    .agg(

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        sd_accuracy=(
            "accuracy",
            "std"
        ),


        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),

        sd_balanced_accuracy=(
            "balanced_accuracy",
            "std"
        ),


        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),

        sd_macro_f1=(
            "macro_f1",
            "std"
        )

    )

    .reset_index()

)



print("\n")
print("="*100)
print("最终汇总")
print("="*100)

print(
    summary.round(4)
    .to_string(index=False)
)



print("\n保存:")
print(output_file)


print(
    "\n完成"
)