# AUTO-GENERATED V4 COPY. SOURCE: 28_xgb_full40_feature_sensitivity.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 28_xgb_full40_feature_sensitivity.py
#
# Analysis step.
# 1. Stable-50：19 features
# Analysis step.
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
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
    "xgb_full40_feature_sensitivity_innercv_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 90)
print("Stable-50 vs Full-40")
print("固定相同样品的特征方案敏感性")
print("=" * 90)

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
# 4. Full-40
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


schemes = {
    "Stable-50": stable50,
    "Full-40": full40
}


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
        c:
        n / (k * counts[c])
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

records = []


for fold in range(5):

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


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    overlap = (
        set(train_raw["CV_GROUP_V4"])
        &
        set(valid_raw["CV_GROUP_V4"])
    )

    if len(overlap) != 0:
        raise RuntimeError(
            f"Fold {fold} CV_GROUP_V4泄漏"
        )


    # --------------------------------------------------------
    # Analysis step.
    # Analysis step.
    #
    # Analysis step.
    # --------------------------------------------------------

    train_available = (
        train_raw[stable50]
        .notna()
        .sum(axis=1)
    )

    valid_available = (
        valid_raw[stable50]
        .notna()
        .sum(axis=1)
    )


    train_keep = (
        train_available >= 10
    )

    valid_keep = (
        valid_available >= 10
    )


    train_df = (
        train_raw[
            train_keep
        ]
        .copy()
    )

    valid_df = (
        valid_raw[
            valid_keep
        ]
        .copy()
    )


    print("\n")
    print("=" * 90)
    print(f"Fold {fold}")
    print("=" * 90)

    print(
        "Train retained:",
        len(train_df)
    )

    print(
        "Valid retained:",
        len(valid_df)
    )


    y_train = (
        train_df["Y"]
        .to_numpy(dtype=int)
    )

    y_valid = (
        valid_df["Y"]
        .to_numpy(dtype=int)
    )


    sample_weight = (
        make_sample_weights(
            y_train
        )
    )


    for scheme_name, features in (
        schemes.items()
    ):

        X_train = (
            train_df[features]
        )

        X_valid = (
            valid_df[features]
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        model = XGBClassifier(

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


        accuracy = accuracy_score(
            y_valid,
            pred
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_valid,
                pred
            )
        )

        macro_f1 = f1_score(
            y_valid,
            pred,
            labels=np.arange(8),
            average="macro",
            zero_division=0
        )


        records.append({

            "fold":
                fold,

            "scheme":
                scheme_name,

            "n_features":
                len(features),

            "train_n":
                len(train_df),

            "valid_n":
                len(valid_df),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1
        })


        print(
            f"{scheme_name:10s}: "
            f"Macro-F1={macro_f1:.4f}, "
            f"BalAcc={balanced_accuracy:.4f}, "
            f"Acc={accuracy:.4f}"
        )


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
        "scheme",
        sort=False
    )
    .agg(

        n_features=(
            "n_features",
            "first"
        ),

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


print("\n\n")
print("=" * 100)
print("固定相同样品：特征方案汇总")
print("=" * 100)

print(
    summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# Analysis step.
# ============================================================

pivot = (
    result
    .pivot(
        index="fold",
        columns="scheme",
        values=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1"
        ]
    )
)


print("\n")
print("=" * 100)
print("每折 Full-40 - Stable-50 差值")
print("=" * 100)


for fold in range(5):

    delta_acc = (
        pivot.loc[
            fold,
            ("accuracy", "Full-40")
        ]
        -
        pivot.loc[
            fold,
            ("accuracy", "Stable-50")
        ]
    )

    delta_bal = (
        pivot.loc[
            fold,
            ("balanced_accuracy", "Full-40")
        ]
        -
        pivot.loc[
            fold,
            ("balanced_accuracy", "Stable-50")
        ]
    )

    delta_f1 = (
        pivot.loc[
            fold,
            ("macro_f1", "Full-40")
        ]
        -
        pivot.loc[
            fold,
            ("macro_f1", "Stable-50")
        ]
    )


    print(
        f"Fold {fold}: "
        f"ΔMacro-F1={delta_f1:+.4f}, "
        f"ΔBalAcc={delta_bal:+.4f}, "
        f"ΔAcc={delta_acc:+.4f}"
    )


print("\n保存：")
print(output_file)

print(
    "\n说明：两个方案使用完全相同的训练和验证样品，"
    "因此差异主要反映特征集本身。"
)