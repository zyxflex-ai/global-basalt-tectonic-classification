# AUTO-GENERATED V4 COPY. SOURCE: 26_xgb_completeness_sensitivity.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 26_xgb_completeness_sensitivity.py
#
# Analysis step.
#
# Analysis step.
# Analysis step.
# >=8/19
# Analysis step.
# >=12/19
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# Config14 + n_estimators=1400
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
    "xgb_completeness_sensitivity_innercv_v4.csv"
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
print("样品完整度敏感性分析")
print("=" * 90)

print(
    "开发集：",
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

features = [
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

thresholds = [
    8,
    10,
    12
]


# ============================================================
# Analysis step.
# ============================================================

df["N_STABLE50_TEMP"] = (
    df[features]
    .notna()
    .sum(axis=1)
)


print("\n")
print("=" * 90)
print("开发集总体保留情况")
print("=" * 90)


for threshold in thresholds:

    keep = (
        df["N_STABLE50_TEMP"]
        >= threshold
    )

    print(
        f"\n>= {threshold}/19"
    )

    print(
        f"保留样品："
        f"{keep.sum()} / {len(df)} "
        f"({keep.mean():.2%})"
    )


    for class_name in classes:

        class_mask = (
            df["LABEL"]
            == class_name
        )

        class_keep = (
            keep
            &
            class_mask
        )

        retention = (
            class_keep.sum()
            /
            class_mask.sum()
        )

        print(
            f"  {class_name:5s}: "
            f"{class_keep.sum():5d} / "
            f"{class_mask.sum():5d} "
            f"({retention:.2%})"
        )


# ============================================================
# Analysis step.
# ============================================================

records = []


for threshold in thresholds:

    print("\n\n")
    print("=" * 90)
    print(
        f"完整度阈值 >= {threshold}/19"
    )
    print("=" * 90)


    for fold in range(5):

        train_raw = (
            df[
                df["INNER_FOLD"]
                != fold
            ]
            .copy()
        )

        valid_raw = (
            df[
                df["INNER_FOLD"]
                == fold
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
                f"Fold {fold} "
                "发现CV_GROUP_V4泄漏"
            )


        # ----------------------------------------------------
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


        if set(selected) != set(features):

            raise RuntimeError(
                f"Fold {fold} "
                "Stable-50特征不一致"
            )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        train_n = (
            train_raw[features]
            .notna()
            .sum(axis=1)
        )

        valid_n = (
            valid_raw[features]
            .notna()
            .sum(axis=1)
        )


        train_keep = (
            train_n >= threshold
        )

        valid_keep = (
            valid_n >= threshold
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


        X_train = (
            train_df[features]
        )

        X_valid = (
            valid_df[features]
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


        # ----------------------------------------------------
        # Analysis step.
        #
        # Analysis step.
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
            sample_weight=
                sample_weight
        )


        pred = (
            model.predict(
                X_valid
            )
            .astype(int)
        )


        accuracy = (
            accuracy_score(
                y_valid,
                pred
            )
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_valid,
                pred
            )
        )

        macro_f1 = (
            f1_score(
                y_valid,
                pred,
                labels=np.arange(8),
                average="macro",
                zero_division=0
            )
        )


        records.append({

            "threshold":
                threshold,

            "fold":
                fold,

            "train_raw_n":
                len(train_raw),

            "train_retained_n":
                len(train_df),

            "train_retention":
                len(train_df)
                /
                len(train_raw),

            "valid_raw_n":
                len(valid_raw),

            "valid_retained_n":
                len(valid_df),

            "valid_retention":
                len(valid_df)
                /
                len(valid_raw),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1
        })


        print(
            f"Fold {fold}: "
            f"Valid n={len(valid_df)}, "
            f"Retention="
            f"{len(valid_df)/len(valid_raw):.2%}, "
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
        "threshold"
    )
    .agg(

        mean_valid_retention=(
            "valid_retention",
            "mean"
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
print("完整度敏感性汇总")
print("=" * 100)


print(
    summary
    .round(4)
    .to_string(
        index=False
    )
)


print("\n保存：")
print(output_file)

print(
    "\n说明："
    "该分析仅用于检验完整度阈值稳健性，"
    "不会依据结果重新选择主模型阈值。"
)