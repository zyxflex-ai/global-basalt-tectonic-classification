# AUTO-GENERATED V4 COPY. SOURCE: 27_xgb_feature_threshold_sensitivity.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 27_xgb_feature_threshold_sensitivity.py
#
# Analysis step.
#
# Analysis step.
# Stable-40
# Analysis step.
# Stable-60
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================

import os
import math
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
    "xgb_feature_threshold_sensitivity_innercv_v4.csv"
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
print("Stable特征缺失率阈值敏感性")
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

schemes = {

    "Stable-40": 0.40,

    "Stable-50": 0.50,

    "Stable-60": 0.60
}


# ============================================================
# Analysis step.
#
# 10 / 19
# ============================================================

main_completeness_ratio = (
    10 / 19
)


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

    if len(counts) != k:
        raise RuntimeError(
            "当前训练折缺少类别"
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


for scheme_name, missing_threshold in (
    schemes.items()
):

    print("\n\n")
    print("=" * 100)
    print(
        f"{scheme_name} "
        f"(特征缺失率 <= "
        f"{missing_threshold:.0%})"
    )
    print("=" * 100)


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
                f"{scheme_name} "
                f"Fold {fold} "
                "存在CV_GROUP_V4泄漏"
            )


        # ----------------------------------------------------
        # Analysis step.
        #
        # Analysis step.
        # ----------------------------------------------------

        missing_rate = (
            train_raw[
                candidate_features
            ]
            .isna()
            .mean()
        )


        selected_features = [
            feature
            for feature in candidate_features
            if (
                missing_rate[feature]
                <=
                missing_threshold
            )
        ]


        n_features = len(
            selected_features
        )


        if n_features == 0:

            raise RuntimeError(
                "没有可用特征"
            )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        min_required = math.ceil(
            n_features
            *
            main_completeness_ratio
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        train_n = (
            train_raw[
                selected_features
            ]
            .notna()
            .sum(axis=1)
        )

        valid_n = (
            valid_raw[
                selected_features
            ]
            .notna()
            .sum(axis=1)
        )


        train_keep = (
            train_n
            >=
            min_required
        )

        valid_keep = (
            valid_n
            >=
            min_required
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


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        if (
            train_df["Y"].nunique()
            !=
            len(classes)
        ):

            raise RuntimeError(
                f"{scheme_name} "
                f"Fold {fold}: "
                "训练集筛选后缺类别"
            )


        if (
            valid_df["Y"].nunique()
            !=
            len(classes)
        ):

            raise RuntimeError(
                f"{scheme_name} "
                f"Fold {fold}: "
                "验证集筛选后缺类别"
            )


        X_train = (
            train_df[
                selected_features
            ]
        )

        X_valid = (
            valid_df[
                selected_features
            ]
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


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        records.append({

            "scheme":
                scheme_name,

            "missing_threshold":
                missing_threshold,

            "fold":
                fold,

            "n_features":
                n_features,

            "min_required_features":
                min_required,

            "completeness_ratio":
                min_required
                /
                n_features,

            "selected_features":
                "|".join(
                    selected_features
                ),

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
            f"\nFold {fold}"
        )

        print(
            f"特征数 = {n_features}"
        )

        print(
            f"最低实测 = "
            f"{min_required}/"
            f"{n_features}"
        )

        print(
            "特征："
        )

        print(
            ", ".join(
                selected_features
            )
        )

        print(
            f"Valid retention = "
            f"{len(valid_df)/len(valid_raw):.2%}"
        )

        print(
            f"Macro-F1 = "
            f"{macro_f1:.4f}, "
            f"BalAcc = "
            f"{balanced_accuracy:.4f}, "
            f"Acc = "
            f"{accuracy:.4f}"
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

        min_features=(
            "n_features",
            "min"
        ),

        max_features=(
            "n_features",
            "max"
        ),

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
print("=" * 110)
print("Stable特征阈值敏感性汇总")
print("=" * 110)


print(
    summary
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 110)
print("各折特征数与最低完整度要求")
print("=" * 110)


print(
    result[
        [
            "scheme",
            "fold",
            "n_features",
            "min_required_features",
            "valid_retention"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n保存：")
print(output_file)

print(
    "\n说明："
    "Stable-40/50/60均在每个训练折内部重新选择特征；"
    "没有使用独立测试集。"
)