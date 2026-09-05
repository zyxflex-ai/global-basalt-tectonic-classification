# AUTO-GENERATED V4 COPY. SOURCE: 12_baseline_models_innercv.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 12_baseline_models_innercv.py
#
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
#    Basalt_train_innercv_v4.csv
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================


import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from _project_paths import FINAL_DIR


warnings.filterwarnings("ignore")


# ============================================================
# Analysis step.
# ============================================================

# Analysis step.
input_file = (
    str(FINAL_DIR) + 
    '\\04_split\\Basalt_train_innercv_v4.csv'
)

# Analysis step.
result_dir = (
    str(FINAL_DIR / "05_results" / "baseline")
)

os.makedirs(
    result_dir,
    exist_ok=True
)

output_file = os.path.join(
    result_dir,
    "baseline_innercv_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 80)
print("读取数据")
print("=" * 80)

print("数据shape：", df.shape)

print(
    "INNER_FOLD：",
    sorted(df["INNER_FOLD"].unique())
)

print(
    "CV_GROUP_V4数量：",
    df["CV_GROUP_V4"].nunique()
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

int_to_class = {
    i: c
    for c, i in class_to_int.items()
}


df["Y"] = (
    df["LABEL"]
    .map(class_to_int)
)


if df["Y"].isna().sum() > 0:

    bad_labels = (
        df.loc[
            df["Y"].isna(),
            "LABEL"
        ]
        .unique()
    )

    raise RuntimeError(
        f"发现未知LABEL：{bad_labels}"
    )


df["Y"] = df["Y"].astype(int)


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


# Analysis step.
missing_columns = [
    c
    for c in candidate_features
    if c not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "以下候选特征不存在：\n"
        + str(missing_columns)
    )


# ============================================================
# Analysis step.
#
# Analysis step.
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


print("\nStable-50预期特征数：", len(stable50_features))


# ============================================================
# Analysis step.
#
# w_c = N / (K * n_c)
#
# Analysis step.
# ============================================================

def make_sample_weights(y):

    y = np.asarray(y)

    n_total = len(y)
    n_classes = len(classes)

    counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    class_weights = {}

    for class_id in range(n_classes):

        if class_id not in counts.index:

            raise RuntimeError(
                f"训练折缺少类别："
                f"{int_to_class[class_id]}"
            )

        n_class = counts[class_id]

        class_weights[class_id] = (
            n_total
            /
            (
                n_classes
                * n_class
            )
        )

    sample_weights = np.array(
        [
            class_weights[int(v)]
            for v in y
        ],
        dtype=float
    )

    return (
        sample_weights,
        class_weights
    )


# ============================================================
# Analysis step.
# ============================================================

def evaluate_model(
    model_name,
    fold,
    y_true,
    y_pred,
    n_train_raw,
    n_valid_raw,
    n_train_used,
    n_valid_used,
    n_features,
    seconds
):

    row = {
        "model": model_name,
        "fold": fold,

        "n_train_raw": n_train_raw,
        "n_valid_raw": n_valid_raw,

        "n_train_used": n_train_used,
        "n_valid_used": n_valid_used,

        "n_features": n_features,

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "seconds": seconds
    }


    # Analysis step.
    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(
            len(classes)
        ),
        average=None,
        zero_division=0
    )


    for class_name, score in zip(
        classes,
        class_f1
    ):

        row[
            f"f1_{class_name}"
        ] = score


    return row


# ============================================================
# Analysis step.
# ============================================================

results = []


for fold in range(5):

    print("\n\n")
    print("=" * 80)
    print(f"INNER FOLD {fold}")
    print("=" * 80)


    # ========================================================
    # Analysis step.
    #
    # Analysis step.
    # ========================================================

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


    print(
        "原始训练：",
        len(train_raw)
    )

    print(
        "原始验证：",
        len(valid_raw)
    )


    # ========================================================
    # Analysis step.
    # ========================================================

    overlap = (
        set(train_raw["CV_GROUP_V4"])
        &
        set(valid_raw["CV_GROUP_V4"])
    )

    print(
        "CV_GROUP_V4重叠：",
        len(overlap)
    )

    if len(overlap) != 0:

        raise RuntimeError(
            f"Fold {fold} 出现CV_GROUP_V4泄漏！"
        )


    # ========================================================
    # Analysis step.
    #
    # Analysis step.
    # Analysis step.
    # ========================================================

    train_feature_missing = (
        train_raw[
            candidate_features
        ]
        .isna()
        .mean()
    )


    selected_features = [
        c
        for c in candidate_features
        if train_feature_missing[c] <= 0.50
    ]


    print(
        "\n当前训练折 Stable-50 特征数：",
        len(selected_features)
    )

    print(
        "当前训练折 Stable-50 特征："
    )

    print(
        selected_features
    )


    # ========================================================
    # Analysis step.
    # ========================================================

    if (
        set(selected_features)
        !=
        set(stable50_features)
    ):

        extra = (
            set(selected_features)
            -
            set(stable50_features)
        )

        missing = (
            set(stable50_features)
            -
            set(selected_features)
        )

        print(
            "\n多出来的特征：",
            sorted(extra)
        )

        print(
            "少掉的特征：",
            sorted(missing)
        )

        raise RuntimeError(
            f"Fold {fold} 的Stable-50"
            "特征与预期19特征不一致。"
        )


    # Analysis step.
    features = (
        stable50_features
    )


    # ========================================================
    # Analysis step.
    #
    # Analysis step.
    # ========================================================

    train_raw[
        "N_STABLE50_FOLD"
    ] = (
        train_raw[
            features
        ]
        .notna()
        .sum(axis=1)
    )


    valid_raw[
        "N_STABLE50_FOLD"
    ] = (
        valid_raw[
            features
        ]
        .notna()
        .sum(axis=1)
    )


    train_df = (
        train_raw[
            train_raw[
                "N_STABLE50_FOLD"
            ] >= 10
        ]
        .copy()
    )


    valid_df = (
        valid_raw[
            valid_raw[
                "N_STABLE50_FOLD"
            ] >= 10
        ]
        .copy()
    )


    print("\n样品完整度规则：>=10/19")

    print(
        "筛选后训练：",
        len(train_df)
    )

    print(
        "训练保留率：",
        round(
            len(train_df)
            /
            len(train_raw),
            4
        )
    )

    print(
        "筛选后验证：",
        len(valid_df)
    )

    print(
        "验证保留率：",
        round(
            len(valid_df)
            /
            len(valid_raw),
            4
        )
    )


    # ========================================================
    # Analysis step.
    # ========================================================

    train_counts = (
        train_df[
            "LABEL"
        ]
        .value_counts()
        .reindex(
            classes,
            fill_value=0
        )
    )

    valid_counts = (
        valid_df[
            "LABEL"
        ]
        .value_counts()
        .reindex(
            classes,
            fill_value=0
        )
    )


    print("\n筛选后训练集类别：")
    print(train_counts)

    print("\n筛选后验证集类别：")
    print(valid_counts)


    if (train_counts == 0).any():

        raise RuntimeError(
            f"Fold {fold} 筛选后"
            "训练集缺失类别！"
        )


    if (valid_counts == 0).any():

        raise RuntimeError(
            f"Fold {fold} 筛选后"
            "验证集缺失类别！"
        )


    # ========================================================
    # Analysis step.
    # ========================================================

    X_train = (
        train_df[
            features
        ]
        .copy()
    )

    X_valid = (
        valid_df[
            features
        ]
        .copy()
    )


    y_train = (
        train_df[
            "Y"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_valid = (
        valid_df[
            "Y"
        ]
        .to_numpy(
            dtype=int
        )
    )


    # ========================================================
    # Analysis step.
    # ========================================================

    (
        sample_weight,
        class_weights
    ) = make_sample_weights(
        y_train
    )


    print("\n当前训练折类别权重：")

    for class_id in range(
        len(classes)
    ):

        print(
            f"{classes[class_id]:5s}: "
            f"{class_weights[class_id]:.4f}"
        )


    # ========================================================
    # A. XGBoost
    #
    # Analysis step.
    # Analysis step.
    # ========================================================

    print("\n" + "-" * 60)
    print("[XGBoost] 开始")
    print("-" * 60)


    start = time.time()


    xgb = XGBClassifier(

        objective="multi:softprob",
        num_class=len(classes),

        n_estimators=500,

        learning_rate=0.05,

        max_depth=6,

        min_child_weight=1,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=1.0,

        tree_method="hist",

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1
    )


    xgb.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )


    pred = (
        xgb
        .predict(
            X_valid
        )
        .astype(int)
    )


    seconds = (
        time.time()
        -
        start
    )


    row = evaluate_model(

        model_name="XGBoost",

        fold=fold,

        y_true=y_valid,

        y_pred=pred,

        n_train_raw=len(
            train_raw
        ),

        n_valid_raw=len(
            valid_raw
        ),

        n_train_used=len(
            train_df
        ),

        n_valid_used=len(
            valid_df
        ),

        n_features=len(
            features
        ),

        seconds=seconds
    )


    results.append(row)


    print(
        f"Macro-F1 = "
        f"{row['macro_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy = "
        f"{row['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy = "
        f"{row['accuracy']:.4f}"
    )

    print(
        f"耗时 = "
        f"{seconds:.1f} 秒"
    )


    # ========================================================
    # B. Random Forest
    #
    # Analysis step.
    # Analysis step.
    # ========================================================

    print("\n" + "-" * 60)
    print("[Random Forest] 开始")
    print("-" * 60)


    start = time.time()


    rf = Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "rf",

                RandomForestClassifier(

                    n_estimators=500,

                    max_features="sqrt",

                    min_samples_leaf=1,

                    random_state=42,

                    n_jobs=-1
                )
            )
        ]
    )


    rf.fit(
        X_train,
        y_train,
        rf__sample_weight=
            sample_weight
    )


    pred = (
        rf
        .predict(
            X_valid
        )
        .astype(int)
    )


    seconds = (
        time.time()
        -
        start
    )


    row = evaluate_model(

        model_name="RandomForest",

        fold=fold,

        y_true=y_valid,

        y_pred=pred,

        n_train_raw=len(
            train_raw
        ),

        n_valid_raw=len(
            valid_raw
        ),

        n_train_used=len(
            train_df
        ),

        n_valid_used=len(
            valid_df
        ),

        n_features=len(
            features
        ),

        seconds=seconds
    )


    results.append(row)


    print(
        f"Macro-F1 = "
        f"{row['macro_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy = "
        f"{row['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy = "
        f"{row['accuracy']:.4f}"
    )

    print(
        f"耗时 = "
        f"{seconds:.1f} 秒"
    )


    # ========================================================
    # C. RBF-SVM
    #
    # Median Imputer
    # -> StandardScaler
    # -> SVC
    #
    # Analysis step.
    # ========================================================

    print("\n" + "-" * 60)
    print("[RBF-SVM] 开始")
    print("-" * 60)


    start = time.time()


    svm = Pipeline(
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

                    C=10.0,

                    gamma="scale",

                    probability=False,

                    cache_size=4000
                )
            )
        ]
    )


    svm.fit(
        X_train,
        y_train,
        svm__sample_weight=
            sample_weight
    )


    pred = (
        svm
        .predict(
            X_valid
        )
        .astype(int)
    )


    seconds = (
        time.time()
        -
        start
    )


    row = evaluate_model(

        model_name="RBF-SVM",

        fold=fold,

        y_true=y_valid,

        y_pred=pred,

        n_train_raw=len(
            train_raw
        ),

        n_valid_raw=len(
            valid_raw
        ),

        n_train_used=len(
            train_df
        ),

        n_valid_used=len(
            valid_df
        ),

        n_features=len(
            features
        ),

        seconds=seconds
    )


    results.append(row)


    print(
        f"Macro-F1 = "
        f"{row['macro_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy = "
        f"{row['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy = "
        f"{row['accuracy']:.4f}"
    )

    print(
        f"耗时 = "
        f"{seconds:.1f} 秒"
    )


# ============================================================
# Analysis step.
# ============================================================

result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
# ============================================================

print("\n\n")
print("=" * 80)
print("5折总体结果")
print("=" * 80)


main_metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1"
]


model_order = [
    "XGBoost",
    "RandomForest",
    "RBF-SVM"
]


for model_name in model_order:

    sub = (
        result_df[
            result_df[
                "model"
            ] == model_name
        ]
        .copy()
    )


    print("\n")
    print(model_name)
    print("-" * 40)


    for metric in main_metrics:

        mean_value = (
            sub[
                metric
            ]
            .mean()
        )

        sd_value = (
            sub[
                metric
            ]
            .std(
                ddof=1
            )
        )


        print(
            f"{metric:20s}: "
            f"{mean_value:.4f} "
            f"± "
            f"{sd_value:.4f}"
        )


# ============================================================
# Analysis step.
# ============================================================

print("\n\n")
print("=" * 80)
print("各类别 F1：5折均值 ± SD")
print("=" * 80)


for model_name in model_order:

    sub = (
        result_df[
            result_df[
                "model"
            ] == model_name
        ]
        .copy()
    )


    print("\n")
    print(model_name)
    print("-" * 40)


    for class_name in classes:

        col = (
            f"f1_{class_name}"
        )

        mean_value = (
            sub[
                col
            ]
            .mean()
        )

        sd_value = (
            sub[
                col
            ]
            .std(
                ddof=1
            )
        )


        print(
            f"{class_name:5s}: "
            f"{mean_value:.4f} "
            f"± "
            f"{sd_value:.4f}"
        )


# ============================================================
# Analysis step.
# ============================================================

print("\n\n")
print("=" * 80)
print("逐折简表")
print("=" * 80)


simple_cols = [
    "model",
    "fold",
    "n_train_used",
    "n_valid_used",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "seconds"
]


print(
    result_df[
        simple_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("基线内部5折比较完成")
print("=" * 80)

print(
    "结果保存："
)

print(
    output_file
)

print(
    "\n注意：本脚本没有读取独立测试集。"
)