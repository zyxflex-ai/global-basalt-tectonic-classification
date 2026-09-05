# AUTO-GENERATED V4 COPY. SOURCE: 19_final_independent_test.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 19_final_independent_test.py
#
# Analysis step.
#
# Analysis step.
# 1. Tuned XGBoost
# 2. Tuned Random Forest
# 3. Tuned RBF-SVM
#
# Analysis step.
# Analysis step.
# >=10/19
#
# Analysis step.
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
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
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

train_file = (
    str(FINAL_DIR) + 
    '\\03_model_data\\Basalt_train_stable50_main_v4.csv'
)

test_file = (
    str(FINAL_DIR) + 
    '\\03_model_data\\Basalt_test_stable50_main_v4.csv'
)

result_dir = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
)

os.makedirs(
    result_dir,
    exist_ok=True
)


metrics_file = os.path.join(
    result_dir,
    "final_independent_test_metrics_v4.csv"
)

predictions_file = os.path.join(
    result_dir,
    "final_independent_test_predictions_v4.csv"
)


# ============================================================
# Analysis step.
# ============================================================

train = pd.read_csv(
    train_file,
    encoding="utf-8-sig",
    low_memory=False
)

test = pd.read_csv(
    test_file,
    encoding="utf-8-sig",
    low_memory=False
)


print("=" * 90)
print("正式独立测试")
print("=" * 90)

print(
    "主训练集：",
    train.shape
)

print(
    "主独立测试集：",
    test.shape
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


train["Y"] = (
    train["LABEL"]
    .map(class_to_int)
)

test["Y"] = (
    test["LABEL"]
    .map(class_to_int)
)


if train["Y"].isna().any():
    raise RuntimeError(
        "训练集发现未知LABEL"
    )

if test["Y"].isna().any():
    raise RuntimeError(
        "测试集发现未知LABEL"
    )


train["Y"] = (
    train["Y"]
    .astype(int)
)

test["Y"] = (
    test["Y"]
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


print(
    "\n正式特征数：",
    len(features)
)


# ============================================================
# Analysis step.
#
# Analysis step.
# Analysis step.
# ============================================================

train_n = (
    train[features]
    .notna()
    .sum(axis=1)
)

test_n = (
    test[features]
    .notna()
    .sum(axis=1)
)


print(
    "训练集最低有效特征数：",
    train_n.min()
)

print(
    "测试集最低有效特征数：",
    test_n.min()
)


if (train_n < 10).any():
    raise RuntimeError(
        "主训练集中出现 <10/19 样品"
    )

if (test_n < 10).any():
    raise RuntimeError(
        "主测试集中出现 <10/19 样品"
    )


# ============================================================
# Analysis step.
# ============================================================

overlap = (
    set(train["CV_GROUP_V4"])
    &
    set(test["CV_GROUP_V4"])
)


print(
    "\n训练/测试CV_GROUP_V4重叠：",
    len(overlap)
)


if len(overlap) != 0:

    raise RuntimeError(
        "训练集和测试集存在CV_GROUP_V4重叠！"
    )


print(
    "训练CV_GROUP_V4：",
    train["CV_GROUP_V4"].nunique()
)

print(
    "测试CV_GROUP_V4：",
    test["CV_GROUP_V4"].nunique()
)


# ============================================================
# Analysis step.
# ============================================================

print("\n训练集类别：")

print(
    train["LABEL"]
    .value_counts()
    .reindex(
        classes
    )
)


print("\n独立测试集类别：")

print(
    test["LABEL"]
    .value_counts()
    .reindex(
        classes
    )
)


# ============================================================
# Analysis step.
# ============================================================

X_train = (
    train[features]
    .copy()
)

X_test = (
    test[features]
    .copy()
)

y_train = (
    train["Y"]
    .to_numpy(
        dtype=int
    )
)

y_test = (
    test["Y"]
    .to_numpy(
        dtype=int
    )
)


# ============================================================
# Analysis step.
#
# w_c = N / (K * n_c)
# ============================================================

def make_sample_weights(y):

    y = np.asarray(y)

    n = len(y)
    k = len(classes)

    counts = (
        pd.Series(y)
        .value_counts()
    )

    class_weights = {}

    for class_id in range(k):

        if class_id not in counts.index:

            raise RuntimeError(
                f"训练集缺少类别 "
                f"{classes[class_id]}"
            )

        class_weights[class_id] = (
            n
            /
            (
                k
                *
                counts[class_id]
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


sample_weight, class_weights = (
    make_sample_weights(
        y_train
    )
)


print("\n最终训练类别权重：")

for class_id in range(
    len(classes)
):

    print(
        f"{classes[class_id]:5s}: "
        f"{class_weights[class_id]:.4f}"
    )


# ============================================================
# Analysis step.
# ============================================================

metrics_rows = []


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    seconds
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_acc = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # overall
    metrics_rows.append({
        "model": model_name,
        "level": "overall",
        "class": "ALL",
        "precision": np.nan,
        "recall": np.nan,
        "f1": macro_f1,
        "support": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "seconds": seconds
    })


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(
                len(classes)
            ),
            zero_division=0
        )
    )


    for i, class_name in enumerate(
        classes
    ):

        metrics_rows.append({
            "model": model_name,
            "level": "class",
            "class": class_name,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
            "accuracy": np.nan,
            "balanced_accuracy":
                np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "seconds": np.nan
        })


    return {
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1":
            weighted_f1
    }


# ============================================================
# Analysis step.
# ============================================================

prediction_df = test[
    [
        "GROUP_ID",
        "CV_GROUP_V4",
        "CITATION",
        "SAMPLE NAME",
        "LABEL",
        "N_STABLE50"
    ]
].copy()


prediction_df = prediction_df.rename(
    columns={
        "LABEL":
            "TRUE_LABEL"
    }
)


# ============================================================
# A. Tuned XGBoost
#
# Config 14
#
# Analysis step.
# 1598, 1847, 1168, 989, 1395
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("1. Tuned XGBoost")
print("=" * 90)


start = time.time()


xgb = XGBClassifier(

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


xgb.fit(
    X_train,
    y_train,
    sample_weight=
        sample_weight
)


xgb_proba = (
    xgb.predict_proba(
        X_test
    )
)


xgb_pred = (
    np.argmax(
        xgb_proba,
        axis=1
    )
    .astype(int)
)


xgb_seconds = (
    time.time()
    -
    start
)


xgb_result = evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred,
    xgb_seconds
)


print(
    f"Accuracy          : "
    f"{xgb_result['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{xgb_result['balanced_accuracy']:.4f}"
)

print(
    f"Macro-F1          : "
    f"{xgb_result['macro_f1']:.4f}"
)

print(
    f"Weighted-F1       : "
    f"{xgb_result['weighted_f1']:.4f}"
)

print(
    f"耗时              : "
    f"{xgb_seconds:.1f} 秒"
)


print("\n各类别：")

print(
    classification_report(
        y_test,
        xgb_pred,
        labels=np.arange(8),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)


# ------------------------------------------------------------
# Analysis step.
# ------------------------------------------------------------

prediction_df[
    "XGB_PRED_LABEL"
] = [
    int_to_class[v]
    for v in xgb_pred
]


prediction_df[
    "XGB_CORRECT"
] = (
    prediction_df[
        "TRUE_LABEL"
    ]
    ==
    prediction_df[
        "XGB_PRED_LABEL"
    ]
)


for i, class_name in enumerate(
    classes
):

    prediction_df[
        f"XGB_PROB_{class_name}"
    ] = (
        xgb_proba[:, i]
    )


prediction_df[
    "XGB_MAX_PROB"
] = (
    np.max(
        xgb_proba,
        axis=1
    )
)


# ------------------------------------------------------------
# Analysis step.
# MAX_PROB < 0.70 = LOW CONFIDENCE
# ------------------------------------------------------------

prediction_df[
    "XGB_CONFIDENCE_FLAG"
] = np.where(
    prediction_df[
        "XGB_MAX_PROB"
    ] < 0.70,
    "LOW",
    "NORMAL"
)


# entropy
eps = 1e-15

prediction_df[
    "XGB_ENTROPY"
] = -np.sum(
    xgb_proba
    *
    np.log(
        np.clip(
            xgb_proba,
            eps,
            1.0
        )
    ),
    axis=1
)


# ============================================================
# B. Tuned Random Forest
#
# Config 4
#
# n_estimators = 300
# max_depth = None
# min_samples_split = 2
# min_samples_leaf = 2
# max_features = sqrt
# ============================================================

print("\n")
print("=" * 90)
print("2. Tuned Random Forest")
print("=" * 90)


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

                n_estimators=300,

                max_depth=None,

                min_samples_split=2,

                min_samples_leaf=2,

                max_features="sqrt",

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


rf_pred = (
    rf.predict(
        X_test
    )
    .astype(int)
)


rf_seconds = (
    time.time()
    -
    start
)


rf_result = evaluate_model(
    "RandomForest",
    y_test,
    rf_pred,
    rf_seconds
)


print(
    f"Accuracy          : "
    f"{rf_result['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{rf_result['balanced_accuracy']:.4f}"
)

print(
    f"Macro-F1          : "
    f"{rf_result['macro_f1']:.4f}"
)

print(
    f"Weighted-F1       : "
    f"{rf_result['weighted_f1']:.4f}"
)

print(
    f"耗时              : "
    f"{rf_seconds:.1f} 秒"
)


print("\n各类别：")

print(
    classification_report(
        y_test,
        rf_pred,
        labels=np.arange(8),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)


prediction_df[
    "RF_PRED_LABEL"
] = [
    int_to_class[v]
    for v in rf_pred
]


prediction_df[
    "RF_CORRECT"
] = (
    prediction_df[
        "TRUE_LABEL"
    ]
    ==
    prediction_df[
        "RF_PRED_LABEL"
    ]
)


# ============================================================
# C. Tuned RBF-SVM
#
# Config 8
#
# C = 30
# gamma = 0.3
#
# probability=False
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("3. Tuned RBF-SVM")
print("=" * 90)


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

                C=30.0,

                gamma=0.3,

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


svm_pred = (
    svm.predict(
        X_test
    )
    .astype(int)
)


svm_seconds = (
    time.time()
    -
    start
)


svm_result = evaluate_model(
    "RBF-SVM",
    y_test,
    svm_pred,
    svm_seconds
)


print(
    f"Accuracy          : "
    f"{svm_result['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{svm_result['balanced_accuracy']:.4f}"
)

print(
    f"Macro-F1          : "
    f"{svm_result['macro_f1']:.4f}"
)

print(
    f"Weighted-F1       : "
    f"{svm_result['weighted_f1']:.4f}"
)

print(
    f"耗时              : "
    f"{svm_seconds:.1f} 秒"
)


print("\n各类别：")

print(
    classification_report(
        y_test,
        svm_pred,
        labels=np.arange(8),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)


prediction_df[
    "SVM_PRED_LABEL"
] = [
    int_to_class[v]
    for v in svm_pred
]


prediction_df[
    "SVM_CORRECT"
] = (
    prediction_df[
        "TRUE_LABEL"
    ]
    ==
    prediction_df[
        "SVM_PRED_LABEL"
    ]
)


# ============================================================
# Analysis step.
# ============================================================

print("\n\n")
print("=" * 90)
print("三模型正式独立测试比较")
print("=" * 90)


comparison = pd.DataFrame({

    "XGBoost": {
        "Accuracy":
            xgb_result[
                "accuracy"
            ],

        "Balanced_Accuracy":
            xgb_result[
                "balanced_accuracy"
            ],

        "Macro_F1":
            xgb_result[
                "macro_f1"
            ],

        "Weighted_F1":
            xgb_result[
                "weighted_f1"
            ]
    },

    "RandomForest": {
        "Accuracy":
            rf_result[
                "accuracy"
            ],

        "Balanced_Accuracy":
            rf_result[
                "balanced_accuracy"
            ],

        "Macro_F1":
            rf_result[
                "macro_f1"
            ],

        "Weighted_F1":
            rf_result[
                "weighted_f1"
            ]
    },

    "RBF-SVM": {
        "Accuracy":
            svm_result[
                "accuracy"
            ],

        "Balanced_Accuracy":
            svm_result[
                "balanced_accuracy"
            ],

        "Macro_F1":
            svm_result[
                "macro_f1"
            ],

        "Weighted_F1":
            svm_result[
                "weighted_f1"
            ]
    }
}).T


print(
    comparison
    .round(4)
    .to_string()
)


# ============================================================
# Analysis step.
# ============================================================

cm = confusion_matrix(
    y_test,
    xgb_pred,
    labels=np.arange(8)
)


cm_df = pd.DataFrame(

    cm,

    index=[
        "TRUE_" + c
        for c in classes
    ],

    columns=[
        "PRED_" + c
        for c in classes
    ]
)


print("\n")
print("=" * 90)
print("XGBoost独立测试混淆矩阵：样品数")
print("=" * 90)


print(
    cm_df.to_string()
)


# ============================================================
# Analysis step.
# ============================================================

row_sum = (
    cm.sum(
        axis=1,
        keepdims=True
    )
)


cm_norm = np.divide(

    cm,

    row_sum,

    out=np.zeros_like(
        cm,
        dtype=float
    ),

    where=row_sum != 0
)


cm_norm_df = pd.DataFrame(

    cm_norm,

    index=[
        "TRUE_" + c
        for c in classes
    ],

    columns=[
        "PRED_" + c
        for c in classes
    ]
)


print("\n")
print("=" * 90)
print(
    "XGBoost独立测试混淆矩阵："
    "按真实类别归一化"
)
print("=" * 90)


print(
    cm_norm_df
    .round(4)
    .to_string()
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("XGBoost独立测试主要误判方向")
print("=" * 90)


for i, true_class in enumerate(
    classes
):

    row = cm[i].copy()

    row[i] = 0

    order = (
        np.argsort(
            row
        )[::-1]
    )

    total = (
        cm[i].sum()
    )


    print(
        f"\n{true_class}"
    )


    shown = 0


    for j in order:

        if row[j] == 0:
            continue


        count = row[j]

        rate = (
            count
            /
            total
        )


        print(
            f"  → {classes[j]:5s}: "
            f"{count:5d} "
            f"({rate:.2%})"
        )


        shown += 1


        if shown == 3:
            break


# ============================================================
# Analysis step.
#
# Analysis step.
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("XGBoost独立测试低置信度诊断")
print("=" * 90)


max_prob = (
    prediction_df[
        "XGB_MAX_PROB"
    ]
    .to_numpy()
)

correct = (
    prediction_df[
        "XGB_CORRECT"
    ]
    .to_numpy()
)


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]:

    high = (
        max_prob >= threshold
    )


    if high.sum() > 0:

        high_acc = (
            correct[
                high
            ]
            .mean()
        )

    else:

        high_acc = np.nan


    print(
        f"MAX_PROB >= "
        f"{threshold:.2f}: "
        f"n={high.sum()}, "
        f"coverage="
        f"{high.mean():.2%}, "
        f"Accuracy="
        f"{high_acc:.4f}"
    )


low = (
    max_prob < 0.70
)


print(
    "\n预先冻结的低置信度规则："
)

print(
    "MAX_PROB < 0.70"
)

print(
    "低置信度样品：",
    low.sum()
)

print(
    "低置信度比例：",
    f"{low.mean():.2%}"
)

print(
    "低置信度Accuracy：",
    round(
        correct[
            low
        ].mean(),
        4
    )
)


# ============================================================
# Analysis step.
# ============================================================

metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_df.to_csv(
    metrics_file,
    index=False,
    encoding="utf-8-sig"
)


prediction_df.to_csv(
    predictions_file,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 90)
print("正式独立测试完成")
print("=" * 90)


print(
    "\n指标保存："
)

print(
    metrics_file
)


print(
    "\n逐样品预测保存："
)

print(
    predictions_file
)


print(
    "\n重要：独立测试结果已经正式查看。"
)

print(
    "从现在开始，不再根据测试集结果修改"
    "特征、阈值、参数或模型选择。"
)