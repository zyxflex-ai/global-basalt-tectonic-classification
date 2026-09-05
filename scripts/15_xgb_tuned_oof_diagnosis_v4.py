# AUTO-GENERATED V4 COPY. SOURCE: 15_xgb_tuned_oof_diagnosis.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 15_xgb_tuned_oof_diagnosis.py
#
# Analysis step.
# Stable-50 + >=10/19
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================


import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    '\\04_split\\Basalt_train_innercv_v4.csv'
)

result_dir = (
    str(FINAL_DIR) + 
    r"\05_results\diagnosis"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

output_file = os.path.join(
    result_dir,
    "xgb_tuned_oof_predictions_v4.csv"
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

print("开发集：", df.shape)
print("CV_GROUP_V4数量：", df["CV_GROUP_V4"].nunique())


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

    sample_weights = np.array(
        [
            weights[v]
            for v in y
        ],
        dtype=float
    )

    return sample_weights


# ============================================================
# Analysis step.
# ============================================================

tuned_params = {

    "learning_rate": 0.02,

    "max_depth": 6,

    "min_child_weight": 1,

    "subsample": 0.8,

    "colsample_bytree": 0.7,

    "reg_alpha": 0.0,

    "reg_lambda": 1.0
}


print("\nConfig 14：")
print(tuned_params)


# ============================================================
# 7. OOF
# ============================================================

oof_list = []
best_iterations = []


for fold in range(5):

    print("\n")
    print("=" * 80)
    print(f"TUNED OOF FOLD {fold}")
    print("=" * 80)


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

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

    print(
        "CV_GROUP_V4重叠：",
        len(overlap)
    )

    if len(overlap) != 0:

        raise RuntimeError(
            f"Fold {fold} 出现CV_GROUP_V4泄漏"
        )


    # --------------------------------------------------------
    # Stable-50
    # Analysis step.
    # --------------------------------------------------------

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

    print(
        "Stable-50特征数：",
        len(selected)
    )

    if set(selected) != set(
        stable50_features
    ):

        extra = (
            set(selected)
            -
            set(stable50_features)
        )

        missing = (
            set(stable50_features)
            -
            set(selected)
        )

        print(
            "多出的特征：",
            sorted(extra)
        )

        print(
            "缺少的特征：",
            sorted(missing)
        )

        raise RuntimeError(
            f"Fold {fold} Stable-50特征异常"
        )


    features = stable50_features


    # --------------------------------------------------------
    # >=10/19
    # --------------------------------------------------------

    train_raw["N_STABLE50"] = (
        train_raw[features]
        .notna()
        .sum(axis=1)
    )

    valid_raw["N_STABLE50"] = (
        valid_raw[features]
        .notna()
        .sum(axis=1)
    )


    train_df = (
        train_raw[
            train_raw["N_STABLE50"] >= 10
        ]
        .copy()
    )

    valid_df = (
        valid_raw[
            valid_raw["N_STABLE50"] >= 10
        ]
        .copy()
    )


    print(
        "训练：",
        len(train_df),
        "| 验证：",
        len(valid_df)
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    X_train = train_df[
        features
    ]

    X_valid = valid_df[
        features
    ]

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


    # --------------------------------------------------------
    # Tuned XGBoost
    # --------------------------------------------------------

    model = XGBClassifier(

        objective="multi:softprob",
        num_class=8,

        n_estimators=3000,

        learning_rate=
            tuned_params[
                "learning_rate"
            ],

        max_depth=
            tuned_params[
                "max_depth"
            ],

        min_child_weight=
            tuned_params[
                "min_child_weight"
            ],

        subsample=
            tuned_params[
                "subsample"
            ],

        colsample_bytree=
            tuned_params[
                "colsample_bytree"
            ],

        reg_alpha=
            tuned_params[
                "reg_alpha"
            ],

        reg_lambda=
            tuned_params[
                "reg_lambda"
            ],

        tree_method="hist",

        eval_metric="mlogloss",

        early_stopping_rounds=100,

        random_state=42,

        n_jobs=-1
    )


    model.fit(

        X_train,
        y_train,

        sample_weight=
            sample_weight,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        verbose=False
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    best_iteration = (
        model.best_iteration
    )

    best_iterations.append(
        best_iteration
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    proba = (
        model
        .predict_proba(
            X_valid
        )
    )

    pred = (
        np.argmax(
            proba,
            axis=1
        )
        .astype(int)
    )


    fold_macro = f1_score(
        y_valid,
        pred,
        average="macro",
        zero_division=0
    )


    fold_bal = (
        balanced_accuracy_score(
            y_valid,
            pred
        )
    )


    print(
        "Best iteration：",
        best_iteration
    )

    print(
        "Macro-F1：",
        round(
            fold_macro,
            4
        )
    )

    print(
        "Balanced Accuracy：",
        round(
            fold_bal,
            4
        )
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    keep_cols = [
        "GROUP_ID",
        "CV_GROUP_V4",
        "CITATION",
        "SAMPLE NAME",
        "LABEL",
        "INNER_FOLD",
        "N_STABLE50"
    ]

    temp = (
        valid_df[
            keep_cols
        ]
        .copy()
    )


    temp["TRUE_ID"] = y_valid

    temp["PRED_ID"] = pred

    temp["PRED_LABEL"] = [
        int_to_class[v]
        for v in pred
    ]

    temp["CORRECT"] = (
        temp["LABEL"]
        ==
        temp["PRED_LABEL"]
    )


    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    for i, c in enumerate(
        classes
    ):

        temp[
            f"PROB_{c}"
        ] = proba[:, i]


    # Analysis step.
    temp["MAX_PROB"] = (
        np.max(
            proba,
            axis=1
        )
    )


    # --------------------------------------------------------
    # Analysis step.
    # Analysis step.
    # --------------------------------------------------------

    eps = 1e-15

    entropy = -np.sum(
        proba
        *
        np.log(
            np.clip(
                proba,
                eps,
                1.0
            )
        ),
        axis=1
    )

    temp["ENTROPY"] = entropy


    # Analysis step.
    temp[
        "BEST_ITERATION"
    ] = best_iteration


    oof_list.append(
        temp
    )


# ============================================================
# Analysis step.
# ============================================================

oof = pd.concat(
    oof_list,
    ignore_index=True
)


print("\n")
print("=" * 80)
print("OOF覆盖")
print("=" * 80)


print(
    "OOF样品：",
    len(oof)
)

print(
    "GROUP_ID重复：",
    oof[
        "GROUP_ID"
    ]
    .duplicated()
    .sum()
)

print(
    "原始开发集：",
    len(df)
)

print(
    "<10/19排除：",
    len(df) - len(oof)
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("5折 Best iteration")
print("=" * 80)


print(
    best_iterations
)

print(
    "平均：",
    round(
        np.mean(
            best_iterations
        ),
        1
    )
)

print(
    "中位数：",
    round(
        np.median(
            best_iterations
        ),
        1
    )
)


# ============================================================
# Analysis step.
# ============================================================

y_true = (
    oof["TRUE_ID"]
    .to_numpy()
)

y_pred = (
    oof["PRED_ID"]
    .to_numpy()
)


print("\n")
print("=" * 80)
print("调参后 OOF 总体指标")
print("=" * 80)


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


print(
    "Accuracy：",
    round(
        accuracy,
        4
    )
)

print(
    "Balanced Accuracy：",
    round(
        balanced_acc,
        4
    )
)

print(
    "Macro-F1：",
    round(
        macro_f1,
        4
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("调参后各类别 Precision / Recall / F1")
print("=" * 80)


print(
    classification_report(

        y_true,
        y_pred,

        labels=np.arange(8),

        target_names=classes,

        digits=4,

        zero_division=0
    )
)


# ============================================================
# Analysis step.
# ============================================================

cm = confusion_matrix(

    y_true,
    y_pred,

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
print("=" * 80)
print("调参后 OOF 混淆矩阵：样品数")
print("=" * 80)


print(
    cm_df.to_string()
)


# ============================================================
# Analysis step.
# ============================================================

row_sum = cm.sum(
    axis=1,
    keepdims=True
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
print("=" * 80)
print("调参后 OOF 混淆矩阵：按真实类别归一化")
print("=" * 80)


print(
    cm_norm_df
    .round(4)
    .to_string()
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("调参后每个真实类别最主要误判方向")
print("=" * 80)


for i, true_class in enumerate(
    classes
):

    row = cm[i].copy()

    # Analysis step.
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


        count = (
            row[j]
        )

        rate = (
            count
            /
            total
        )


        print(
            f"  → "
            f"{classes[j]:5s}: "
            f"{count:5d} "
            f"({rate:.2%})"
        )


        shown += 1


        if shown == 3:

            break


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("预测置信度诊断")
print("=" * 80)


print(
    "全部样品平均MAX_PROB：",
    round(
        oof["MAX_PROB"]
        .mean(),
        4
    )
)


print(
    "预测正确样品平均MAX_PROB：",
    round(
        oof.loc[
            oof["CORRECT"],
            "MAX_PROB"
        ]
        .mean(),
        4
    )
)


print(
    "预测错误样品平均MAX_PROB：",
    round(
        oof.loc[
            ~oof["CORRECT"],
            "MAX_PROB"
        ]
        .mean(),
        4
    )
)


# ============================================================
# Analysis step.
# ============================================================

oof.to_csv(

    output_file,

    index=False,

    encoding="utf-8-sig"
)


print("\n")
print("=" * 80)
print("调参后 OOF 诊断完成")
print("=" * 80)


print(
    "保存：",
    output_file
)

print(
    "\n独立测试集未读取。"
)