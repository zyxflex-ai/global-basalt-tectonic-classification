# AUTO-GENERATED V4 COPY. SOURCE: 22_shap_global_independent_test.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 22_shap_global_independent_test.py
#
# Frozen Tuned XGBoost
# Independent-test SHAP global + class-specific importance
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap

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
    r"\05_results\shap"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

importance_file = os.path.join(
    result_dir,
    "shap_importance_independent_test_v4.csv"
)

figure_file = os.path.join(
    result_dir,
    "shap_global_importance_v4.png"
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

print("=" * 80)
print("SHAP：独立测试集")
print("=" * 80)

print("训练集：", train.shape)
print("测试集：", test.shape)


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

train["Y"] = (
    train["LABEL"]
    .map(class_to_int)
    .astype(int)
)

test["Y"] = (
    test["LABEL"]
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


X_train = train[features].copy()
X_test = test[features].copy()

y_train = train["Y"].to_numpy(dtype=int)


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


sample_weight = make_sample_weights(
    y_train
)


# ============================================================
# Analysis step.
# ============================================================

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


print("\n训练最终冻结模型...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight
)

print("训练完成。")


# ============================================================
# 6. TreeSHAP
# ============================================================

print("\n计算独立测试集SHAP...")

explainer = shap.TreeExplainer(
    model
)

raw_shap = explainer.shap_values(
    X_test
)


# ============================================================
# Analysis step.
#
# Analysis step.
# samples × features × classes
# ============================================================

if isinstance(
    raw_shap,
    list
):

    # Analysis step.
    # list[class] -> samples × features
    shap_values = np.stack(
        raw_shap,
        axis=-1
    )

else:

    arr = np.asarray(
        raw_shap
    )

    if arr.ndim != 3:

        raise RuntimeError(
            "无法识别SHAP数组形状："
            + str(arr.shape)
        )


    # Analysis step.
    # samples × features × classes
    if (
        arr.shape[0] == len(X_test)
        and
        arr.shape[1] == len(features)
        and
        arr.shape[2] == len(classes)
    ):

        shap_values = arr


    # samples × classes × features
    elif (
        arr.shape[0] == len(X_test)
        and
        arr.shape[1] == len(classes)
        and
        arr.shape[2] == len(features)
    ):

        shap_values = np.transpose(
            arr,
            (0, 2, 1)
        )


    # classes × samples × features
    elif (
        arr.shape[0] == len(classes)
        and
        arr.shape[1] == len(X_test)
        and
        arr.shape[2] == len(features)
    ):

        shap_values = np.transpose(
            arr,
            (1, 2, 0)
        )


    else:

        raise RuntimeError(
            "无法识别SHAP数组形状："
            + str(arr.shape)
        )


print(
    "统一后的SHAP shape：",
    shap_values.shape
)

print(
    "应为：",
    (
        len(X_test),
        len(features),
        len(classes)
    )
)


# ============================================================
# Analysis step.
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
# ============================================================

global_importance = (
    np.abs(
        shap_values
    )
    .mean(
        axis=(0, 2)
    )
)


importance_df = pd.DataFrame({

    "feature":
        features,

    "GLOBAL_MEAN_ABS_SHAP":
        global_importance
})


# ============================================================
# Analysis step.
#
# Analysis step.
# ============================================================

for class_id, class_name in enumerate(
    classes
):

    importance_df[
        f"SHAP_{class_name}"
    ] = (
        np.abs(
            shap_values[
                :,
                :,
                class_id
            ]
        )
        .mean(axis=0)
    )


# ============================================================
# Analysis step.
# ============================================================

importance_df = (
    importance_df
    .sort_values(
        "GLOBAL_MEAN_ABS_SHAP",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


importance_df[
    "GLOBAL_RANK"
] = (
    np.arange(
        1,
        len(importance_df) + 1
    )
)


print("\n")
print("=" * 80)
print("SHAP 全局重要性排名")
print("=" * 80)


print(
    importance_df[
        [
            "GLOBAL_RANK",
            "feature",
            "GLOBAL_MEAN_ABS_SHAP"
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("各类别 SHAP Top 5")
print("=" * 80)


for class_name in classes:

    col = (
        f"SHAP_{class_name}"
    )

    top5 = (
        importance_df[
            [
                "feature",
                col
            ]
        ]
        .sort_values(
            col,
            ascending=False
        )
        .head(5)
    )


    print(
        f"\n{class_name}"
    )

    print(
        top5
        .round(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# Analysis step.
# ============================================================

importance_df.to_csv(
    importance_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
# ============================================================

plot_df = (
    importance_df
    .head(15)
    .sort_values(
        "GLOBAL_MEAN_ABS_SHAP"
    )
)


plt.figure(
    figsize=(8, 7)
)

plt.barh(
    plot_df["feature"],
    plot_df[
        "GLOBAL_MEAN_ABS_SHAP"
    ]
)

plt.xlabel(
    "Mean |SHAP value|"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Global SHAP Importance\nIndependent Test Set"
)

plt.tight_layout()

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 80)
print("SHAP全局分析完成")
print("=" * 80)

print(
    "重要性表："
)

print(
    importance_file
)

print(
    "\n全局重要性图："
)

print(
    figure_file
)

print(
    "\n注意：SHAP表示模型预测贡献，"
    "不能直接解释为地球化学因果作用。"
)