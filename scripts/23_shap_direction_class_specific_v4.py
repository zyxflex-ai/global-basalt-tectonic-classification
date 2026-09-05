# AUTO-GENERATED V4 COPY. SOURCE: 23_shap_direction_class_specific.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 23_shap_direction_class_specific.py
#
# Analysis step.
#
# Analysis step.
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

from scipy.stats import spearmanr

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

figure_dir = os.path.join(
    result_dir,
    "class_beeswarm"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

os.makedirs(
    figure_dir,
    exist_ok=True
)

summary_file = os.path.join(
    result_dir,
    "shap_direction_class_specific_v4.csv"
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
print("类别特异 SHAP 方向分析")
print("=" * 90)

print(
    "训练集：",
    train.shape
)

print(
    "独立测试集：",
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
    .to_numpy(dtype=int)
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

    class_weights = {
        c:
        n / (k * counts[c])
        for c in range(k)
    }

    return np.array(
        [
            class_weights[v]
            for v in y
        ],
        dtype=float
    )


sample_weight = (
    make_sample_weights(
        y_train
    )
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
# 6. SHAP
# ============================================================

print("\n计算SHAP...")

explainer = shap.TreeExplainer(
    model
)

raw_shap = (
    explainer.shap_values(
        X_test
    )
)


# ============================================================
# Analysis step.
#
# samples × features × classes
# ============================================================

if isinstance(
    raw_shap,
    list
):

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
            "无法识别SHAP维度："
            + str(arr.shape)
        )


    if (
        arr.shape[0] == len(X_test)
        and
        arr.shape[1] == len(features)
        and
        arr.shape[2] == len(classes)
    ):

        shap_values = arr


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
            "无法识别SHAP shape："
            + str(arr.shape)
        )


print(
    "SHAP shape：",
    shap_values.shape
)


# ============================================================
# Analysis step.
# ============================================================

records = []


for class_id, class_name in enumerate(
    classes
):

    print("\n")
    print("=" * 90)
    print(class_name)
    print("=" * 90)


    class_shap = (
        shap_values[
            :,
            :,
            class_id
        ]
    )


    for feature_id, feature in enumerate(
        features
    ):

        values = (
            X_test[
                feature
            ]
            .to_numpy(
                dtype=float
            )
        )

        shap_col = (
            class_shap[
                :,
                feature_id
            ]
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        measured = (
            ~np.isnan(values)
        )

        missing = (
            np.isnan(values)
        )


        measured_values = (
            values[
                measured
            ]
        )

        measured_shap = (
            shap_col[
                measured
            ]
        )


        n_measured = int(
            measured.sum()
        )

        n_missing = int(
            missing.sum()
        )


        # ----------------------------------------------------
        # Spearman
        # ----------------------------------------------------

        if (
            n_measured >= 10
            and
            np.unique(
                measured_values
            ).size > 1
        ):

            rho, p_value = (
                spearmanr(
                    measured_values,
                    measured_shap
                )
            )

        else:

            rho = np.nan
            p_value = np.nan


        # ----------------------------------------------------
        # Q1 / Q4
        #
        # Analysis step.
        # ----------------------------------------------------

        if n_measured >= 20:

            q25 = np.nanpercentile(
                measured_values,
                25
            )

            q75 = np.nanpercentile(
                measured_values,
                75
            )


            low_mask = (
                measured_values
                <=
                q25
            )

            high_mask = (
                measured_values
                >=
                q75
            )


            low_mean_shap = (
                np.mean(
                    measured_shap[
                        low_mask
                    ]
                )
            )

            high_mean_shap = (
                np.mean(
                    measured_shap[
                        high_mask
                    ]
                )
            )


            delta_high_low = (
                high_mean_shap
                -
                low_mean_shap
            )

        else:

            q25 = np.nan
            q75 = np.nan

            low_mean_shap = np.nan
            high_mean_shap = np.nan
            delta_high_low = np.nan


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        if n_missing > 0:

            missing_mean_shap = (
                np.mean(
                    shap_col[
                        missing
                    ]
                )
            )

        else:

            missing_mean_shap = np.nan


        # ----------------------------------------------------
        # mean absolute SHAP
        # ----------------------------------------------------

        mean_abs_shap = (
            np.mean(
                np.abs(
                    shap_col
                )
            )
        )


        # ----------------------------------------------------
        # Analysis step.
        #
        # Analysis step.
        # Analysis step.
        # ----------------------------------------------------

        if np.isnan(
            delta_high_low
        ):

            direction = "NA"

        elif delta_high_low > 0:

            direction = (
                "HIGHER_VALUE_MORE_POSITIVE"
            )

        elif delta_high_low < 0:

            direction = (
                "LOWER_VALUE_MORE_POSITIVE"
            )

        else:

            direction = "NO_DIFFERENCE"


        records.append({

            "class":
                class_name,

            "feature":
                feature,

            "mean_abs_shap":
                mean_abs_shap,

            "spearman_rho":
                rho,

            "spearman_p":
                p_value,

            "q25":
                q25,

            "q75":
                q75,

            "low_q_mean_shap":
                low_mean_shap,

            "high_q_mean_shap":
                high_mean_shap,

            "delta_high_minus_low":
                delta_high_low,

            "direction":
                direction,

            "n_measured":
                n_measured,

            "n_missing":
                n_missing,

            "missing_rate":
                n_missing
                /
                len(test),

            "missing_mean_shap":
                missing_mean_shap
        })


# ============================================================
# Analysis step.
# ============================================================

summary = pd.DataFrame(
    records
)


# Analysis step.
summary[
    "class_rank"
] = (
    summary
    .groupby(
        "class"
    )[
        "mean_abs_shap"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


summary = (
    summary
    .sort_values(
        [
            "class",
            "class_rank"
        ]
    )
)


summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
# ============================================================

print("\n\n")
print("=" * 100)
print("各类别 Top 5 SHAP方向")
print("=" * 100)


for class_name in classes:

    sub = (
        summary[
            summary[
                "class"
            ] == class_name
        ]
        .sort_values(
            "class_rank"
        )
        .head(5)
    )


    print(
        f"\n{class_name}"
    )

    print(
        sub[
            [
                "class_rank",
                "feature",
                "mean_abs_shap",
                "spearman_rho",
                "low_q_mean_shap",
                "high_q_mean_shap",
                "delta_high_minus_low",
                "direction",
                "missing_rate",
                "missing_mean_shap"
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# Analysis step.
#
# Analysis step.
# Analysis step.
# ============================================================

summary[
    "missing_effect_abs"
] = (
    summary[
        "missing_mean_shap"
    ]
    .abs()
)


missing_watch = (
    summary[
        summary[
            "missing_rate"
        ] >= 0.10
    ]
    .sort_values(
        "missing_effect_abs",
        ascending=False
    )
)


print("\n\n")
print("=" * 100)
print("缺失模式警惕：Top 20")
print("=" * 100)


print(
    missing_watch[
        [
            "class",
            "feature",
            "missing_rate",
            "missing_mean_shap",
            "mean_abs_shap"
        ]
    ]
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# Analysis step.
# ============================================================

print("\n生成各类别 beeswarm 图...")


for class_id, class_name in enumerate(
    classes
):

    class_shap = (
        shap_values[
            :,
            :,
            class_id
        ]
    )


    plt.figure()


    shap.summary_plot(

        class_shap,

        X_test,

        feature_names=
            features,

        max_display=15,

        show=False
    )


    plt.title(
        f"{class_name} - Class-specific SHAP"
    )

    plt.tight_layout()


    figure_file = os.path.join(
        figure_dir,
        f"shap_beeswarm_{class_name}_v4.png"
    )


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
print("=" * 90)
print("类别特异 SHAP 方向分析完成")
print("=" * 90)


print(
    "方向汇总表："
)

print(
    summary_file
)


print(
    "\nBeeswarm目录："
)

print(
    figure_dir
)


print(
    "\n解释原则："
)

print(
    "delta_high_minus_low > 0："
    "该变量高值总体上比低值更推动该类别模型输出。"
)

print(
    "delta_high_minus_low < 0："
    "该变量低值总体上比高值更推动该类别模型输出。"
)

print(
    "但SHAP关系可能非线性，"
    "最终还需结合beeswarm和地球化学背景解释。"
)