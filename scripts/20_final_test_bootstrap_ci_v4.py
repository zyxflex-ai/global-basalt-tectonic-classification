# AUTO-GENERATED V4 COPY. SOURCE: 20_final_test_bootstrap_ci.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 20_final_test_bootstrap_ci.py
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# - Accuracy
# - Balanced Accuracy
# - Macro-F1
# - Weighted-F1
# Analysis step.
#
# Analysis step.
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
    f1_score
)
from _project_paths import FINAL_DIR


# ============================================================
# Analysis step.
# ============================================================

input_file = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
    r"\final_independent_test_predictions_v4.csv"
)

result_dir = (
    str(FINAL_DIR) + 
    r"\05_results\final_test"
)

output_file = os.path.join(
    result_dir,
    "final_test_bootstrap_ci_v4.csv"
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
print("最终独立测试 Bootstrap CI")
print("=" * 90)

print(
    "测试样品数：",
    len(df)
)

print(
    "独立CV_GROUP_V4数：",
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


# ============================================================
# Analysis step.
# ============================================================

df["TRUE_ID"] = (
    df["TRUE_LABEL"]
    .map(class_to_int)
)


if df["TRUE_ID"].isna().any():

    raise RuntimeError(
        "发现未知TRUE_LABEL"
    )


df["TRUE_ID"] = (
    df["TRUE_ID"]
    .astype(int)
)


# ============================================================
# Analysis step.
# ============================================================

model_columns = {

    "XGBoost":
        "XGB_PRED_LABEL",

    "RandomForest":
        "RF_PRED_LABEL",

    "RBF-SVM":
        "SVM_PRED_LABEL"
}


for model_name, col in (
    model_columns.items()
):

    df[
        f"{model_name}_ID"
    ] = (
        df[col]
        .map(class_to_int)
    )

    if df[
        f"{model_name}_ID"
    ].isna().any():

        raise RuntimeError(
            f"{model_name} "
            "存在未知预测类别"
        )

    df[
        f"{model_name}_ID"
    ] = (
        df[
            f"{model_name}_ID"
        ]
        .astype(int)
    )


# ============================================================
# Analysis step.
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy":
            accuracy_score(
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
                labels=np.arange(8),
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=np.arange(8),
                zero_division=0
            )
    }


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("原始独立测试点估计")
print("=" * 90)


point_estimates = {}


for model_name in model_columns:

    y_true = (
        df["TRUE_ID"]
        .to_numpy()
    )

    y_pred = (
        df[
            f"{model_name}_ID"
        ]
        .to_numpy()
    )


    metrics = (
        calculate_metrics(
            y_true,
            y_pred
        )
    )


    point_estimates[
        model_name
    ] = metrics


    print(
        f"\n{model_name}"
    )

    for metric, value in (
        metrics.items()
    ):

        print(
            f"{metric:20s}: "
            f"{value:.4f}"
        )


# ============================================================
# Analysis step.
# ============================================================

n_bootstrap = 2000

random_seed = 42

rng = np.random.default_rng(
    random_seed
)


groups = (
    df["CV_GROUP_V4"]
    .drop_duplicates()
    .to_numpy()
)

n_groups = len(groups)


print("\n")
print("=" * 90)
print("开始CV_GROUP_V4级Bootstrap")
print("=" * 90)

print(
    "Bootstrap次数：",
    n_bootstrap
)

print(
    "每次抽取CV_GROUP_V4数：",
    n_groups
)


bootstrap_records = []


# ============================================================
# Analysis step.
# ============================================================

group_indices = {

    g: np.where(
        df["CV_GROUP_V4"]
        .to_numpy() == g
    )[0]

    for g in groups
}


# ============================================================
# Analysis step.
# ============================================================

for b in range(
    n_bootstrap
):

    # --------------------------------------------------------
    # Analysis step.
    # --------------------------------------------------------

    sampled_groups = (
        rng.choice(
            groups,
            size=n_groups,
            replace=True
        )
    )


    # --------------------------------------------------------
    # Analysis step.
    # Analysis step.
    # Analysis step.
    # --------------------------------------------------------

    sampled_indices = np.concatenate(
        [
            group_indices[g]
            for g in sampled_groups
        ]
    )


    boot = df.iloc[
        sampled_indices
    ]


    y_true = (
        boot["TRUE_ID"]
        .to_numpy()
    )


    # --------------------------------------------------------
    # Analysis step.
    # Analysis step.
    # --------------------------------------------------------

    if len(
        np.unique(y_true)
    ) < len(classes):

        continue


    for model_name in model_columns:

        y_pred = (
            boot[
                f"{model_name}_ID"
            ]
            .to_numpy()
        )


        metrics = (
            calculate_metrics(
                y_true,
                y_pred
            )
        )


        record = {

            "bootstrap":
                b,

            "model":
                model_name,

            **metrics
        }


        # ====================================================
        # Analysis step.
        # ====================================================

        if model_name == "XGBoost":

            class_f1 = (
                f1_score(
                    y_true,
                    y_pred,

                    labels=
                        np.arange(8),

                    average=None,

                    zero_division=0
                )
            )


            for i, c in enumerate(
                classes
            ):

                record[
                    f"f1_{c}"
                ] = class_f1[i]


        bootstrap_records.append(
            record
        )


    # Analysis step.
    if (
        (b + 1) % 200
        == 0
    ):

        print(
            f"已完成 "
            f"{b + 1}/"
            f"{n_bootstrap}"
        )


# ============================================================
# Analysis step.
# ============================================================

boot_df = pd.DataFrame(
    bootstrap_records
)


print(
    "\n有效Bootstrap轮数：",
    boot_df[
        "bootstrap"
    ].nunique()
)


# ============================================================
# 11. 95% percentile CI
# ============================================================

summary_rows = []


overall_metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1"
]


for model_name in model_columns:

    sub = (
        boot_df[
            boot_df["model"]
            == model_name
        ]
    )


    for metric in (
        overall_metrics
    ):

        values = (
            sub[metric]
            .dropna()
            .to_numpy()
        )


        lower = np.percentile(
            values,
            2.5
        )

        upper = np.percentile(
            values,
            97.5
        )


        point = (
            point_estimates[
                model_name
            ][metric]
        )


        summary_rows.append({

            "model":
                model_name,

            "level":
                "overall",

            "metric":
                metric,

            "class":
                "ALL",

            "estimate":
                point,

            "ci_lower":
                lower,

            "ci_upper":
                upper
        })


# ============================================================
# Analysis step.
# ============================================================

xgb_original_true = (
    df["TRUE_ID"]
    .to_numpy()
)

xgb_original_pred = (
    df["XGBoost_ID"]
    .to_numpy()
)


original_class_f1 = (
    f1_score(

        xgb_original_true,

        xgb_original_pred,

        labels=
            np.arange(8),

        average=None,

        zero_division=0
    )
)


xgb_boot = (
    boot_df[
        boot_df["model"]
        == "XGBoost"
    ]
)


for i, class_name in enumerate(
    classes
):

    col = (
        f"f1_{class_name}"
    )


    values = (
        xgb_boot[
            col
        ]
        .dropna()
        .to_numpy()
    )


    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )


    summary_rows.append({

        "model":
            "XGBoost",

        "level":
            "class",

        "metric":
            "f1",

        "class":
            class_name,

        "estimate":
            original_class_f1[i],

        "ci_lower":
            lower,

        "ci_upper":
            upper
    })


# ============================================================
# Analysis step.
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("总体指标：95% Bootstrap CI")
print("=" * 90)


for model_name in model_columns:

    print(
        f"\n{model_name}"
    )


    sub = summary[
        (summary["model"] == model_name)
        &
        (summary["level"] == "overall")
    ]


    for _, row in (
        sub.iterrows()
    ):

        print(
            f"{row['metric']:20s}: "
            f"{row['estimate']:.4f} "
            f"["
            f"{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}"
            f"]"
        )


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("XGBoost各类别F1：95% Bootstrap CI")
print("=" * 90)


sub = summary[
    (summary["model"] == "XGBoost")
    &
    (summary["level"] == "class")
]


for _, row in (
    sub.iterrows()
):

    print(
        f"{row['class']:5s}: "
        f"{row['estimate']:.4f} "
        f"["
        f"{row['ci_lower']:.4f}, "
        f"{row['ci_upper']:.4f}"
        f"]"
    )


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 90)
print("Bootstrap CI完成")
print("=" * 90)

print(
    "保存："
)

print(
    output_file
)

print(
    "\n方法：CV_GROUP_V4级有放回Bootstrap，"
    "95% percentile confidence interval。"
)