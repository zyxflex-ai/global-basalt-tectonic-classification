# AUTO-GENERATED V4 COPY. SOURCE: 24_shap_trueclass_measured_robustness.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 24_shap_trueclass_measured_robustness.py
#
# Analysis step.
# 1. all samples + measured only
# 2. true-class samples + measured only
#
# Analysis step.
# Analysis step.
#
# Analysis step.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
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

output_file = os.path.join(
    result_dir,
    "shap_trueclass_measured_robustness_v4.csv"
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
y_test = test["Y"].to_numpy(dtype=int)


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


print("训练最终冻结模型...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight
)

print("计算SHAP...")


# ============================================================
# 4. SHAP
# ============================================================

explainer = shap.TreeExplainer(
    model
)

raw_shap = explainer.shap_values(
    X_test
)


if isinstance(raw_shap, list):

    shap_values = np.stack(
        raw_shap,
        axis=-1
    )

else:

    arr = np.asarray(raw_shap)

    if (
        arr.shape
        ==
        (
            len(X_test),
            len(features),
            len(classes)
        )
    ):

        shap_values = arr

    elif (
        arr.shape
        ==
        (
            len(X_test),
            len(classes),
            len(features)
        )
    ):

        shap_values = np.transpose(
            arr,
            (0, 2, 1)
        )

    elif (
        arr.shape
        ==
        (
            len(classes),
            len(X_test),
            len(features)
        )
    ):

        shap_values = np.transpose(
            arr,
            (1, 2, 0)
        )

    else:

        raise RuntimeError(
            "未知SHAP shape: "
            + str(arr.shape)
        )


print(
    "SHAP shape:",
    shap_values.shape
)


# ============================================================
# Analysis step.
# ============================================================

records = []


for class_id, class_name in enumerate(classes):

    true_mask = (
        y_test == class_id
    )

    print(
        f"\n{class_name}: "
        f"真实样品 n={true_mask.sum()}"
    )


    for feature_id, feature in enumerate(
        features
    ):

        x = (
            X_test[feature]
            .to_numpy(dtype=float)
        )

        sv = (
            shap_values[
                :,
                feature_id,
                class_id
            ]
        )


        measured = ~np.isnan(x)

        true_measured = (
            true_mask
            &
            measured
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        all_mean_abs = np.mean(
            np.abs(sv)
        )


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        if measured.sum() > 0:

            all_measured_mean_abs = (
                np.mean(
                    np.abs(
                        sv[measured]
                    )
                )
            )

        else:

            all_measured_mean_abs = np.nan


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        if true_mask.sum() > 0:

            trueclass_mean_abs = (
                np.mean(
                    np.abs(
                        sv[true_mask]
                    )
                )
            )

            trueclass_mean_signed = (
                np.mean(
                    sv[true_mask]
                )
            )

        else:

            trueclass_mean_abs = np.nan
            trueclass_mean_signed = np.nan


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        if true_measured.sum() > 0:

            true_measured_mean_abs = (
                np.mean(
                    np.abs(
                        sv[
                            true_measured
                        ]
                    )
                )
            )

            true_measured_mean_signed = (
                np.mean(
                    sv[
                        true_measured
                    ]
                )
            )

        else:

            true_measured_mean_abs = np.nan
            true_measured_mean_signed = np.nan


        # ----------------------------------------------------
        # Analysis step.
        # ----------------------------------------------------

        class_n = true_mask.sum()

        class_measured_n = (
            true_measured.sum()
        )

        class_missing_rate = (
            1
            -
            class_measured_n
            /
            class_n
        )


        records.append({

            "class":
                class_name,

            "feature":
                feature,

            "all_mean_abs_shap":
                all_mean_abs,

            "all_measured_mean_abs_shap":
                all_measured_mean_abs,

            "trueclass_mean_abs_shap":
                trueclass_mean_abs,

            "trueclass_mean_signed_shap":
                trueclass_mean_signed,

            "trueclass_measured_mean_abs_shap":
                true_measured_mean_abs,

            "trueclass_measured_mean_signed_shap":
                true_measured_mean_signed,

            "trueclass_n":
                int(class_n),

            "trueclass_measured_n":
                int(class_measured_n),

            "trueclass_missing_rate":
                class_missing_rate
        })


# ============================================================
# Analysis step.
# ============================================================

result = pd.DataFrame(
    records
)


for col, rank_col in [

    (
        "all_mean_abs_shap",
        "rank_all"
    ),

    (
        "all_measured_mean_abs_shap",
        "rank_all_measured"
    ),

    (
        "trueclass_mean_abs_shap",
        "rank_trueclass"
    ),

    (
        "trueclass_measured_mean_abs_shap",
        "rank_trueclass_measured"
    )

]:

    result[
        rank_col
    ] = (
        result
        .groupby("class")[col]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )


# ============================================================
# Analysis step.
# ============================================================

result[
    "rank_shift_missingness"
] = (
    result[
        "rank_all_measured"
    ]
    -
    result[
        "rank_all"
    ]
)


result[
    "rank_shift_trueclass"
] = (
    result[
        "rank_trueclass_measured"
    ]
    -
    result[
        "rank_all"
    ]
)


result.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
# ============================================================

print("\n")
print("=" * 110)
print("真类别 + 实测值：各类别 Top 5")
print("=" * 110)


for class_name in classes:

    sub = (
        result[
            result["class"]
            == class_name
        ]
        .sort_values(
            "rank_trueclass_measured"
        )
        .head(5)
    )


    print(
        f"\n{class_name}"
    )


    print(
        sub[
            [
                "rank_trueclass_measured",
                "feature",
                "trueclass_measured_mean_abs_shap",
                "trueclass_measured_mean_signed_shap",
                "trueclass_measured_n",
                "trueclass_missing_rate",
                "rank_all",
                "rank_all_measured",
                "rank_trueclass"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# Analysis step.
# ============================================================

result[
    "abs_rank_shift_trueclass"
] = (
    result[
        "rank_shift_trueclass"
    ]
    .abs()
)


watch = (
    result
    .sort_values(
        "abs_rank_shift_trueclass",
        ascending=False
    )
)


print("\n")
print("=" * 110)
print("排名变化警惕：Top 20")
print("=" * 110)


print(
    watch[
        [
            "class",
            "feature",
            "rank_all",
            "rank_all_measured",
            "rank_trueclass",
            "rank_trueclass_measured",
            "trueclass_missing_rate"
        ]
    ]
    .head(20)
    .round(4)
    .to_string(index=False)
)


print("\n保存：")
print(output_file)