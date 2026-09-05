# AUTO-GENERATED V4 COPY. SOURCE: 25_shap_dependence_key_features.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.
# ============================================================
# 25_shap_dependence_key_features.py
#
# Analysis step.
#
# Analysis step.
# CAB  : TiO2, Sr
# IAB  : CaO, TiO2
# IOAB : TiO2, Nb
# BABB : SiO2, Nb
# MORB : Sr, Y
# OIB  : TiO2, CaO
# OPB  : P2O5, Sr
# CFB  : FE_TOTAL, Zr
#
# Analysis step.
# Analysis step.
# Analysis step.
# Analysis step.
#
# Analysis step.
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
    r"\05_results\shap\dependence"
)

os.makedirs(
    result_dir,
    exist_ok=True
)

summary_file = os.path.join(
    result_dir,
    "shap_dependence_key_features_v4.csv"
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

key_pairs = {

    "CAB": [
        "TIO2(WT%)",
        "SR(PPM)"
    ],

    "IAB": [
        "CAO(WT%)",
        "TIO2(WT%)"
    ],

    "IOAB": [
        "TIO2(WT%)",
        "NB(PPM)"
    ],

    "BABB": [
        "SIO2(WT%)",
        "NB(PPM)"
    ],

    "MORB": [
        "SR(PPM)",
        "Y(PPM)"
    ],

    "OIB": [
        "TIO2(WT%)",
        "CAO(WT%)"
    ],

    "OPB": [
        "P2O5(WT%)",
        "SR(PPM)"
    ],

    "CFB": [
        "FE_TOTAL(WT%)",
        "ZR(PPM)"
    ]
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


print("训练冻结模型...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight
)

print("计算SHAP...")


# ============================================================
# 5. SHAP
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
        arr.shape ==
        (
            len(X_test),
            len(features),
            len(classes)
        )
    ):

        shap_values = arr

    elif (
        arr.shape ==
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
        arr.shape ==
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
#
# Analysis step.
# Analysis step.
# ============================================================

def make_quantile_bins(
    x,
    sv,
    n_bins=15
):

    temp = pd.DataFrame({
        "x": x,
        "shap": sv
    })

    temp = temp.dropna()

    if len(temp) < 30:
        return pd.DataFrame()


    try:

        temp["bin"] = pd.qcut(
            temp["x"],
            q=n_bins,
            duplicates="drop"
        )

    except ValueError:

        return pd.DataFrame()


    out = (
        temp
        .groupby(
            "bin",
            observed=True
        )
        .agg(
            x_median=("x", "median"),
            x_min=("x", "min"),
            x_max=("x", "max"),

            shap_median=("shap", "median"),
            shap_mean=("shap", "mean"),

            shap_q25=(
                "shap",
                lambda s:
                np.percentile(s, 25)
            ),

            shap_q75=(
                "shap",
                lambda s:
                np.percentile(s, 75)
            ),

            n=("shap", "size")
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# Analysis step.
# ============================================================

records = []


for class_name, pair_features in (
    key_pairs.items()
):

    class_id = class_to_int[
        class_name
    ]

    print("\n")
    print("=" * 90)
    print(class_name)
    print("=" * 90)


    for feature in pair_features:

        feature_id = features.index(
            feature
        )

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

        x_measured = x[
            measured
        ]

        sv_measured = sv[
            measured
        ]

        labels_measured = y_test[
            measured
        ]


        print(
            f"\n{feature}"
        )

        print(
            "实测样品数：",
            len(x_measured)
        )


        # ====================================================
        # Analysis step.
        #
        # Analysis step.
        # Analysis step.
        # ====================================================

        x01 = np.percentile(
            x_measured,
            1
        )

        x99 = np.percentile(
            x_measured,
            99
        )

        plot_mask = (
            (x_measured >= x01)
            &
            (x_measured <= x99)
        )


        xp = x_measured[
            plot_mask
        ]

        sp = sv_measured[
            plot_mask
        ]

        lp = labels_measured[
            plot_mask
        ]


        # ====================================================
        # Analysis step.
        # ====================================================

        bins = make_quantile_bins(
            x_measured,
            sv_measured,
            n_bins=15
        )


        if len(bins) == 0:

            print(
                "无法生成稳定分箱，跳过。"
            )

            continue


        # ====================================================
        # Analysis step.
        # ====================================================

        for bin_id, row in (
            bins.iterrows()
        ):

            records.append({

                "class":
                    class_name,

                "feature":
                    feature,

                "bin":
                    bin_id + 1,

                "x_median":
                    row[
                        "x_median"
                    ],

                "x_min":
                    row[
                        "x_min"
                    ],

                "x_max":
                    row[
                        "x_max"
                    ],

                "shap_median":
                    row[
                        "shap_median"
                    ],

                "shap_mean":
                    row[
                        "shap_mean"
                    ],

                "shap_q25":
                    row[
                        "shap_q25"
                    ],

                "shap_q75":
                    row[
                        "shap_q75"
                    ],

                "n":
                    int(
                        row["n"]
                    )
            })


        # ====================================================
        # Analysis step.
        # ====================================================

        print(
            bins[
                [
                    "x_median",
                    "shap_median",
                    "n"
                ]
            ]
            .round(4)
            .to_string(
                index=False
            )
        )


        # ====================================================
        # Analysis step.
        #
        # Analysis step.
        # Analysis step.
        # Analysis step.
        # ====================================================

        plt.figure(
            figsize=(7.5, 5.5)
        )


        plt.scatter(
            xp,
            sp,
            s=8,
            alpha=0.18,
            label="All measured test samples"
        )


        true_target = (
            lp == class_id
        )


        plt.scatter(
            xp[
                true_target
            ],
            sp[
                true_target
            ],
            s=25,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label=f"True {class_name}"
        )


        plt.plot(
            bins[
                "x_median"
            ],
            bins[
                "shap_median"
            ],
            marker="o",
            linewidth=2,
            label="Binned median SHAP"
        )


        plt.axhline(
            0,
            linewidth=1
        )


        plt.xlabel(
            feature
        )

        plt.ylabel(
            f"SHAP value for {class_name}"
        )

        plt.title(
            f"{class_name}: {feature}"
        )

        plt.legend(
            frameon=False
        )

        plt.tight_layout()


        safe_feature = (
            feature
            .replace("(WT%)", "_WT")
            .replace("(PPM)", "_PPM")
            .replace("%", "")
            .replace("/", "_")
        )


        figure_file = os.path.join(
            result_dir,
            f"dependence_{class_name}_{safe_feature}_v4.png"
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

summary = pd.DataFrame(
    records
)

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Analysis step.
#
# Analysis step.
# ============================================================

print("\n")
print("=" * 100)
print("分箱中位SHAP > 0 的区间（仅辅助解释）")
print("=" * 100)


for class_name, pair_features in (
    key_pairs.items()
):

    print(
        f"\n{class_name}"
    )


    for feature in pair_features:

        sub = summary[
            (summary["class"] == class_name)
            &
            (summary["feature"] == feature)
        ].copy()


        positive = sub[
            sub[
                "shap_median"
            ] > 0
        ]


        print(
            f"\n{feature}"
        )


        if len(positive) == 0:

            print(
                "  没有分箱中位SHAP > 0"
            )

        else:

            for _, row in (
                positive.iterrows()
            ):

                print(
                    f"  "
                    f"{row['x_min']:.4g}"
                    f" – "
                    f"{row['x_max']:.4g}"
                    f", median SHAP="
                    f"{row['shap_median']:.3f}"
                )


print("\n")
print("=" * 90)
print("Dependence分析完成")
print("=" * 90)

print(
    "趋势数据："
)

print(
    summary_file
)

print(
    "\n图片目录："
)

print(
    result_dir
)

print(
    "\n注意："
    "分箱SHAP>0区间只能用于描述模型响应，"
    "不能直接作为新的构造判别阈值。"
)