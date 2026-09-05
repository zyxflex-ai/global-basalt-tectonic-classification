# -*- coding: utf-8 -*-
# AUTO-GENERATED V4 COPY. SOURCE: 30_final_full40_independent_test.py. DO NOT EDIT THE V3 SOURCE VIA THIS FILE.

import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

train_file = str(FINAL_DIR.joinpath('03_model_data', 'Basalt_train_stable50_main_v4.csv'))
test_file  = str(FINAL_DIR.joinpath('03_model_data', 'Basalt_test_stable50_main_v4.csv'))


result_dir = str(FINAL_DIR.joinpath('05_results', 'final_test'))

os.makedirs(result_dir, exist_ok=True)


# ============================================================
# Analysis step.
# ============================================================

print("="*90)
print("FINAL FULL-40 XGBoost Independent Test")
print("="*90)


train = pd.read_csv(
    train_file,
    low_memory=False
)

test = pd.read_csv(
    test_file,
    low_memory=False
)


print("训练集:", train.shape)
print("测试集:", test.shape)


# ============================================================
# Analysis step.
# ============================================================

label_col = "LABEL"

le = LabelEncoder()

y_train = le.fit_transform(train[label_col])
y_test = le.transform(test[label_col])

classes = le.classes_


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

"SC(PPM)",
"V(PPM)",
"CR(PPM)",
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
"TH(PPM)",
"U(PPM)",
"CO(PPM)",
"PB(PPM)"

]


print("\nFull-40 特征数:",len(features))


X_train = train[features].copy()
X_test  = test[features].copy()



# ============================================================
# Median Imputation
# ============================================================

print("\n进行median imputation")


imputer = SimpleImputer(
    strategy="median"
)


X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)



# ============================================================
# Analysis step.
# ============================================================

from sklearn.utils.class_weight import compute_class_weight


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


class_weights = dict(
    zip(
        np.unique(y_train),
        weights
    )
)


print("\n类别权重")

for k,v in class_weights.items():
    print(classes[k],":",round(v,4))



sample_weight = np.array(
    [
        class_weights[y]
        for y in y_train
    ]
)



# ============================================================
# Analysis step.
# ============================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(classes),

    learning_rate=0.02,

    max_depth=6,

    min_child_weight=1,

    subsample=0.8,

    colsample_bytree=0.7,

    reg_alpha=0.0,

    reg_lambda=1.0,

    n_estimators=1400,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,

    n_jobs=-1

)



# ============================================================
# Analysis step.
# ============================================================


print("\n训练最终Full40模型...")


t=time.time()


model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight
)


print("完成:",round(time.time()-t,2),"秒")



# ============================================================
# Analysis step.
# ============================================================

pred = model.predict(X_test)

prob = model.predict_proba(X_test)



# ============================================================
# Analysis step.
# ============================================================


acc = accuracy_score(
    y_test,
    pred
)

bal = balanced_accuracy_score(
    y_test,
    pred
)

macro = f1_score(
    y_test,
    pred,
    average="macro"
)

weighted = f1_score(
    y_test,
    pred,
    average="weighted"
)



print("\n")
print("="*90)
print("Full40 XGBoost Independent Test Result")
print("="*90)


print(
"Accuracy:",
round(acc,4)
)

print(
"Balanced Accuracy:",
round(bal,4)
)

print(
"Macro-F1:",
round(macro,4)
)

print(
"Weighted-F1:",
round(weighted,4)
)



print("\n分类报告")

print(
classification_report(
    y_test,
    pred,
    target_names=classes,
    digits=4
)
)



# ============================================================
# Analysis step.
# ============================================================

cm = confusion_matrix(
    y_test,
    pred
)


cm_df = pd.DataFrame(
    cm,
    index=["TRUE_"+x for x in classes],
    columns=["PRED_"+x for x in classes]
)


print("\n混淆矩阵")

print(cm_df)



# ============================================================
# Analysis step.
# ============================================================


metrics = pd.DataFrame(
    {

"model":["XGBoost_Full40_imputed"],

"Accuracy":[acc],

"Balanced_Accuracy":[bal],

"Macro_F1":[macro],

"Weighted_F1":[weighted]

    }
)


metrics_file=os.path.join(
    result_dir,
    "final_full40_independent_test_metrics_v4.csv"
)


metrics.to_csv(
    metrics_file,
    index=False,
    encoding="utf-8-sig"
)



# ============================================================
# Analysis step.
# ============================================================


pred_df=test.copy()


pred_df["TRUE_LABEL"]=le.inverse_transform(y_test)

pred_df["PRED_LABEL"]=le.inverse_transform(pred)


pred_df["MAX_PROB"]=prob.max(axis=1)


for i,c in enumerate(classes):

    pred_df[
        "PROB_"+c
    ]=prob[:,i]



pred_file=os.path.join(
    result_dir,
    "final_full40_independent_test_predictions_v4.csv"
)


pred_df.to_csv(
    pred_file,
    index=False,
    encoding="utf-8-sig"
)



print("\n保存完成")

print(metrics_file)

print(pred_file)



print("\n注意:")
print("测试集结果已重新正式冻结，不再根据测试集调整模型。")