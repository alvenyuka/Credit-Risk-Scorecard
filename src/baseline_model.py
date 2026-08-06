"""
Week 6 — blind rebuild, step 3: naive sklearn logistic regression baseline.

Goal here isn't a good model -- it's a correct, honest number to improve on
in Week 7 once WoE/IV replaces this crude impute+scale+one-hot pipeline.

Pipeline choices:
- Median imputation for numeric NaNs (mean would be pulled around by the
  skewed AMT_* columns).
- StandardScaler because raw features are on wildly different scales
  (AGE_YEARS ~20-70 vs CREDIT_INCOME_RATIO ~0-20) and plain LogisticRegression
  is scale-sensitive.
- OneHotEncoder(handle_unknown="ignore") for categoricals so the held-out
  split doesn't blow up on a category the train split didn't see.
- class_weight="balanced": TARGET is ~92/8, and an unweighted LR would just
  learn to always predict "no default" and still look 92% accurate. AUC is
  the metric that matters here, not accuracy, but balanced weighting still
  helps the decision boundary actually separate the classes instead of
  collapsing toward the majority class.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline_features import BASE_NUMERIC_COLS, CATEGORICAL_COLS, engineer_baseline
from io_raw import load_application

ENGINEERED_NUMERIC = [
    "AGE_YEARS", "DAYS_EMPLOYED_ANOM", "EMPLOYED_YEARS",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
]
NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    train_raw = load_application("train")
    feats = engineer_baseline(train_raw)

    y = feats["TARGET"]
    X = feats.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    train_pred = pipe.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)

    print(f"train AUC: {train_auc:.4f}")
    print(f"val   AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    main()
