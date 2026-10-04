"""
Application-only baseline: a scikit-learn logistic regression on imputed, scaled
and one-hot-encoded features. It sets the floor that the WoE scorecard has to beat.

Pipeline choices:
- Median imputation for numeric NaNs (the AMT_* columns are skewed, so a mean
  would be pulled around).
- StandardScaler, because raw features sit on very different scales and plain
  logistic regression is scale-sensitive.
- OneHotEncoder(handle_unknown="ignore"), so a category unseen in training does
  not break scoring of the holdout.
- class_weight="balanced": TARGET is about 92/8, and AUC rather than accuracy is
  the metric that matters.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline_features import engineer_baseline
from feature_lists import BASELINE_NUMERIC, CATEGORICAL_COLS
from io_raw import load_application

NUMERIC_COLS = BASELINE_NUMERIC


def build_pipeline() -> Pipeline:
    """Unfitted application-only benchmark: impute, scale, one-hot encode, then a balanced logistic regression."""
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
    """Fit the benchmark on the 80/20 split and print its training and holdout AUC."""
    train_raw = load_application("train")
    feats = engineer_baseline(train_raw)

    y = feats["TARGET"]
    X = feats.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    train_auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])

    print(f"train AUC: {train_auc:.4f}")
    print(f"val   AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    main()
