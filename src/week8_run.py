"""
Week 8 — full feature set: application + relational aggregations, WoE-encoded,
IV-ranked, fit with both sklearn and the from-scratch LR. The point this week
isn't a higher AUC, it's deliberately reproducing and diagnosing the
sklearn-vs-scratch coefficient divergence documented in the oracle's known
limitations -- expected to show up here because several relational features
are near-duplicates of each other by construction (mean vs max of the same
underlying series, DPD across POS/CC/installments all describing the same
"did they pay late" behavior from different tables).

IV threshold is intentionally lower than Week 7's (0.01 instead of 0.02) --
keeping more, more-correlated features is the point this week, not curating
them away.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from baseline_features import BASE_NUMERIC_COLS, CATEGORICAL_COLS, engineer_baseline
from from_scratch_lr import FromScratchLogisticRegression
from io_raw import load_application
from metrics_scratch import auc_rank_sum, gini, ks_statistic
from relational_features import build_all_relational_features
from woe_iv import fit_woe, iv_strength, transform_woe

ENGINEERED_NUMERIC = [
    "AGE_YEARS", "EMPLOYED_YEARS",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
]
FLAG_AS_CATEGORICAL = ["DAYS_EMPLOYED_ANOM"]

RELATIONAL_NUMERIC = [
    "BUREAU_COUNT", "BUREAU_ACTIVE_SHARE", "BUREAU_DAYS_CREDIT_MIN",
    "BUREAU_DAYS_CREDIT_MEAN", "BUREAU_OVERDUE_MAX", "BUREAU_OVERDUE_MEAN",
    "BUREAU_CREDIT_SUM_TOTAL", "BUREAU_CREDIT_SUM_DEBT_TOTAL",
    "BUREAU_EVER_DELINQUENT_SHARE",
    "PREV_APP_COUNT", "PREV_APPROVED_SHARE", "PREV_REFUSED_SHARE",
    "PREV_AMT_APPLICATION_MEAN", "PREV_AMT_CREDIT_MEAN", "PREV_DAYS_DECISION_MAX",
    "POS_DPD_MAX", "POS_DPD_MEAN", "POS_DPD_DEF_MAX", "POS_CNT_INSTALMENT_MEAN",
    "CC_BALANCE_MEAN", "CC_BALANCE_MAX", "CC_UTILIZATION_MEAN", "CC_DPD_MAX",
    "INSTAL_COUNT", "INSTAL_DAYS_LATE_MEAN", "INSTAL_DAYS_LATE_MAX",
    "INSTAL_SHORTFALL_MEAN", "INSTAL_SHORTFALL_SUM",
]


def build_full_feature_table():
    train_raw = load_application("train")
    app_feats = engineer_baseline(train_raw)

    cache = Path(__file__).resolve().parent / "_cache_relational_features.parquet"
    rel_feats = pd.read_parquet(cache) if cache.exists() else build_all_relational_features()

    full = app_feats.merge(rel_feats, on="SK_ID_CURR", how="left")
    return full


def main():
    full = build_full_feature_table()
    y = full["TARGET"]
    X = full.drop(columns=["TARGET", "SK_ID_CURR"])

    numeric_cols = [c for c in BASE_NUMERIC_COLS + ENGINEERED_NUMERIC + RELATIONAL_NUMERIC]
    cat_cols = CATEGORICAL_COLS + FLAG_AS_CATEGORICAL

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    woe_fits = {}
    for col in numeric_cols:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)
    for col in cat_cols:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=True)

    iv_table = sorted(((c, r["iv"]) for c, r in woe_fits.items()), key=lambda t: t[1], reverse=True)
    print(f"total candidate features: {len(woe_fits)}")
    print("\nTop 20 by IV:")
    for c, iv in iv_table[:20]:
        print(f"  {c:32s} IV={iv:.4f}  ({iv_strength(iv)})")

    kept_cols = [c for c, iv in iv_table if iv >= 0.01]
    print(f"\nkeeping {len(kept_cols)} / {len(woe_fits)} features (IV >= 0.01, deliberately permissive)")

    def woe_encode(df, cols):
        return np.column_stack([transform_woe(df[c], woe_fits[c]).values for c in cols])

    Xw_train = woe_encode(X_train, kept_cols)
    Xw_val = woe_encode(X_val, kept_cols)

    mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
    std[std == 0] = 1.0
    Xw_train_s = (Xw_train - mean) / std
    Xw_val_s = (Xw_val - mean) / std

    # --- pairwise correlation among the kept WoE features: where's the collinearity? ---
    corr = np.corrcoef(Xw_train_s, rowvar=False)
    n = len(kept_cols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((kept_cols[i], kept_cols[j], corr[i, j]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print("\nTop 10 most correlated WoE feature pairs (this is the divergence's likely source):")
    for a, b, r in pairs[:10]:
        print(f"  {a:28s} <-> {b:28s}  r={r:+.3f}")

    # --- fit both models ---
    sk = LogisticRegression(max_iter=3000, class_weight="balanced")
    sk.fit(Xw_train_s, y_train)
    sk_val_pred = sk.predict_proba(Xw_val_s)[:, 1]

    mine = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    mine.fit(Xw_train_s, y_train.values, class_weight="balanced")
    my_val_pred = mine.predict_proba(Xw_val_s)

    coef_diff = np.abs(sk.coef_.ravel() - mine.coef_)
    pred_corr = np.corrcoef(sk_val_pred, my_val_pred)[0, 1]

    print(f"\nsklearn val AUC : {roc_auc_score(y_val, sk_val_pred):.4f}")
    print(f"scratch val AUC : {auc_rank_sum(y_val, my_val_pred):.4f}")
    print(f"scratch val GINI: {gini(y_val, my_val_pred):.4f}")
    print(f"scratch val KS  : {ks_statistic(y_val, my_val_pred):.4f}")
    print(f"\nmax |coef diff| vs sklearn : {coef_diff.max():.4f}")
    print(f"mean |coef diff| vs sklearn: {coef_diff.mean():.4f}")
    print(f"prediction correlation     : {pred_corr:.6f}")

    print("\nfeatures with the largest sklearn-vs-scratch coefficient gap:")
    order = np.argsort(-coef_diff)
    for idx in order[:8]:
        print(f"  {kept_cols[idx]:28s} sklearn={sk.coef_.ravel()[idx]:+.4f}  scratch={mine.coef_[idx]:+.4f}  diff={coef_diff[idx]:.4f}")

    return {
        "kept_cols": kept_cols,
        "woe_fits": woe_fits,
        "corr_pairs": pairs[:10],
        "coef_diff_max": float(coef_diff.max()),
        "pred_corr": float(pred_corr),
        "val_auc": float(auc_rank_sum(y_val, my_val_pred)),
        "val_ks": float(ks_statistic(y_val, my_val_pred)),
        "model": mine,
        "standardize_mean": mean,
        "standardize_std": std,
        "X_val_raw": X_val,
        "y_val": y_val,
    }


if __name__ == "__main__":
    main()
