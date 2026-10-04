"""
WoE baseline on the application table alone: WoE-encode the application
features, rank them by IV, fit the from-scratch logistic regression, and report
AUC, GINI, KS and train-to-holdout PSI with the from-scratch metrics. The same
WoE features are also fitted with scikit-learn as a cross-check.

This is the intermediate step between the impute-scale baseline
(baseline_model.py) and the full scorecard (run_pipeline.py); it validates the
primitives on real data before relational features are added.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from baseline_features import engineer_baseline
from feature_lists import APPLICATION_NUMERIC, CATEGORICAL_COLS
from from_scratch_lr import FromScratchLogisticRegression
from io_raw import load_application
from metrics_scratch import auc_rank_sum, gini, ks_statistic, psi
from woe_iv import fit_woe, iv_strength, transform_woe


def main():
    """Fit the application-only WoE logistic regression and print its holdout AUC."""
    train_raw = load_application("train")
    feats = engineer_baseline(train_raw)

    y = feats["TARGET"]
    X = feats.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- fit WoE on train only, transform both splits ---
    woe_fits = {}
    for col in APPLICATION_NUMERIC:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)
    for col in CATEGORICAL_COLS:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=True)

    iv_table = sorted(
        ((col, res["iv"]) for col, res in woe_fits.items()),
        key=lambda t: t[1], reverse=True,
    )
    print("Top 15 features by Information Value:")
    for col, iv in iv_table[:15]:
        print(f"  {col:28s} IV={iv:.4f}  ({iv_strength(iv)})")

    # keep anything at least "weak" (IV >= 0.02); useless features only add noise
    kept_cols = [col for col, iv in iv_table if iv >= 0.02]
    print(f"\nkeeping {len(kept_cols)} / {len(woe_fits)} features (IV >= 0.02)")

    def woe_encode(df, cols):
        """Matrix with one column per feature, each value replaced by its bin's WoE."""
        return np.column_stack([transform_woe(df[c], woe_fits[c]).values for c in cols])

    Xw_train = woe_encode(X_train, kept_cols)
    Xw_val = woe_encode(X_val, kept_cols)

    # standardise for faster, more reliable gradient-descent convergence
    mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
    std[std == 0] = 1.0
    Xw_train_s = (Xw_train - mean) / std
    Xw_val_s = (Xw_val - mean) / std

    # --- from-scratch LR ---
    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3, verbose=False)
    scratch.fit(Xw_train_s, y_train.values, class_weight="balanced")
    scratch_val_pred = scratch.predict_proba(Xw_val_s)
    scratch_train_pred = scratch.predict_proba(Xw_train_s)

    # --- sklearn LR on the identical WoE features, as a cross-check ---
    sk = LogisticRegression(max_iter=2000, class_weight="balanced")
    sk.fit(Xw_train_s, y_train)
    sk_val_pred = sk.predict_proba(Xw_val_s)[:, 1]

    print("\n--- results on WoE-encoded features ---")
    print(f"sklearn LR   val AUC (roc_auc_score): {roc_auc_score(y_val, sk_val_pred):.4f}")
    print(f"scratch LR   val AUC (auc_rank_sum) : {auc_rank_sum(y_val, scratch_val_pred):.4f}")
    print(f"scratch LR   val AUC (sklearn check) : {roc_auc_score(y_val, scratch_val_pred):.4f}")
    print(f"scratch LR   val GINI               : {gini(y_val, scratch_val_pred):.4f}")
    print(f"scratch LR   val KS                 : {ks_statistic(y_val, scratch_val_pred):.4f}")
    print(f"scratch LR   train->val PSI (scores) : {psi(scratch_train_pred, scratch_val_pred):.4f}")

    coef_diff = np.max(np.abs(sk.coef_.ravel() - scratch.coef_))
    pred_corr = np.corrcoef(sk_val_pred, scratch_val_pred)[0, 1]
    print(f"\nmax |coef diff| vs sklearn: {coef_diff:.4f}")
    print(f"prediction correlation vs sklearn: {pred_corr:.6f}")

    return {
        "kept_cols": kept_cols,
        "val_auc": auc_rank_sum(y_val, scratch_val_pred),
        "val_gini": gini(y_val, scratch_val_pred),
        "val_ks": ks_statistic(y_val, scratch_val_pred),
        "coef_diff": coef_diff,
    }


if __name__ == "__main__":
    main()
