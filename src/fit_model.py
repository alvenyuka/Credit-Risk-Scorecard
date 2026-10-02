"""
Fits the scorecard model on the full feature set: application features plus
relational aggregates, WoE-encoded, screened, and fitted with both the
from-scratch logistic regression and scikit-learn as a cross-check.

Feature screening, in order, all on the training split only:

1. Information value: drop candidates with IV below IV_THRESHOLD (0.01).
2. Correlation gate: of any WoE pair with |r| above CORRELATION_THRESHOLD (0.9),
   keep the feature with the higher IV (ties broken by name), so near-duplicate
   encodings of one characteristic never enter together.
3. Sign gate: WoE = ln(good/bad), so in a P(bad) model every coefficient must be
   negative. A positive coefficient means that, given the other features, the
   characteristic's effect has reversed (a suppression effect of collinearity),
   which would award points to the riskier bins and produce wrong reason codes.
   The feature with the largest positive coefficient is removed and the model
   refitted until every coefficient is negative. The gate runs on scikit-learn's
   fit of the same class-balanced objective for speed, then finishes on the
   from-scratch fit the scorecard uses (which penalises weights slightly
   differently), and scorecard.assert_woe_coefficient_signs checks the result.

The BUREAU_OVERDUE_MAX / BUREAU_OVERDUE_MEAN pair correlates at r = 1.000 not
because the features are redundant by definition but because decile binning of a
zero-inflated column collapses both to the same two-bin split (see
fit_continuous_bins in woe_iv.py); the correlation gate removes one of them.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from baseline_features import engineer_baseline
from feature_lists import (
    CORRELATION_THRESHOLD,
    IV_THRESHOLD,
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    PROHIBITED_BASES,
)
from from_scratch_lr import FromScratchLogisticRegression
from io_raw import DATA_DIR, load_application
from metrics_scratch import auc_rank_sum, gini, ks_statistic
from relational_features import load_relational_features
from woe_iv import fit_woe, iv_strength, transform_woe

SEED = 42


def build_full_feature_table() -> pd.DataFrame:
    """Application features joined to the relational aggregates, one row per applicant.

    The row order (and RangeIndex) is that of application_train.csv.
    """
    app_feats = engineer_baseline(load_application("train"))
    rel_feats = load_relational_features()
    return app_feats.merge(rel_feats, on="SK_ID_CURR", how="left")


def rank_by_iv(woe_fits: dict, cols=None) -> list:
    """Feature names by descending IV, ties broken by name (deterministic)."""
    cols = list(woe_fits) if cols is None else list(cols)
    return sorted(cols, key=lambda c: (-woe_fits[c]["iv"], c))


def correlation_gate(W: np.ndarray, cols: list, ivs: dict, threshold: float = CORRELATION_THRESHOLD):
    """Drop one feature of every pair whose WoE columns correlate above `threshold`.

    Features are visited in descending IV (ties by name); a feature is kept only if
    its |r| with every feature already kept is at most `threshold`, so of each
    offending pair the higher-IV feature survives.

    Returns (kept, dropped) where dropped lists (feature, kept_partner, r).
    """
    pos = {c: i for i, c in enumerate(cols)}
    corr = np.corrcoef(W, rowvar=False)
    kept, dropped = [], []
    for c in sorted(cols, key=lambda k: (-ivs[k], k)):
        partner = next((k for k in kept if abs(corr[pos[c], pos[k]]) > threshold), None)
        if partner is None:
            kept.append(c)
        else:
            dropped.append((c, partner, float(corr[pos[c], pos[partner]])))
    return kept, dropped


def _sklearn_balanced(Xs: np.ndarray, y) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, y)


def eliminate_wrong_signs(Xs: np.ndarray, y, cols: list, fit=_sklearn_balanced):
    """Remove features with positive coefficients one at a time until none remain.

    Xs holds the standardised WoE columns in the order of `cols`. Each round refits
    on the surviving columns and removes the feature with the largest positive
    coefficient (ties by name). Returns (kept, removed) where removed lists
    (feature, coefficient at removal) in removal order.
    """
    kept = list(cols)
    pos = {c: i for i, c in enumerate(cols)}
    removed = []
    while kept:
        model = fit(Xs[:, [pos[c] for c in kept]], y)
        coef = np.ravel(model.coef_)
        wrong = [(c, float(b)) for c, b in zip(kept, coef) if b > 0]
        if not wrong:
            break
        victim = max(wrong, key=lambda t: (t[1], t[0]))
        removed.append(victim)
        kept.remove(victim[0])
    return kept, removed


def fit_full_model(full: pd.DataFrame = None, verbose: bool = True) -> dict:
    if full is None:
        full = build_full_feature_table()
    y = full["TARGET"]
    X = full.drop(columns=["TARGET", "SK_ID_CURR"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    woe_fits = {}
    for col in MODEL_NUMERIC:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10)
    for col in MODEL_CATEGORICAL:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=True)
    ivs = {c: r["iv"] for c, r in woe_fits.items()}

    iv_ranked = rank_by_iv(woe_fits)
    iv_kept = [c for c in iv_ranked if ivs[c] >= IV_THRESHOLD]
    if verbose:
        print(f"total candidate features: {len(woe_fits)}")
        print("\nTop 20 by IV:")
        for c in iv_ranked[:20]:
            print(f"  {c:32s} IV={ivs[c]:.4f}  ({iv_strength(ivs[c])})")
        print(f"\nIV screen: {len(iv_kept)} of {len(woe_fits)} features have IV >= {IV_THRESHOLD}")

    def woe_encode(df, cols):
        return np.column_stack([transform_woe(df[c], woe_fits[c]).to_numpy(dtype=float) for c in cols])

    W_iv = woe_encode(X_train, iv_kept)
    corr_kept, corr_dropped = correlation_gate(W_iv, iv_kept, ivs)
    if verbose:
        print(f"correlation gate (|r| > {CORRELATION_THRESHOLD}): {len(corr_dropped)} dropped")
        for c, partner, r in corr_dropped:
            print(f"  dropped {c:28s} kept {partner:28s} r={r:+.3f}")

    W_corr = W_iv[:, [iv_kept.index(c) for c in corr_kept]]
    m, s = W_corr.mean(axis=0), W_corr.std(axis=0)
    s[s == 0] = 1.0
    kept_cols, sign_removed = eliminate_wrong_signs((W_corr - m) / s, y_train.to_numpy(), corr_kept)
    if verbose:
        print(f"sign gate: {len(sign_removed)} removed for a positive coefficient")
        for c, b in sign_removed:
            print(f"  removed {c:28s} coefficient={b:+.4f}")
        print(f"\nkeeping {len(kept_cols)} of {len(woe_fits)} candidate features")

    # Final fits. The from-scratch solver penalises weights differently from
    # scikit-learn's default, so a coefficient close to zero can change sign
    # between the two. The sign gate therefore finishes on the from-scratch model
    # that the scorecard actually uses: any feature it still gives a positive
    # coefficient is removed (largest first) and both models are refitted.
    while True:
        Xw_train = woe_encode(X_train, kept_cols)
        Xw_val = woe_encode(X_val, kept_cols)
        mean, std = Xw_train.mean(axis=0), Xw_train.std(axis=0)
        std[std == 0] = 1.0
        Xw_train_s = (Xw_train - mean) / std
        Xw_val_s = (Xw_val - mean) / std

        scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
        scratch.fit(Xw_train_s, y_train.to_numpy(), class_weight="balanced")
        wrong = [(c, float(b)) for c, b in zip(kept_cols, scratch.coef_) if b > 0]
        if not wrong:
            break
        victim = max(wrong, key=lambda t: (t[1], t[0]))
        sign_removed.append(victim)
        kept_cols = [c for c in kept_cols if c != victim[0]]
        if verbose:
            print(f"  removed {victim[0]:28s} coefficient={victim[1]:+.4f} (from-scratch fit)")
    if verbose:
        print(f"final feature count: {len(kept_cols)}")

    corr_final = np.corrcoef(Xw_train_s, rowvar=False)
    pairs = sorted(
        ((kept_cols[i], kept_cols[j], float(corr_final[i, j]))
         for i in range(len(kept_cols)) for j in range(i + 1, len(kept_cols))),
        key=lambda t: -abs(t[2]),
    )
    if verbose:
        print("\nmost correlated WoE pairs among the kept features:")
        for a, b, r in pairs[:5]:
            print(f"  {a:28s} <-> {b:28s}  r={r:+.3f}")

    sk = _sklearn_balanced(Xw_train_s, y_train)
    sk_val_pred = sk.predict_proba(Xw_val_s)[:, 1]
    scratch_val_pred = scratch.predict_proba(Xw_val_s)

    coef_diff = np.abs(sk.coef_.ravel() - scratch.coef_)
    pred_corr = np.corrcoef(sk_val_pred, scratch_val_pred)[0, 1]
    val_auc = auc_rank_sum(y_val, scratch_val_pred)

    if verbose:
        print(f"\nsklearn val AUC : {roc_auc_score(y_val, sk_val_pred):.4f}")
        print(f"scratch val AUC : {val_auc:.4f}")
        print(f"scratch val GINI: {gini(y_val, scratch_val_pred):.4f}")
        print(f"scratch val KS  : {ks_statistic(y_val, scratch_val_pred):.4f}")
        print(f"\nmax |coef diff| vs sklearn : {coef_diff.max():.4f}")
        print(f"prediction correlation     : {pred_corr:.6f}")
        print(f"from-scratch solver: {scratch.n_iter_} iterations, converged={scratch.converged_}, "
              f"last cost change={scratch.final_cost_change_:.2e}")

    return {
        "kept_cols": kept_cols,
        "iv_ranked": iv_ranked,
        "n_candidate_features": len(woe_fits),
        "n_categorical_features": len(MODEL_CATEGORICAL),
        "n_after_iv_screen": len(iv_kept),
        "n_after_correlation_gate": len(corr_kept),
        "correlation_gate_dropped": corr_dropped,
        "sign_gate_removed": sign_removed,
        "woe_fits": woe_fits,
        "corr_pairs": pairs[:10],
        "coef_diff_max": float(coef_diff.max()),
        "pred_corr": float(pred_corr),
        "val_auc": float(val_auc),
        "val_ks": float(ks_statistic(y_val, scratch_val_pred)),
        "val_pred": scratch_val_pred,
        "sk_coef": sk.coef_.ravel(),
        "model": scratch,
        "Xw_train": Xw_train,
        "Xw_val": Xw_val,
        "standardize_mean": mean,
        "standardize_std": std,
        "train_bad_rate": float(y_train.mean()),
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_val_raw": X_val,
        "y_val": y_val,
    }


def prohibited_basis_ablation(result: dict) -> dict:
    """What excluding sex and marital status costs in ranking power.

    Refits scikit-learn's class-balanced model on the final feature set with and
    without the two prohibited bases added back, on the same split, and reports
    both holdout AUCs. This is a documentation check only: the prohibited bases
    never enter the scorecard.
    """
    raw = pd.read_csv(DATA_DIR / "application_train.csv", usecols=PROHIBITED_BASES)
    X_train, X_val = result["X_train_raw"], result["X_val_raw"]
    y_train, y_val = result["y_train"], result["y_val"]
    tr_raw, va_raw = raw.loc[X_train.index], raw.loc[X_val.index]
    fits = {c: fit_woe(tr_raw[c], y_train, is_categorical=True) for c in PROHIBITED_BASES}
    extra_tr = np.column_stack([transform_woe(tr_raw[c], fits[c]).to_numpy(dtype=float) for c in PROHIBITED_BASES])
    extra_va = np.column_stack([transform_woe(va_raw[c], fits[c]).to_numpy(dtype=float) for c in PROHIBITED_BASES])

    def auc_for(Wt, Wv):
        m, s = Wt.mean(axis=0), Wt.std(axis=0)
        s[s == 0] = 1.0
        model = _sklearn_balanced((Wt - m) / s, y_train)
        return float(auc_rank_sum(y_val, model.predict_proba((Wv - m) / s)[:, 1]))

    return {
        "val_auc_sklearn_final_features": auc_for(result["Xw_train"], result["Xw_val"]),
        "val_auc_sklearn_with_prohibited_bases": auc_for(
            np.column_stack([result["Xw_train"], extra_tr]), np.column_stack([result["Xw_val"], extra_va])),
        "prohibited_bases_iv": {c: float(fits[c]["iv"]) for c in PROHIBITED_BASES},
    }


if __name__ == "__main__":
    fit_full_model()
