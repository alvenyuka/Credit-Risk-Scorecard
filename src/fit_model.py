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
    if cols is None:
        cols = list(woe_fits)

    # Sorting (negative IV, name) tuples puts the highest IV first
    # and orders equal IVs alphabetically.
    pairs = []
    for col in cols:
        pairs.append((-woe_fits[col]["iv"], col))
    pairs.sort()

    ranked = []
    for neg_iv, col in pairs:
        ranked.append(col)
    return ranked


def correlation_gate(W: np.ndarray, cols: list, ivs: dict, threshold: float = CORRELATION_THRESHOLD):
    """Drop one feature of every pair whose WoE columns correlate above `threshold`.

    Features are visited in descending IV (ties by name); a feature is kept only if
    its |r| with every feature already kept is at most `threshold`, so of each
    offending pair the higher-IV feature survives.

    Returns (kept, dropped) where dropped lists (feature, kept_partner, r).
    """
    # Column position of each feature inside W, to find its row in the correlation matrix.
    position = {}
    for i, col in enumerate(cols):
        position[col] = i
    corr = np.corrcoef(W, rowvar=False)

    # Visit the features highest IV first (ties by name).
    pairs = []
    for col in cols:
        pairs.append((-ivs[col], col))
    pairs.sort()

    kept = []
    dropped = []
    for neg_iv, col in pairs:
        # Find the first already-kept feature this one is too correlated with.
        partner = None
        for other in kept:
            r = corr[position[col], position[other]]
            if abs(r) > threshold:
                partner = other
                break

        if partner is None:
            kept.append(col)
        else:
            r = float(corr[position[col], position[partner]])
            dropped.append((col, partner, r))
    return kept, dropped


def _sklearn_balanced(Xs: np.ndarray, y) -> LogisticRegression:
    """scikit-learn's class-balanced logistic regression, fitted on standardised WoE columns."""
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, y)


def column_mean_std(W: np.ndarray):
    """Mean and standard deviation of each column. A constant column gets std 1 so we never divide by 0."""
    mean = W.mean(axis=0)
    std = W.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def find_worst_positive(names: list, coefs) -> tuple:
    """The feature with the largest positive coefficient, as (name, coefficient).

    Returns (None, 0.0) when no coefficient is positive. If two features share the
    same largest coefficient, the one with the later name (alphabetically) is
    picked, so the result never depends on the column order.
    """
    largest_name, largest_coef = None, 0.0
    for name, coef in zip(names, coefs):
        coef = float(coef)
        if coef <= 0:
            continue
        if coef > largest_coef:
            largest_name, largest_coef = name, coef
        elif coef == largest_coef and name > largest_name:
            largest_name, largest_coef = name, coef
    return largest_name, largest_coef


def eliminate_wrong_signs(Xs: np.ndarray, y, cols: list, fit=_sklearn_balanced):
    """Remove features with positive coefficients one at a time until none remain.

    Xs holds the standardised WoE columns in the order of `cols`. Each round refits
    on the surviving columns and removes the feature with the largest positive
    coefficient (ties by name). Returns (kept, removed) where removed lists
    (feature, coefficient at removal) in removal order.
    """
    kept = list(cols)
    position = {}
    for i, col in enumerate(cols):
        position[col] = i

    removed = []
    while len(kept) > 0:
        # Refit on the columns that are still in.
        column_numbers = []
        for col in kept:
            column_numbers.append(position[col])
        model = fit(Xs[:, column_numbers], y)
        coefs = np.ravel(model.coef_)

        worst_name, worst_coef = find_worst_positive(kept, coefs)
        if worst_name is None:
            break
        removed.append((worst_name, worst_coef))
        kept.remove(worst_name)
    return kept, removed


def woe_encode(df: pd.DataFrame, cols: list, woe_fits: dict) -> np.ndarray:
    """Matrix with one column per feature in `cols`, each value replaced by its bin's WoE."""
    columns = []
    for col in cols:
        woe_values = transform_woe(df[col], woe_fits[col])
        columns.append(woe_values.to_numpy(dtype=float))
    return np.column_stack(columns)


def split_data(full: pd.DataFrame):
    """80/20 stratified split. Returns X_train, X_val, y_train, y_val."""
    y = full["TARGET"]
    X = full.drop(columns=["TARGET", "SK_ID_CURR"])
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def fit_woe_bins(X_train: pd.DataFrame, y_train, binning: str) -> dict:
    """Fit the bins and WoE of every candidate feature on the training split only."""
    woe_fits = {}
    for col in MODEL_NUMERIC:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=False, n_bins=10, binning=binning)
    for col in MODEL_CATEGORICAL:
        woe_fits[col] = fit_woe(X_train[col], y_train, is_categorical=True)
    return woe_fits


def apply_iv_gate(woe_fits: dict, verbose: bool):
    """Gate 1: keep the features with IV at or above IV_THRESHOLD. Returns (iv_ranked, iv_kept)."""
    iv_ranked = rank_by_iv(woe_fits)
    iv_kept = []
    for col in iv_ranked:
        if woe_fits[col]["iv"] >= IV_THRESHOLD:
            iv_kept.append(col)

    if verbose:
        print(f"total candidate features: {len(woe_fits)}")
        print("\nTop 20 by IV:")
        for col in iv_ranked[:20]:
            iv = woe_fits[col]["iv"]
            print(f"  {col:32s} IV={iv:.4f}  ({iv_strength(iv)})")
        print(f"\nIV screen: {len(iv_kept)} of {len(woe_fits)} features have IV >= {IV_THRESHOLD}")
    return iv_ranked, iv_kept


def apply_correlation_gate(X_train, woe_fits: dict, iv_kept: list, verbose: bool):
    """Gate 2: of two near-duplicate WoE columns keep the higher-IV one.

    Returns (corr_kept, corr_dropped, W_corr), where W_corr is the training WoE matrix
    of the surviving features.
    """
    ivs = {}
    for col in woe_fits:
        ivs[col] = woe_fits[col]["iv"]

    W_iv = woe_encode(X_train, iv_kept, woe_fits)
    corr_kept, corr_dropped = correlation_gate(W_iv, iv_kept, ivs)

    if verbose:
        print(f"correlation gate (|r| > {CORRELATION_THRESHOLD}): {len(corr_dropped)} dropped")
        for col, partner, r in corr_dropped:
            print(f"  dropped {col:28s} kept {partner:28s} r={r:+.3f}")

    # Keep only the columns of W_iv that survived the gate.
    surviving_columns = []
    for col in corr_kept:
        surviving_columns.append(iv_kept.index(col))
    W_corr = W_iv[:, surviving_columns]
    return corr_kept, corr_dropped, W_corr


def apply_sign_gate(W_corr: np.ndarray, y_train, corr_kept: list, n_candidates: int, verbose: bool):
    """Gate 3, first pass (scikit-learn): drop positive-coefficient features. Returns (kept_cols, sign_removed)."""
    mean, std = column_mean_std(W_corr)
    W_corr_scaled = (W_corr - mean) / std
    kept_cols, sign_removed = eliminate_wrong_signs(W_corr_scaled, y_train.to_numpy(), corr_kept)

    if verbose:
        print(f"sign gate: {len(sign_removed)} removed for a positive coefficient")
        for col, coef in sign_removed:
            print(f"  removed {col:28s} coefficient={coef:+.4f}")
        print(f"\nkeeping {len(kept_cols)} of {n_candidates} candidate features")
    return kept_cols, sign_removed


def fit_scratch_model(X_train, X_val, y_train, kept_cols: list, woe_fits: dict) -> dict:
    """WoE-encode, standardise on the training split, and fit the from-scratch logistic regression."""
    Xw_train = woe_encode(X_train, kept_cols, woe_fits)
    Xw_val = woe_encode(X_val, kept_cols, woe_fits)

    # The holdout is scaled with the training mean and std, never its own.
    mean, std = column_mean_std(Xw_train)
    Xw_train_s = (Xw_train - mean) / std
    Xw_val_s = (Xw_val - mean) / std

    scratch = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    scratch.fit(Xw_train_s, y_train.to_numpy(), class_weight="balanced")
    return {
        "Xw_train": Xw_train, "Xw_val": Xw_val,
        "mean": mean, "std": std,
        "Xw_train_s": Xw_train_s, "Xw_val_s": Xw_val_s,
        "scratch": scratch,
    }


def apply_final_sign_gate(X_train, X_val, y_train, kept_cols: list, sign_removed: list, woe_fits: dict, verbose: bool):
    """Gate 3, final pass: repeat the sign check on the from-scratch model the scorecard uses.

    The from-scratch solver penalises weights differently from scikit-learn's
    default, so a coefficient close to zero can change sign between the two. Any
    feature it still gives a positive coefficient is removed (largest first) and the
    model is refitted. Returns (kept_cols, fitted, sign_removed), where fitted is the
    output of fit_scratch_model for the final feature set and sign_removed is the list
    passed in plus any features removed here.
    """
    sign_removed = list(sign_removed)  # a copy, so the caller's list is not changed
    while True:
        fitted = fit_scratch_model(X_train, X_val, y_train, kept_cols, woe_fits)

        worst_name, worst_coef = find_worst_positive(kept_cols, fitted["scratch"].coef_)
        if worst_name is None:
            break

        sign_removed.append((worst_name, worst_coef))
        kept_cols = [col for col in kept_cols if col != worst_name]
        if verbose:
            print(f"  removed {worst_name:28s} coefficient={worst_coef:+.4f} (from-scratch fit)")

    if verbose:
        print(f"final feature count: {len(kept_cols)}")
    return kept_cols, fitted, sign_removed


def negative_abs_r(pair):
    """Sort key for (name_a, name_b, r) pairs: largest |r| first when sorted ascending."""
    return -abs(pair[2])


def most_correlated_pairs(Xw_train_s: np.ndarray, kept_cols: list) -> list:
    """Every pair of kept features as (name_a, name_b, r), largest |r| first."""
    corr_final = np.corrcoef(Xw_train_s, rowvar=False)
    pairs = []
    for i in range(len(kept_cols)):
        for j in range(i + 1, len(kept_cols)):
            pairs.append((kept_cols[i], kept_cols[j], float(corr_final[i, j])))

    pairs.sort(key=negative_abs_r)  # stable, so equal |r| keep their original order
    return pairs


def compare_with_sklearn(fitted: dict, y_train, y_val, verbose: bool) -> dict:
    """Fit scikit-learn on the same data and compare coefficients, predictions and holdout metrics."""
    scratch = fitted["scratch"]
    sk = _sklearn_balanced(fitted["Xw_train_s"], y_train)
    sk_val_pred = sk.predict_proba(fitted["Xw_val_s"])[:, 1]
    scratch_val_pred = scratch.predict_proba(fitted["Xw_val_s"])

    coef_diff = np.abs(sk.coef_.ravel() - scratch.coef_)
    pred_corr = np.corrcoef(sk_val_pred, scratch_val_pred)[0, 1]
    val_auc = auc_rank_sum(y_val, scratch_val_pred)
    val_ks = ks_statistic(y_val, scratch_val_pred)

    if verbose:
        print(f"\nsklearn val AUC : {roc_auc_score(y_val, sk_val_pred):.4f}")
        print(f"scratch val AUC : {val_auc:.4f}")
        print(f"scratch val GINI: {gini(y_val, scratch_val_pred):.4f}")
        print(f"scratch val KS  : {val_ks:.4f}")
        print(f"\nmax |coef diff| vs sklearn : {coef_diff.max():.4f}")
        print(f"prediction correlation     : {pred_corr:.6f}")
        print(f"from-scratch solver: {scratch.n_iter_} iterations, converged={scratch.converged_}, "
              f"last cost change={scratch.final_cost_change_:.2e}")

    return {
        "sk_coef": sk.coef_.ravel(),
        "coef_diff_max": float(coef_diff.max()),
        "pred_corr": float(pred_corr),
        "val_auc": float(val_auc),
        "val_ks": float(val_ks),
        "val_pred": scratch_val_pred,
    }


def fit_full_model(full: pd.DataFrame = None, verbose: bool = True, binning: str = "decile") -> dict:
    """Fit the scorecard model on the 80/20 split and return everything later steps need.

    Fits WoE on the training split, applies the IV, correlation and sign gates,
    fits the from-scratch logistic regression (repeating the sign check on it) and
    cross-checks it against scikit-learn. Returns a dict with the kept features, the
    gate decisions, the WoE fits, the fitted model, the WoE matrices, holdout AUC and
    KS, and the raw train and holdout splits. `binning` is passed to fit_woe for the
    numeric features ("decile" for the published scorecard, "monotone" for coarse classing).
    """
    if full is None:
        full = build_full_feature_table()

    # Step 1: split, then learn the bins on the training rows only.
    X_train, X_val, y_train, y_val = split_data(full)
    woe_fits = fit_woe_bins(X_train, y_train, binning)

    # Step 2: the three gates.
    iv_ranked, iv_kept = apply_iv_gate(woe_fits, verbose)
    corr_kept, corr_dropped, W_corr = apply_correlation_gate(X_train, woe_fits, iv_kept, verbose)
    kept_cols, sign_removed = apply_sign_gate(W_corr, y_train, corr_kept, len(woe_fits), verbose)
    kept_cols, fitted, sign_removed = apply_final_sign_gate(
        X_train, X_val, y_train, kept_cols, sign_removed, woe_fits, verbose)

    # Step 3: report the correlation left among the kept features, then compare with scikit-learn.
    pairs = most_correlated_pairs(fitted["Xw_train_s"], kept_cols)
    if verbose:
        print("\nmost correlated WoE pairs among the kept features:")
        for a, b, r in pairs[:5]:
            print(f"  {a:28s} <-> {b:28s}  r={r:+.3f}")
    comparison = compare_with_sklearn(fitted, y_train, y_val, verbose)

    return {
        "binning": binning,
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
        "coef_diff_max": comparison["coef_diff_max"],
        "pred_corr": comparison["pred_corr"],
        "val_auc": comparison["val_auc"],
        "val_ks": comparison["val_ks"],
        "val_pred": comparison["val_pred"],
        "sk_coef": comparison["sk_coef"],
        "model": fitted["scratch"],
        "Xw_train": fitted["Xw_train"],
        "Xw_val": fitted["Xw_val"],
        "standardize_mean": fitted["mean"],
        "standardize_std": fitted["std"],
        "train_bad_rate": float(y_train.mean()),
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_val_raw": X_val,
        "y_val": y_val,
    }


def holdout_auc_sklearn(W_train: np.ndarray, W_val: np.ndarray, y_train, y_val) -> float:
    """Holdout AUC of a balanced scikit-learn fit on the given train and holdout WoE matrices."""
    mean, std = column_mean_std(W_train)
    model = _sklearn_balanced((W_train - mean) / std, y_train)
    val_pred = model.predict_proba((W_val - mean) / std)[:, 1]
    return float(auc_rank_sum(y_val, val_pred))


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
    train_raw = raw.loc[X_train.index]
    val_raw = raw.loc[X_val.index]

    fits = {}
    for col in PROHIBITED_BASES:
        fits[col] = fit_woe(train_raw[col], y_train, is_categorical=True)
    extra_train = woe_encode(train_raw, PROHIBITED_BASES, fits)
    extra_val = woe_encode(val_raw, PROHIBITED_BASES, fits)

    auc_final_features = holdout_auc_sklearn(result["Xw_train"], result["Xw_val"], y_train, y_val)
    auc_with_prohibited = holdout_auc_sklearn(
        np.column_stack([result["Xw_train"], extra_train]),
        np.column_stack([result["Xw_val"], extra_val]),
        y_train, y_val,
    )

    prohibited_iv = {}
    for col in PROHIBITED_BASES:
        prohibited_iv[col] = float(fits[col]["iv"])

    return {
        "val_auc_sklearn_final_features": auc_final_features,
        "val_auc_sklearn_with_prohibited_bases": auc_with_prohibited,
        "prohibited_bases_iv": prohibited_iv,
    }


if __name__ == "__main__":
    fit_full_model()
