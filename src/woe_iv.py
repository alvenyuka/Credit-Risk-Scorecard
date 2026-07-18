"""
Weight of Evidence / Information Value, implemented from scratch (Siddiqi, 2006).

WoE and IV are the two workhorse concepts in classic credit scoring, and
they answer two different questions:

  WoE (per bin): "within this bin, what's the log-odds of being good
                  vs. bad, relative to the population as a whole?"
  IV (per feature): "summed across all its bins, how much does this
                     feature separate good applicants from bad ones?"

The reason this combination is still standard in regulated lending, even
with gradient boosting freely available, is that WoE-transformed features
feed directly into a logistic regression whose coefficients stay
interpretable: a positive WoE bin pulls the score up, a negative one pulls
it down, and the size of the pull is visible per bin, not buried inside a
tree ensemble. See from_scratch_lr.py and README.md's "Why WoE/IV" section
for the rest of that argument.
"""

import numpy as np
import pandas as pd


def _woe_iv_table(bin_series: pd.Series, target: pd.Series) -> pd.DataFrame:
    """
    Given a binned/categorical feature and a binary target, compute WoE and IV per bin.

    "Good" here follows credit-scoring convention: good = non-default
    (TARGET == 0), bad = default (TARGET == 1). WoE = ln(%good / %bad) in a
    bin, so a bin with more good applicants than the population baseline
    gets a positive WoE, and a riskier-than-average bin gets a negative one.
    """
    df = pd.DataFrame({"bin": bin_series, "target": target})
    grouped = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    # Laplace-style smoothing avoids -inf/+inf WoE on empty bins. Without it,
    # a bin with zero defaults (common in small, high-quality bins) computes
    # ln(x/0) = inf, which would make that single bin dominate every
    # downstream IV comparison. eps=0.5 is the traditional credit-scoring
    # convention (a "half event" added to every bin), not a tuned parameter.
    eps = 0.5
    grouped["bad_rate"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
    grouped["good_rate"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))

    grouped["woe"] = np.log(grouped["good_rate"] / grouped["bad_rate"])
    # IV contribution per bin: (%good - %bad) * WoE. Bins that are both
    # skewed (big %good - %bad gap) AND have a big WoE swing contribute the
    # most; a bin that's 50/50 good/bad contributes ~0 regardless of size.
    grouped["iv_contribution"] = (grouped["good_rate"] - grouped["bad_rate"]) * grouped["woe"]

    return grouped


def calc_woe_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE table and total IV for one feature.

    Numeric features are quantile-binned (equal-frequency, not equal-width --
    equal-width bins would put almost all applicants in one or two bins for
    a right-skewed feature like AMT_INCOME_TOTAL); low-cardinality / object
    features are treated as already-categorical, one bin per category.
    """
    if pd.api.types.is_numeric_dtype(feature) and feature.nunique() > bins:
        binned = pd.qcut(feature, q=bins, duplicates="drop")
    else:
        binned = feature.astype("category")

    table = _woe_iv_table(binned, target)
    total_iv = table["iv_contribution"].sum()
    return table, total_iv


def iv_ranking(df: pd.DataFrame, target_col: str, feature_cols: list[str], bins: int = 10) -> pd.DataFrame:
    """
    Rank every feature in feature_cols by Information Value.

    Must be called on the train split only -- ranking on the full dataset
    (including val/test rows) would leak information about the holdout sets
    into feature selection, inflating every downstream metric. run_pipeline.py
    enforces this by only ever passing X_train into this function.
    """
    rows = []
    for col in feature_cols:
        try:
            valid = df[[col, target_col]].dropna()
            # Skip features that are almost entirely missing or constant --
            # qcut can't form meaningful bins from fewer than ~100 valid rows
            # or a single unique value, and the resulting IV would be noise,
            # not signal.
            if len(valid) < 100 or valid[col].nunique() < 2:
                continue
            _, iv = calc_woe_iv(valid[col], valid[target_col], bins=bins)
            rows.append({"feature": col, "iv": iv})
        except (ValueError, TypeError):
            # qcut raises ValueError on some pathological distributions
            # (e.g. a feature with too few distinct values to fill `bins`
            # quantiles even after the nunique check above); skipping rather
            # than crashing the whole ranking over one bad column.
            continue
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# Conventional IV interpretation bands from Siddiqi (2006), used throughout
# the credit-scoring industry as a rule of thumb for "is this feature worth
# including." An IV above 0.5 is flagged as suspicious rather than simply
# "excellent" because in practice a feature that predictive is more often a
# sign of leakage (the feature encodes something only knowable after the
# outcome) than a genuinely powerful predictor.
IV_STRENGTH_BANDS = [
    (0.02, "Not useful"),
    (0.10, "Weak"),
    (0.30, "Medium"),
    (0.50, "Strong"),
    (float("inf"), "Suspicious (check for leakage)"),
]


def iv_strength(iv: float) -> str:
    for threshold, label in IV_STRENGTH_BANDS:
        if iv < threshold:
            return label
    return IV_STRENGTH_BANDS[-1][1]
