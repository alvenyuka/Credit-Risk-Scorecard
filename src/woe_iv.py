"""Weight of Evidence / Information Value, implemented from scratch (Siddiqi, 2006)."""

import numpy as np
import pandas as pd


def _woe_iv_table(bin_series: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Given a binned/categorical feature and a binary target, compute WoE and IV per bin."""
    df = pd.DataFrame({"bin": bin_series, "target": target})
    grouped = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    # Laplace-style smoothing avoids -inf/+inf WoE on empty bins.
    eps = 0.5
    grouped["bad_rate"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
    grouped["good_rate"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))

    grouped["woe"] = np.log(grouped["good_rate"] / grouped["bad_rate"])
    grouped["iv_contribution"] = (grouped["good_rate"] - grouped["bad_rate"]) * grouped["woe"]

    return grouped


def calc_woe_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE table and total IV for one feature.

    Numeric features are quantile-binned; low-cardinality / object features
    are treated as already-categorical.
    """
    if pd.api.types.is_numeric_dtype(feature) and feature.nunique() > bins:
        binned = pd.qcut(feature, q=bins, duplicates="drop")
    else:
        binned = feature.astype("category")

    table = _woe_iv_table(binned, target)
    total_iv = table["iv_contribution"].sum()
    return table, total_iv


def iv_ranking(df: pd.DataFrame, target_col: str, feature_cols: list[str], bins: int = 10) -> pd.DataFrame:
    """Rank every feature in feature_cols by Information Value."""
    rows = []
    for col in feature_cols:
        try:
            valid = df[[col, target_col]].dropna()
            if len(valid) < 100 or valid[col].nunique() < 2:
                continue
            _, iv = calc_woe_iv(valid[col], valid[target_col], bins=bins)
            rows.append({"feature": col, "iv": iv})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


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
