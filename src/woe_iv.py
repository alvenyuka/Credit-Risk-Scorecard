"""
Week 7 — Weight of Evidence / Information Value, from scratch.

Convention used (the standard credit-scoring one):
    WoE_bin = ln( dist_good_bin / dist_bad_bin )
where dist_good_bin = (# non-defaults in bin) / (total non-defaults),
      dist_bad_bin  = (# defaults in bin) / (total defaults).

A positive WoE means a bin is safer than average (over-represented among
goods); negative means riskier. IV is the WoE-weighted gap between the two
distributions, summed across bins — a standard predictive-power score:
  <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious
  (often a leak) — that last bucket is a genuinely useful red flag, not just
  a good score.

Binning is fit once on train (quantile edges for numeric columns, the raw
category set for categorical ones) and then *applied* to any other split —
computing fresh quantiles per split would leak information about that split's
own label distribution into its own bins.

Zero-count bins would send WoE to +/-inf (log of 0). Laplace-style smoothing
(epsilon added to both good and bad counts) avoids that without distorting
well-populated bins much.
"""
import numpy as np
import pandas as pd

MISSING_LABEL = "Missing"


def fit_continuous_bins(x: pd.Series, n_bins: int = 10) -> np.ndarray:
    finite = x.dropna()
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array([finite.min(), finite.max()])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_continuous_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = pd.cut(x, bins=edges, include_lowest=True).astype(str)
    labels = labels.where(~x.isna(), MISSING_LABEL)
    return labels


def _bin_labels(x: pd.Series, is_categorical: bool, edges) -> pd.Series:
    if is_categorical:
        labels = x.astype(str)
        labels = labels.where(~x.isna(), MISSING_LABEL)
        return labels
    return apply_continuous_bins(x, edges)


def fit_woe(x: pd.Series, y: pd.Series, is_categorical: bool = False,
            n_bins: int = 10, epsilon: float = 0.5) -> dict:
    edges = None if is_categorical else fit_continuous_bins(x, n_bins)
    bins = _bin_labels(x, is_categorical, edges)

    df = pd.DataFrame({"bin": bins, "y": y.values})
    total_good = int((df["y"] == 0).sum())
    total_bad = int((df["y"] == 1).sum())

    grouped = df.groupby("bin", observed=True)["y"].agg(n="count", bad="sum")
    grouped["good"] = grouped["n"] - grouped["bad"]

    n_bins_actual = len(grouped)
    dist_good = (grouped["good"] + epsilon) / (total_good + epsilon * n_bins_actual)
    dist_bad = (grouped["bad"] + epsilon) / (total_bad + epsilon * n_bins_actual)
    grouped["woe"] = np.log(dist_good / dist_bad)
    grouped["iv_contrib"] = (dist_good - dist_bad) * grouped["woe"]

    return {
        "edges": edges,
        "is_categorical": is_categorical,
        "table": grouped,
        "iv": float(grouped["iv_contrib"].sum()),
    }


def transform_woe(x: pd.Series, fit_result: dict) -> pd.Series:
    bins = _bin_labels(x, fit_result["is_categorical"], fit_result["edges"])
    mapped = bins.map(fit_result["table"]["woe"])
    return mapped.fillna(0.0)  # unseen bin/category on a new split -> neutral


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong (check for leakage)"


if __name__ == "__main__":
    # Sanity checks before trusting this on real data:
    # 1. a feature perfectly separating the classes should have very high IV.
    # 2. a feature with no relationship to the target should have IV ~ 0.
    rng = np.random.default_rng(0)
    n = 20000
    y = pd.Series(rng.integers(0, 2, n))

    perfect = pd.Series(y.values + rng.normal(0, 0.01, n))  # near-perfect separator
    noise = pd.Series(rng.normal(0, 1, n))                  # unrelated to y

    fit_perfect = fit_woe(perfect, y, n_bins=10)
    fit_noise = fit_woe(noise, y, n_bins=10)

    print(f"near-perfect separator IV: {fit_perfect['iv']:.3f} ({iv_strength(fit_perfect['iv'])})")
    print(f"pure noise IV:             {fit_noise['iv']:.3f} ({iv_strength(fit_noise['iv'])})")

    assert fit_perfect["iv"] > 1.0, "a near-perfect separator should have a very high IV"
    assert fit_noise["iv"] < 0.02, "pure noise should score as useless"
    print("sanity checks passed")
