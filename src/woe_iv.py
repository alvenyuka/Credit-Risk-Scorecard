"""
Weight of Evidence and Information Value, implemented from first principles.

Convention used (the standard credit-scoring one):
    WoE_bin = ln( dist_good_bin / dist_bad_bin )
where dist_good_bin = (# non-defaults in bin) / (total non-defaults),
      dist_bad_bin  = (# defaults in bin) / (total defaults).

A positive WoE means a bin is safer than average (over-represented among
goods); negative means riskier. IV is the WoE-weighted gap between the two
distributions, summed across bins, a standard predictive-power score:
  <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious
  (often a leak). That last bucket is a useful red flag, not just
  a good score.

Binning is fit once on train (quantile edges for numeric columns, the raw
category set for categorical ones) and then *applied* to any other split.
Computing fresh quantiles per split would leak information about that split's
own label distribution into its own bins.

Zero-count bins would send WoE to +/-inf (log of 0). Laplace-style smoothing
(epsilon added to both good and bad counts) avoids that without distorting
well-populated bins much.

The WoE table of a continuous feature is kept in bin order (lowest edge first,
Missing last), so it can be read and checked for monotonicity directly;
monotonicity_report summarises that check per feature.
"""
import numpy as np
import pandas as pd

MISSING_LABEL = "Missing"


def fit_continuous_bins(x: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Bin edges at the deciles of the non-missing values, with open first and last bins."""
    # Always float64: a low-cardinality integer column (for example a 0/1 flag)
    # can make np.quantile return an int array, which cannot hold -inf.
    finite = x.dropna().astype(float)
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1))).astype(float)
    if len(edges) < 3:
        edges = np.array([finite.min(), finite.max()], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_monotone_bins(x: pd.Series, y: pd.Series, n_bins: int = 10, zero_share: float = 0.05) -> np.ndarray:
    """Coarse classing: start from fine bins, then merge neighbours until the default rate moves one way.

    1. Fine bins are the deciles of the non-missing values. A column that is never negative and is zero
       for at least `zero_share` of its values (the delinquency counts, for example) gets a bin of its own
       for 0 and deciles of its positive values, so its tail is not lost to repeated quantile edges.
    2. The direction is the overall trend of the default rate across the fine bins (weighted by size).
    3. Adjacent bins that go against that direction are pooled (the pool-adjacent-violators algorithm)
       until the default rate is monotonic.

    Missing values are not binned here; fit_woe gives them their own bin. Returns the merged edges.
    """
    finite = x.dropna().astype(float)
    target = pd.Series(np.asarray(y), index=x.index).loc[finite.index].to_numpy(dtype=float)
    if len(finite) == 0:
        return fit_continuous_bins(x, n_bins)
    if (finite >= 0).all() and (finite == 0).mean() >= zero_share and (finite > 0).any():
        inner = np.unique(np.quantile(finite[finite > 0], np.linspace(0, 1, n_bins + 1))[1:-1])
        edges = np.concatenate([[-np.inf, 0.0], inner[inner > 0], [np.inf]]).astype(float)
    else:
        edges = fit_continuous_bins(finite, n_bins)
    if len(edges) < 3:
        return edges

    codes = pd.cut(finite, bins=edges, labels=False, include_lowest=True).to_numpy()
    n = np.bincount(codes, minlength=len(edges) - 1).astype(float)
    bad = np.bincount(codes, weights=target, minlength=len(edges) - 1)
    keep = n > 0
    uppers = edges[1:][keep]
    n, bad = n[keep], bad[keep]

    rate = bad / n
    idx = np.arange(len(n))
    trend = np.sum(n * (idx - np.average(idx, weights=n)) * (rate - np.average(rate, weights=n)))
    direction = 1.0 if trend >= 0 else -1.0

    blocks = []  # each block: [count, defaults, upper edge]
    for count, defaults, upper in zip(n, bad, uppers):
        blocks.append([count, defaults, upper])
        while len(blocks) >= 2 and direction * (blocks[-2][1] / blocks[-2][0] - blocks[-1][1] / blocks[-1][0]) > 0:
            last = blocks.pop()
            blocks[-1] = [blocks[-1][0] + last[0], blocks[-1][1] + last[1], last[2]]
    merged = np.array([-np.inf] + [b[2] for b in blocks[:-1]] + [np.inf], dtype=float)
    return merged


def apply_continuous_bins(x: pd.Series, edges: np.ndarray) -> pd.Series:
    """Interval label of each value under `edges`; missing values get the Missing label."""
    labels = pd.cut(x, bins=edges, include_lowest=True).astype(str)
    labels = labels.where(~x.isna(), MISSING_LABEL)
    return labels


def _bin_labels(x: pd.Series, is_categorical: bool, edges) -> pd.Series:
    """Bin label of each value: the category itself, or its interval for a numeric feature."""
    if is_categorical:
        labels = x.astype(str)
        labels = labels.where(~x.isna(), MISSING_LABEL)
        return labels
    return apply_continuous_bins(x, edges)


def fit_woe(x: pd.Series, y: pd.Series, is_categorical: bool = False,
            n_bins: int = 10, epsilon: float = 0.5, binning: str = "decile") -> dict:
    """Fit bins and WoE for one feature on training data.

    Returns a dict with the bin edges (None for a categorical feature), the table of
    counts, defaults, WoE and IV contribution per bin, and the feature's total IV.
    epsilon is added to the good and bad counts so an empty bin never gives log(0).
    binning="decile" uses decile bins (the published scorecard); binning="monotone" uses
    fit_monotone_bins, so the default rate, and with it the WoE, moves in one direction.
    """
    if binning not in ("decile", "monotone"):
        raise ValueError(f"unknown binning {binning!r}")
    if is_categorical:
        edges = None
    elif binning == "monotone":
        edges = fit_monotone_bins(x, y, n_bins)
    else:
        edges = fit_continuous_bins(x, n_bins)
    bins = _bin_labels(x, is_categorical, edges)

    df = pd.DataFrame({"bin": bins, "y": y.values})
    total_good = int((df["y"] == 0).sum())
    total_bad = int((df["y"] == 1).sum())

    grouped = df.groupby("bin", observed=True)["y"].agg(n="count", bad="sum")
    if not is_categorical:
        grouped = grouped.loc[_ordered_labels(grouped.index, edges)]
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


def _ordered_labels(labels, edges) -> list:
    """Continuous bin labels in edge order, with Missing last."""
    interval_labels = pd.cut(pd.Series(edges[1:]), bins=edges, include_lowest=True).astype(str).tolist()
    present = set(labels)
    ordered = [lab for lab in dict.fromkeys(interval_labels) if lab in present]
    ordered += sorted(lab for lab in present if lab not in ordered)
    return ordered


def is_monotonic_woe(fit_result: dict) -> bool | None:
    """True when WoE moves in one direction across the ordered numeric bins.

    The Missing bin is excluded, since it has no position on the scale.
    Returns None for categorical features, which have no natural order.
    """
    if fit_result["is_categorical"]:
        return None
    woe = fit_result["table"]["woe"].drop(MISSING_LABEL, errors="ignore").to_numpy()
    if len(woe) < 2:
        return True
    diffs = np.diff(woe)
    return bool((diffs >= 0).all() or (diffs <= 0).all())


def monotonicity_report(woe_fits: dict, cols: list) -> pd.DataFrame:
    """One row per feature: type, number of bins, IV and whether WoE is monotonic."""
    rows = []
    for c in cols:
        fit = woe_fits[c]
        rows.append({
            "feature": c,
            "type": "categorical" if fit["is_categorical"] else "numeric",
            "n_bins": int(len(fit["table"])),
            "iv": fit["iv"],
            "monotonic_woe": is_monotonic_woe(fit),
        })
    return pd.DataFrame(rows)


def transform_woe(x: pd.Series, fit_result: dict) -> pd.Series:
    """Replace each value with the WoE of its bin; a category unseen in training gets 0 (neutral)."""
    bins = _bin_labels(x, fit_result["is_categorical"], fit_result["edges"])
    mapped = bins.map(fit_result["table"]["woe"])
    return mapped.fillna(0.0)  # unseen bin/category on a new split -> neutral


def iv_strength(iv: float) -> str:
    """Conventional reading of an information value: useless, weak, medium, strong or suspicious."""
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
