"""Credit-risk evaluation metrics, implemented from scratch: AUC, GINI, KS, PSI, Brier, calibration."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Mann-Whitney U formulation of ROC-AUC: the probability a random positive
    scores higher than a random negative. O(n log n) via rank-sum, not O(n^2).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc requires both classes present")

    ranks = rankdata(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """GINI coefficient = 2*AUC - 1, the standard credit-scoring discrimination metric."""
    return 2 * roc_auc(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of scores for the good and bad populations.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)
    y_sorted = y_true[order]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg

    return float(np.max(np.abs(cum_pos - cum_neg)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and actual binary outcomes."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI between two score distributions (e.g. train vs. test, or train vs. a
    later production window). Bin edges are the expected distribution's
    deciles; PSI < 0.1 = stable, 0.1-0.25 = moderate shift, > 0.25 = major shift.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    bin_edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bin_edges) < 3:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts = np.histogram(expected, bins=bin_edges)[0]
    act_counts = np.histogram(actual, bins=bin_edges)[0]

    exp_pct = np.clip(exp_counts / len(expected), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs. observed event rate, per quantile bin."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index(drop=True)
    return out
