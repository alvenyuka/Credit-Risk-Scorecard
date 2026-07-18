"""
Credit-risk evaluation metrics, implemented from scratch: AUC, GINI, KS, PSI, Brier, calibration.

Every function here is checked against its scikit-learn equivalent in
run_pipeline.py's Phase G (see output/results.json's diffs of 0.0 to
1.1e-16 -- floating-point noise, not implementation error). The point of
hand-coding metrics that scikit-learn already provides is the same as
from_scratch_lr.py: knowing exactly what "AUC" or "KS" means well enough to
derive it, not just well enough to call roc_auc_score.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Mann-Whitney U formulation of ROC-AUC: the probability a random positive
    scores higher than a random negative. O(n log n) via rank-sum, not O(n^2).

    The usual definition of AUC (area under the TPR-vs-FPR curve swept across
    every threshold) is O(n^2) to compute the naive way -- for every pair of
    (positive, negative) examples, check if the positive scored higher. The
    Mann-Whitney U statistic proves this is mathematically identical to a
    single rank-sum computation: rank every prediction, sum the ranks of the
    positives, subtract the minimum possible sum n_pos*(n_pos+1)/2, and
    divide by the number of pairs n_pos*n_neg. Same answer, one sort instead
    of a double loop.
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

    Sort every applicant by predicted score, then walk down that sorted list
    tracking what fraction of all "goods" and all "bads" have been passed so
    far. KS is the largest gap between those two running fractions -- the
    single score threshold where the model best separates the two
    populations. A model with KS near 0 can't separate them at any
    threshold; this dataset's LightGBM benchmark hits KS 0.4251 (see
    output/results.json), meaning at its best threshold it captures ~42.5
    percentage points more of the bad population than the good one.
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
    """
    Mean squared error between predicted probabilities and actual binary outcomes.

    Unlike AUC/GINI/KS (which only care about rank-ordering), Brier score
    punishes miscalibration directly -- a model that says "80% risk" for
    someone who never defaults is penalized here even if its rank-ordering
    of applicants is perfect. This is exactly the gap the README's
    calibration finding exploits: LightGBM's AUC beats the from-scratch LR's,
    but LightGBM's Brier score (0.1782) is worse than the LR's (0.0681),
    because LightGBM's raw probabilities are badly under-calibrated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI between two score distributions (e.g. train vs. test, or train vs. a
    later production window). Bin edges are the expected distribution's
    deciles; PSI < 0.1 = stable, 0.1-0.25 = moderate shift, > 0.25 = major shift.

    This is the metric a deployed scorecard gets monitored with long after
    training: bin the "expected" (training-time) score distribution into
    deciles, then check what fraction of a new population falls into each of
    those same bins. If the population hasn't shifted, each bin should still
    hold ~10% of applicants. PSI sums (actual% - expected%) * ln(actual% /
    expected%) across bins -- structurally the same cross-entropy-flavoured
    formula as WoE/IV in woe_iv.py, applied to population drift instead of
    target separation. run_pipeline.py uses this only as a sanity check
    (splitting the test set in half and comparing it to itself, PSI ~0.0007,
    i.e. "a random split of the same data looks stable," which it should) --
    a real deployment would compare a live scoring window against the
    original training population instead.
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

    # Clip to a small floor rather than allowing zero -- an empty bin would
    # otherwise make ln(0) = -inf and silently poison the whole PSI sum.
    exp_pct = np.clip(exp_counts / len(expected), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Mean predicted probability vs. observed event rate, per quantile bin.

    A perfectly calibrated model's points would sit exactly on the y=x
    diagonal: "of the applicants I said were 20% likely to default, 20% of
    them actually did." figs/calibration_curve.png plots this for both
    models -- the from-scratch LR tracks the diagonal closely, LightGBM's
    raw output does not (see README's "uncomfortable finding").
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        count=("y_true", "size"),
    ).reset_index(drop=True)
    return out
