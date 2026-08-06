"""
Week 7 — AUC, GINI, KS, PSI, all from scratch.

AUC via the rank-sum (Mann-Whitney U) identity instead of numerically
integrating the ROC curve:
    AUC = (sum of ranks of the positive class - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
Average ranks are used for tied scores (matches how AUC is conventionally
defined under ties -- a strict rank without averaging would bias the result).

GINI = 2*AUC - 1 (this is the credit-scoring GINI, not the inequality-index
GINI -- same name, unrelated formula, worth being explicit about since it's a
common mix-up).

KS statistic: the largest gap between the cumulative-good and cumulative-bad
distributions as you sweep the score threshold. Equivalent to the two-sample
Kolmogorov-Smirnov statistic applied to the good-score and bad-score samples,
which is how it's validated below.

PSI: bins the *reference* distribution (e.g. train scores) into deciles, then
checks how much a second population's (val/test/OOT) share in each decile has
drifted. <0.1 stable, 0.1-0.25 moderate drift worth watching, >0.25 the model
likely needs review before being trusted on the new population.
"""
import numpy as np
import pandas as pd


def _average_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)

    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            tie_positions = order[i:j + 1]
            ranks[tie_positions] = ranks[tie_positions].mean()
        i = j + 1
    return ranks


def auc_rank_sum(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    ranks = _average_rank(y_score)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined with only one class present")

    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gini(y_true, y_score) -> float:
    return 2 * auc_rank_sum(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Max gap between the cumulative-bad and cumulative-good distributions.

    Ties matter here: the empirical CDF only moves at each *distinct* score
    value, jumping by the full count of ties at that value at once. Computing
    a running cumulative sum row-by-row after a plain sort (with ties broken
    arbitrarily by sort stability) evaluates the gap at intermediate,
    not-really-there points partway through a tied block, which can produce
    a spurious max that doesn't match the textbook two-sample KS statistic.
    Aggregating to one row per distinct score value first avoids that.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score, dtype=float)})
    n_bad = (df["y"] == 1).sum()
    n_good = (df["y"] == 0).sum()

    per_score = df.groupby("score")["y"].agg(bad="sum", n="count")
    per_score["good"] = per_score["n"] - per_score["bad"]
    per_score = per_score.sort_index(ascending=False)

    cum_bad = per_score["bad"].cumsum() / n_bad
    cum_good = per_score["good"].cumsum() / n_good
    return float((cum_bad - cum_good).abs().max())


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    floor = 1e-4
    expected_pct = np.where(expected_pct == 0, floor, expected_pct)
    actual_pct = np.where(actual_pct == 0, floor, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, n)
    score = rng.normal(0, 1, n) + y * 0.8  # score correlated with label, some ties from rounding
    score = np.round(score, 2)

    my_auc = auc_rank_sum(y, score)
    sk_auc = roc_auc_score(y, score)
    print(f"AUC   scratch={my_auc:.6f}  sklearn={sk_auc:.6f}  diff={abs(my_auc - sk_auc):.2e}")
    assert abs(my_auc - sk_auc) < 1e-9, "AUC should match sklearn to floating-point precision"

    my_gini = gini(y, score)
    print(f"GINI  scratch={my_gini:.6f}  (2*AUC-1)={2*sk_auc-1:.6f}")
    assert abs(my_gini - (2 * sk_auc - 1)) < 1e-9

    my_ks = ks_statistic(y, score)
    sp_ks = ks_2samp(score[y == 1], score[y == 0]).statistic
    print(f"KS    scratch={my_ks:.6f}  scipy ks_2samp={sp_ks:.6f}  diff={abs(my_ks - sp_ks):.2e}")
    assert abs(my_ks - sp_ks) < 1e-9, "KS should match scipy's two-sample KS statistic"

    # PSI sanity: identical distributions -> ~0; a shifted distribution -> clearly above the 0.25 flag
    same_psi = psi(score, score)
    shifted_psi = psi(score, score + 1.5)
    print(f"PSI   identical={same_psi:.4f}  shifted={shifted_psi:.4f}")
    assert same_psi < 1e-6, "PSI of a distribution against itself should be ~0"
    assert shifted_psi > 0.25, "a 1.5-sigma shift should clearly trip the PSI review threshold"

    print("all metric checks passed")
