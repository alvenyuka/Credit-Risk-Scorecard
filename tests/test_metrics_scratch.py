"""Tests for the hand-coded metrics in src/metrics_scratch.py.

These are not decoration. Every metric here was written from scratch and the
README's headline numbers depend on them being right, so each one is checked
against the library implementation it is supposed to reproduce:

  auc_rank_sum -> sklearn.metrics.roc_auc_score
  ks_statistic -> scipy.stats.ks_2samp
  gini         -> 2 * sklearn.metrics.roc_auc_score - 1
  psi          -> its own definition

The gini test deliberately goes through sklearn rather than through this
module's own auc_rank_sum. Comparing gini(y, s) against 2 * auc_rank_sum(y, s)
- 1 restates the function body and cannot fail for any implementation.

test_ks_handles_heavy_ties is a regression test for a real bug found during the
rebuild: computing the cumulative gap row-by-row after a plain sort evaluates it
partway through a block of tied scores, at points that do not exist in the
empirical CDF, which produces a spurious maximum. Aggregating to one row per
distinct score first is the fix. That bug is invisible on continuous scores and
appears the moment scores are rounded or bucketed, which is exactly what a
scorecard does.
"""
import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from metrics_scratch import auc_rank_sum, gini, ks_statistic, psi

SEED = 42


@pytest.fixture
def scores_and_labels():
    """An imbalanced, separable-but-noisy set, shaped like real credit data."""
    rng = np.random.default_rng(SEED)
    n = 4000
    y = rng.binomial(1, 0.08, size=n)                 # ~8% default rate
    score = rng.normal(loc=y * 0.9, scale=1.0)        # bads score higher
    return y, score


def test_auc_matches_sklearn(scores_and_labels):
    y, score = scores_and_labels
    assert auc_rank_sum(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_auc_matches_sklearn_with_ties(scores_and_labels):
    """Rounding creates large tied blocks. The rank-sum form must still agree."""
    y, score = scores_and_labels
    tied = np.round(score, 1)
    assert auc_rank_sum(y, tied) == pytest.approx(roc_auc_score(y, tied), abs=1e-12)


def test_auc_perfect_and_inverted_separators():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auc_rank_sum(y, np.array([1.0, 2, 3, 4, 5, 6])) == pytest.approx(1.0)
    assert auc_rank_sum(y, np.array([6.0, 5, 4, 3, 2, 1])) == pytest.approx(0.0)


def test_gini_matches_sklearn_auc(scores_and_labels):
    """The credit-scoring GINI, 2*AUC-1, against sklearn's AUC as the oracle."""
    y, score = scores_and_labels
    assert gini(y, score) == pytest.approx(2 * roc_auc_score(y, score) - 1, abs=1e-12)


def test_ks_matches_scipy_two_sample(scores_and_labels):
    """KS here is the two-sample statistic between the bad and good score
    distributions, so scipy's ks_2samp is the correct oracle."""
    y, score = scores_and_labels
    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert ks_statistic(y, score) == pytest.approx(expected, abs=1e-9)


def test_ks_handles_heavy_ties():
    """Regression test. Scores bucketed to 5 distinct values, which is where the
    row-by-row cumulative-sum version produced a spurious maximum."""
    rng = np.random.default_rng(SEED)
    n = 2000
    y = rng.binomial(1, 0.3, size=n)
    score = np.round(rng.normal(loc=y * 1.5, scale=1.0) * 2) / 2  # coarse buckets
    assert len(np.unique(score)) < n / 50, "test needs heavy ties to be meaningful"

    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert ks_statistic(y, score) == pytest.approx(expected, abs=1e-9)


def test_ks_bounds(scores_and_labels):
    y, score = scores_and_labels
    assert 0.0 <= ks_statistic(y, score) <= 1.0


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_shift():
    """PSI must be monotonic in the size of the shift, or it cannot be used as a
    drift threshold."""
    rng = np.random.default_rng(SEED)
    base = rng.normal(size=8000)
    small = psi(base, rng.normal(loc=0.2, size=8000))
    large = psi(base, rng.normal(loc=1.0, size=8000))
    assert 0 < small < large


def test_psi_is_symmetric_enough_to_be_reported(scores_and_labels):
    """PSI is not mathematically symmetric, but for the shifts it is used on it
    should not flip sign or change order of magnitude when reversed."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=6000)
    b = rng.normal(loc=0.4, size=6000)
    forward, backward = psi(a, b), psi(b, a)
    assert forward > 0 and backward > 0
    assert 0.5 < forward / backward < 2.0
