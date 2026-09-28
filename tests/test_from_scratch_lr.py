"""Tests for the from-scratch logistic regression in src/from_scratch_lr.py.

The README claims this implementation reproduces sklearn. These tests are what
makes that claim checkable rather than asserted.

Two things worth knowing about the comparison, both documented in the repo:

1. On a small, low-collinearity feature set the coefficients agree closely. On
   the full selected set they diverge more while predictions still correlate
   above 0.999, because several selected features are near-duplicates by
   construction. That is multicollinearity, not a solver bug, so these tests
   deliberately use clean, low-correlation synthetic features: they test the
   solver, not the feature selection.

2. class_weight="balanced" must match sklearn's convention exactly. Comparing a
   weighted fit against an unweighted one is comparing two different objectives,
   not two solvers of the same problem. That mistake was made and caught during
   the rebuild, so test_balanced_matches_sklearn_balanced exists to keep it
   caught.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from from_scratch_lr import FromScratchLogisticRegression

SEED = 42


def _clean_data(n=3000, d=5, imbalance=0.5):
    """Standardised, low-collinearity features with a known linear signal."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, d))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    true_w = np.array([1.2, -0.8, 0.5, 0.0, -0.3])[:d]
    logit = X @ true_w + np.log(imbalance / (1 - imbalance))
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return X, y


def _sklearn_fit(X, y, class_weight=None):
    """Weak L2 so both implementations optimise nearly the same objective."""
    m = LogisticRegression(
        C=1e4, solver="lbfgs", max_iter=5000, class_weight=class_weight
    )
    m.fit(X, y)
    return m


def test_coefficients_match_sklearn_on_clean_features():
    X, y = _clean_data()
    mine = FromScratchLogisticRegression(lr=0.5, n_iter=6000, l2=1e-6).fit(X, y)
    theirs = _sklearn_fit(X, y)
    max_diff = np.max(np.abs(mine.coef_ - theirs.coef_[0]))
    assert max_diff < 0.05, f"max coefficient difference {max_diff:.4f}"


def test_intercept_matches_sklearn():
    X, y = _clean_data()
    mine = FromScratchLogisticRegression(lr=0.5, n_iter=6000, l2=1e-6).fit(X, y)
    theirs = _sklearn_fit(X, y)
    assert mine.intercept_ == pytest.approx(theirs.intercept_[0], abs=0.05)


def test_predictions_correlate_with_sklearn():
    """The headline claim in the README. Predictions are what the scorecard uses,
    so this matters more than coefficient agreement."""
    X, y = _clean_data()
    mine = FromScratchLogisticRegression(lr=0.5, n_iter=6000, l2=1e-6).fit(X, y)
    theirs = _sklearn_fit(X, y)
    r = np.corrcoef(mine.predict_proba(X), theirs.predict_proba(X)[:, 1])[0, 1]
    assert r > 0.999, f"prediction correlation {r:.6f}"


def test_balanced_matches_sklearn_balanced():
    """On an 8%/92% target, the balanced fit must be compared against sklearn's
    balanced fit. This is the comparison that was wrong before."""
    X, y = _clean_data(imbalance=0.08)
    mine = FromScratchLogisticRegression(lr=0.5, n_iter=6000, l2=1e-6).fit(
        X, y, class_weight="balanced"
    )
    theirs = _sklearn_fit(X, y, class_weight="balanced")
    r = np.corrcoef(mine.predict_proba(X), theirs.predict_proba(X)[:, 1])[0, 1]
    assert r > 0.999, f"balanced prediction correlation {r:.6f}"


def test_balanced_shifts_predictions_upward():
    """Sanity check that class_weight is doing something at all: weighting the
    minority class up must raise mean predicted probability."""
    X, y = _clean_data(imbalance=0.08)
    plain = FromScratchLogisticRegression(n_iter=3000).fit(X, y)
    bal = FromScratchLogisticRegression(n_iter=3000).fit(X, y, class_weight="balanced")
    assert bal.predict_proba(X).mean() > plain.predict_proba(X).mean()


def test_cost_decreases_monotonically():
    """Gradient descent with a fixed step can diverge if the learning rate is
    wrong. A non-decreasing cost history is the cheapest way to catch that."""
    X, y = _clean_data()
    m = FromScratchLogisticRegression(lr=0.5, n_iter=500).fit(X, y)
    hist = np.asarray(m.cost_history_)
    assert len(hist) > 1
    assert np.all(np.diff(hist) <= 1e-9), "cost increased during training"


def test_predict_proba_in_unit_interval():
    X, y = _clean_data()
    p = FromScratchLogisticRegression(n_iter=1000).fit(X, y).predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_separable_data_ranks_correctly():
    """A trivially separable problem must come out ordered correctly, regardless
    of how large the coefficients grow."""
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    p = FromScratchLogisticRegression(lr=0.5, n_iter=4000, l2=1e-6).fit(X, y).predict_proba(X)
    assert p[0] < p[1] < p[2] < p[3]
