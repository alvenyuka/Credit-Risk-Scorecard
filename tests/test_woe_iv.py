"""Tests for the from-scratch Weight of Evidence and Information Value in
src/woe_iv.py.

There is no library implementation to check these against, which is exactly why
they had no tests until now while the metrics and the solver did. Every point
value the scorecard publishes is `-factor * coef * woe`, so this module sits
under every published number in the README and was the one piece of the
pipeline being asserted in a `__main__` block rather than tested.

Since there is no oracle, these check properties instead:

  * a near-perfect separator scores a high IV, pure noise scores near zero
  * the WoE sign convention is the credit-scoring one (positive = safer)
  * Laplace smoothing keeps an empty bin finite instead of +/- inf
  * a category unseen at fit time maps to a neutral 0, not NaN
  * binning is fit once on train and applied unchanged to another split

The last two tests pin a known limitation rather than a desired behaviour.
Quantile binning collapses zero-inflated columns, and a 0/1 numeric column
collapses to a single bin with IV exactly 0, silently. That is documented under
Known Limitations in the README; it is pinned here so that a future change to
`fit_continuous_bins` shows up as a failing test rather than as a quiet shift in
every published point value.
"""
import numpy as np
import pandas as pd
import pytest

from woe_iv import (
    apply_continuous_bins,
    fit_continuous_bins,
    fit_woe,
    iv_strength,
    transform_woe,
)

SEED = 42


@pytest.fixture
def separable():
    """A near-perfect separator and an unrelated column, on the same labels."""
    rng = np.random.default_rng(SEED)
    n = 20000
    y = pd.Series(rng.integers(0, 2, n))
    perfect = pd.Series(y.to_numpy() + rng.normal(0, 0.01, n))
    noise = pd.Series(rng.normal(0, 1, n))
    return y, perfect, noise


def test_iv_separates_signal_from_noise(separable):
    y, perfect, noise = separable
    assert fit_woe(perfect, y, n_bins=10)["iv"] > 1.0
    assert fit_woe(noise, y, n_bins=10)["iv"] < 0.02


def test_iv_strength_labels_match_the_documented_bands():
    assert iv_strength(0.01) == "useless"
    assert iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.4) == "strong"
    assert "leakage" in iv_strength(0.6)


def test_woe_sign_is_the_credit_scoring_convention():
    """Positive WoE means a bin is safer than average, negative means riskier.

    Getting this backwards flips every point value in the scorecard while
    leaving AUC untouched, so the sign is worth a test of its own.
    """
    x = pd.Series(["safe"] * 1000 + ["risky"] * 1000)
    y = pd.Series([0] * 950 + [1] * 50 + [0] * 500 + [1] * 500)
    table = fit_woe(x, y, is_categorical=True)["table"]
    assert table.loc["safe", "woe"] > 0
    assert table.loc["risky", "woe"] < 0


def test_iv_is_never_negative(separable):
    """IV sums (dist_good - dist_bad) * woe over bins. Both factors carry the
    same sign in every bin, so a negative total means the arithmetic is wrong."""
    y, perfect, noise = separable
    for col in (perfect, noise):
        assert fit_woe(col, y, n_bins=10)["iv"] >= 0.0


def test_smoothing_keeps_an_empty_bin_finite():
    """A bin with zero bads would send ln(dist_good / dist_bad) to +inf without
    the epsilon. One infinite WoE poisons the IV sum and every downstream
    score."""
    x = pd.Series(["clean"] * 500 + ["mixed"] * 500)
    y = pd.Series([0] * 500 + [0] * 250 + [1] * 250)
    fit = fit_woe(x, y, is_categorical=True)
    assert np.isfinite(fit["table"]["woe"]).all()
    assert np.isfinite(fit["iv"])


def test_unseen_category_maps_to_neutral_zero():
    """A category that appears only in validation has no fitted WoE. It must map
    to 0, the neutral value, rather than NaN, which would propagate into the
    standardisation and silently drop the row."""
    x_train = pd.Series(["a", "b"] * 100)
    y_train = pd.Series([0, 1] * 100)
    fit = fit_woe(x_train, y_train, is_categorical=True)

    x_val = pd.Series(["a", "b", "c"])
    mapped = transform_woe(x_val, fit)
    assert mapped.notna().all()
    assert mapped.iloc[2] == 0.0


def test_missing_values_get_their_own_bin():
    """NaN is information in credit data, not an absence of it, so it is binned
    rather than dropped."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    y = pd.Series([0, 0, 1, 1, 1, 1])
    fit = fit_woe(x, y, n_bins=4)
    assert "Missing" in fit["table"].index


def test_bins_are_fit_on_train_and_applied_unchanged():
    """Refitting quantiles on a new split would let that split's own label
    distribution choose its own bin edges. The edges must travel with the fit."""
    rng = np.random.default_rng(SEED)
    x_train = pd.Series(rng.normal(0, 1, 5000))
    edges = fit_continuous_bins(x_train, 10)

    x_val = pd.Series(rng.normal(3, 1, 5000))
    labels = apply_continuous_bins(x_val, edges)
    assert set(labels.unique()).issubset(set(apply_continuous_bins(x_train, edges).unique()))
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_binary_numeric_column_collapses_to_one_bin_with_zero_iv():
    """Pins a known limitation, not a desired behaviour.

    Ten quantiles of a 0/1 column dedupe to two edges, which is fewer than the
    three `fit_continuous_bins` needs, so it falls back to a single
    (-inf, inf) bin. One bin has WoE 0 everywhere and IV exactly 0, whatever
    the column predicts, and nothing warns. See Known Limitations in the README.
    """
    rng = np.random.default_rng(SEED)
    n = 20000
    flag = pd.Series(rng.binomial(1, 0.5, n).astype(float))
    y = pd.Series(np.where(flag > 0, rng.binomial(1, 0.4, n), rng.binomial(1, 0.02, n)))

    fit = fit_woe(flag, y, is_categorical=False, n_bins=10)
    assert len(fit["table"]) == 1
    assert fit["iv"] == pytest.approx(0.0, abs=1e-12)

    # The same column read as categorical recovers the signal, which is what
    # week8_run.py does for DAYS_EMPLOYED_ANOM and why that one survives.
    assert fit_woe(flag, y, is_categorical=True)["iv"] > 0.3


def test_zero_inflated_column_loses_its_tail_to_the_quantile_edges():
    """Pins the same limitation on the shape the delinquency features have.

    Most applicants have never been overdue, so the lower deciles all sit at
    zero, dedupe to one edge, and the entire non-zero tail is squeezed into a
    single bin. Two differently defined aggregates of the same column therefore
    land on the identical split, which is why `BUREAU_OVERDUE_MAX` and
    `BUREAU_OVERDUE_MEAN` correlate at r = 1.000 in the real run.
    """
    rng = np.random.default_rng(SEED)
    n = 20000
    nonzero = rng.random(n) >= 0.85
    overdue_max = pd.Series(np.where(nonzero, rng.gamma(2, 50, n), 0.0))
    overdue_mean = pd.Series(np.where(nonzero, rng.gamma(2, 20, n), 0.0))

    assert len(fit_continuous_bins(overdue_max, 10)) - 1 == 2
    max_bins = fit_woe(overdue_max, pd.Series(rng.binomial(1, 0.08, n)))["table"]
    assert len(max_bins) == 2

    # Both columns are zero on exactly the same rows, so the two-bin split is
    # the same split, and the WoE encodings become perfectly collinear.
    assert (overdue_max > 0).equals(overdue_mean > 0)
