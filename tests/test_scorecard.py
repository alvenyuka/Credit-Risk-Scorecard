"""Tests for src/scorecard.py, the module every score, cut-off and reason code depends on.

All on synthetic data with a fixed seed:

  * de-standardised coefficients give the same log-odds as the standardised model
  * score_dataframe equals offset - factor * logit before clipping
  * the prior correction restores the training default rate and leaves ranking unchanged
  * reason codes are points below the best bin and never a characteristic scored at maximum
  * the WoE coefficient-sign check rejects a positive coefficient
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from from_scratch_lr import FromScratchLogisticRegression
from metrics_scratch import auc_rank_sum
from scorecard import (
    assert_woe_coefficient_signs,
    build_scorecard,
    destandardize_coefficients,
    odds_at_score,
    prior_correct_intercept,
    reason_codes,
    score_dataframe,
    score_to_pd,
)
from woe_iv import _bin_labels, fit_woe, transform_woe

SEED = 42


def _book(n=20000, bad_rate=0.08):
    """Two numeric drivers and one categorical driver of default, about 8% bad."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=n)
    x2 = rng.gamma(2.0, 2.0, size=n)
    x3 = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    logit = np.log(bad_rate / (1 - bad_rate)) + 0.9 * x1 + 0.25 * (x2 - 4) + np.where(x3 == "c", 0.7, 0.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, pd.Series(y)


@lru_cache(maxsize=2)
def _fitted_scorecard(prior_correct=True):
    """Fitted once per setting and shared; tests must not mutate the result."""
    X, y = _book()
    fits = {"x1": fit_woe(X["x1"], y), "x2": fit_woe(X["x2"], y),
            "x3": fit_woe(X["x3"], y, is_categorical=True)}
    cols = ["x1", "x2", "x3"]
    W = np.column_stack([transform_woe(X[c], fits[c]).to_numpy(dtype=float) for c in cols])
    mean, std = W.mean(axis=0), W.std(axis=0)
    model = FromScratchLogisticRegression(lr=0.5, n_iter=3000, l2=1e-3)
    model.fit((W - mean) / std, y.to_numpy(), class_weight="balanced")
    coef, intercept = destandardize_coefficients(model.coef_, model.intercept_, mean, std)
    if prior_correct:
        intercept = prior_correct_intercept(intercept, float(y.mean()))
    sc = build_scorecard(coef, intercept, cols, fits)
    return X, y, W, coef, intercept, model, mean, std, sc


def test_destandardized_coefficients_reproduce_the_log_odds():
    rng = np.random.default_rng(SEED)
    X = rng.normal(3.0, 2.0, size=(500, 4))
    mean, std = X.mean(axis=0), X.std(axis=0)
    coef_std, b_std = rng.normal(size=4), 0.3
    coef_raw, b_raw = destandardize_coefficients(coef_std, b_std, mean, std)
    np.testing.assert_allclose((X - mean) / std @ coef_std + b_std, X @ coef_raw + b_raw, atol=1e-10)


def test_score_is_offset_minus_factor_times_logit_before_clipping():
    X, y, W, coef, intercept, *_, sc = _fitted_scorecard()
    raw = score_dataframe(X, sc, clip=False)
    expected = sc["offset"] - sc["factor"] * (intercept + W @ coef)
    np.testing.assert_allclose(raw.to_numpy(), expected, atol=1e-8)
    clipped = score_dataframe(X, sc)
    assert clipped.min() >= 300 and clipped.max() <= 850


def test_prior_correction_restores_the_training_default_rate():
    """The balanced fit predicts a mean PD near 50%; the corrected one near the true rate."""
    X, y, W, coef, intercept, model, mean, std, sc = _fitted_scorecard()
    pd_corrected = score_to_pd(score_dataframe(X, sc, clip=False), sc)
    assert abs(pd_corrected.mean() - y.mean()) < 0.005

    balanced_pd = model.predict_proba((W - mean) / std)
    assert balanced_pd.mean() > 0.3


def test_prior_correction_leaves_ranking_unchanged():
    X, y, *_, sc_corrected = _fitted_scorecard(prior_correct=True)
    sc_balanced = _fitted_scorecard(prior_correct=False)[-1]
    auc_c = auc_rank_sum(y, -score_dataframe(X, sc_corrected, clip=False))
    auc_b = auc_rank_sum(y, -score_dataframe(X, sc_balanced, clip=False))
    assert auc_c == pytest.approx(auc_b, abs=1e-12)


def test_prior_correction_arithmetic():
    p = 0.08
    assert prior_correct_intercept(0.0, p) == pytest.approx(np.log(p / (1 - p)))
    assert prior_correct_intercept(1.0, 0.5) == pytest.approx(1.0)


def test_base_score_means_base_odds():
    sc = _fitted_scorecard()[-1]
    assert odds_at_score(600, sc) == pytest.approx(20.0)
    assert odds_at_score(640, sc) == pytest.approx(40.0)
    assert score_to_pd(600, sc) == pytest.approx(1 / 21)


def test_reason_codes_are_points_below_the_best_bin_and_never_positive():
    X, y, *_, sc = _fitted_scorecard()
    for i in range(50):
        row = X.iloc[i]
        reasons = reason_codes(row, sc, top_n=3)
        assert all(v > 0 for _, v in reasons)
        assert [v for _, v in reasons] == sorted((v for _, v in reasons), reverse=True)
        for col, lost in reasons:
            pts = sc["points_tables"][col]
            fit = sc["woe_fits"][col]
            label = _bin_labels(pd.Series([row[col]]), fit["is_categorical"], fit["edges"]).iloc[0]
            assert lost == pytest.approx(pts.max() - pts[label])


def test_applicant_in_every_best_bin_gets_no_reasons():
    X, y, *_, sc = _fitted_scorecard()
    best = {}
    for col in sc["kept_cols"]:
        fit = sc["woe_fits"][col]
        labels = _bin_labels(X[col], fit["is_categorical"], fit["edges"])
        best[col] = X[col][labels == sc["points_tables"][col].idxmax()].iloc[0]
    assert reason_codes(pd.Series(best), sc) == []


def test_sign_check_rejects_a_positive_coefficient():
    assert_woe_coefficient_signs(np.array([-0.5, -0.1]), ["a", "b"])
    with pytest.raises(ValueError, match="b"):
        assert_woe_coefficient_signs(np.array([-0.5, 0.2]), ["a", "b"])


def test_fitted_coefficients_on_woe_inputs_are_negative():
    """A correctly specified P(bad) model on WoE = ln(good/bad) gives negative coefficients."""
    _, _, _, coef, *_ = _fitted_scorecard()
    assert (coef < 0).all()
