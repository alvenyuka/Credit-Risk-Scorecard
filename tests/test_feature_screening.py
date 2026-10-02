"""The feature-screening gates in src/fit_model.py and the feature engineering they read.

* engineer_baseline sets DAYS_EMPLOYED_ANOM and leaves EMPLOYED_YEARS missing on the same rows
* the correlation gate keeps the higher-IV feature of a near-duplicate pair, deterministically
* the sign gate removes a suppressor whose coefficient reverses given a collinear partner
"""
import numpy as np
import pandas as pd

from baseline_features import DAYS_EMPLOYED_ANOMALY, engineer_baseline
from feature_lists import BASE_NUMERIC_COLS, CATEGORICAL_COLS
from fit_model import correlation_gate, eliminate_wrong_signs, rank_by_iv

SEED = 42


def _application_rows():
    rng = np.random.default_rng(SEED)
    n = 6
    df = pd.DataFrame({c: rng.uniform(1, 10, n) for c in BASE_NUMERIC_COLS})
    for c in CATEGORICAL_COLS:
        df[c] = "x"
    df["SK_ID_CURR"] = range(n)
    df["TARGET"] = [0, 1, 0, 0, 1, 0]
    df["DAYS_BIRTH"] = -365.25 * np.array([25, 40, 63, 70, 33, 51])
    df["DAYS_EMPLOYED"] = [-365.25, DAYS_EMPLOYED_ANOMALY, -3652.5, DAYS_EMPLOYED_ANOMALY, -730.5, -100.0]
    return df


def test_anomaly_flag_and_missing_employment_coincide():
    out = engineer_baseline(_application_rows())
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0, 1, 0, 0]
    assert out["EMPLOYED_YEARS"].isna().tolist() == [False, True, False, True, False, False]
    assert out.loc[0, "EMPLOYED_YEARS"] == 1.0
    assert out.loc[2, "AGE_YEARS"] == 63.0


def test_engineered_table_carries_no_prohibited_basis():
    rows = _application_rows()
    rows["CODE_GENDER"] = "F"
    rows["NAME_FAMILY_STATUS"] = "Married"
    out = engineer_baseline(rows)
    assert "CODE_GENDER" not in out.columns and "NAME_FAMILY_STATUS" not in out.columns


def test_correlation_gate_keeps_the_higher_iv_feature():
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=2000)
    W = np.column_stack([a, a + rng.normal(0, 0.05, 2000), rng.normal(size=2000)])
    cols = ["dup_low_iv", "dup_high_iv", "independent"]
    ivs = {"dup_low_iv": 0.10, "dup_high_iv": 0.30, "independent": 0.05}
    kept, dropped = correlation_gate(W, cols, ivs, threshold=0.9)
    assert kept == ["dup_high_iv", "independent"]
    assert [(d, k) for d, k, _ in dropped] == [("dup_low_iv", "dup_high_iv")]


def test_correlation_gate_breaks_iv_ties_by_name():
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=1000)
    W = np.column_stack([a, a])
    kept, _ = correlation_gate(W, ["zeta", "alpha"], {"zeta": 0.2, "alpha": 0.2})
    assert kept == ["alpha"]


def test_rank_by_iv_is_deterministic():
    fits = {"b": {"iv": 0.2}, "a": {"iv": 0.2}, "c": {"iv": 0.5}}
    assert rank_by_iv(fits) == ["c", "a", "b"]


def test_sign_gate_removes_a_reversed_suppressor():
    """x2 is mildly protective on its own but, given x1, its coefficient reverses.

    Every column is coded so that a higher value is safer (as WoE is), so each
    coefficient in a P(bad) model should be negative; the gate removes x2.
    """
    rng = np.random.default_rng(SEED)
    n = 20000
    z = rng.normal(size=n)
    x1 = z + rng.normal(0, 0.3, n)
    x2 = 0.8 * z + rng.normal(0, 0.6, n)
    x3 = rng.normal(size=n)
    logit = -2.4 - 1.2 * x1 + 0.5 * x2 - 0.6 * x3
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    X = np.column_stack([x1, x2, x3])
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    assert np.corrcoef(x2, y)[0, 1] < 0  # protective on its own

    kept, removed = eliminate_wrong_signs(Xs, y, ["x1", "x2", "x3"])
    assert kept == ["x1", "x3"]
    assert [c for c, _ in removed] == ["x2"] and removed[0][1] > 0
