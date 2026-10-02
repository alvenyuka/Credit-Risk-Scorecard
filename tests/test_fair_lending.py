"""Fair-lending conditions on the feature lists and on any fitted scorecard.

* Sex and marital status (prohibited bases under ECOA and Regulation B) appear in
  no feature list the scorecard or the baselines can read.
* Regulation B allows age in an empirically derived system only if applicants
  aged 62 or over get no negative factor: the check passes on a scorecard that
  rewards age and fails on one that penalises it, including a bin that straddles 62.
"""
import numpy as np
import pandas as pd
import pytest

import feature_lists as fl
from scorecard import age_62_plus_points, assert_no_negative_points_for_age_62_plus


def test_prohibited_bases_are_not_candidates():
    lists = [fl.BASE_NUMERIC_COLS, fl.CATEGORICAL_COLS, fl.ENGINEERED_NUMERIC, fl.RELATIONAL_NUMERIC,
             fl.EDA_ONLY_COLS, fl.APPLICATION_NUMERIC, fl.BASELINE_NUMERIC, fl.MODEL_NUMERIC,
             fl.MODEL_CATEGORICAL]
    for cols in lists:
        assert not set(cols) & set(fl.PROHIBITED_BASES)


def test_sex_and_marital_status_are_named_as_prohibited():
    assert {"CODE_GENDER", "NAME_FAMILY_STATUS"} <= set(fl.PROHIBITED_BASES)


def test_anomaly_flag_is_not_offered_to_the_scorecard():
    assert "DAYS_EMPLOYED_ANOM" not in fl.MODEL_NUMERIC + fl.MODEL_CATEGORICAL


def _age_scorecard(points: list, labels=None) -> dict:
    labels = labels or ["(-inf, 30.0]", "(30.0, 45.0]", "(45.0, 60.709]", "(60.709, inf]"]
    return {"kept_cols": ["AGE_YEARS"],
            "points_tables": {"AGE_YEARS": pd.Series(points, index=labels, dtype=float)}}


def test_age_check_passes_when_older_bins_gain_points():
    summary = assert_no_negative_points_for_age_62_plus(_age_scorecard([-12.0, -3.0, 4.0, 9.5]))
    assert summary["passed"] and summary["min_points_62_plus"] == 9.5
    assert list(summary["bins_covering_62_plus"]) == ["(60.709, inf]"]


def test_age_check_fails_when_the_top_bin_loses_points():
    with pytest.raises(ValueError, match="Regulation B"):
        assert_no_negative_points_for_age_62_plus(_age_scorecard([5.0, 2.0, 1.0, -0.5]))


def test_a_bin_straddling_62_counts_as_covering_62_plus():
    labels = ["(-inf, 40.0]", "(40.0, 63.5]", "(63.5, inf]"]
    pts = age_62_plus_points(_age_scorecard([1.0, -2.0, 3.0], labels))
    assert list(pts.index) == ["(40.0, 63.5]", "(63.5, inf]"]
    with pytest.raises(ValueError):
        assert_no_negative_points_for_age_62_plus(_age_scorecard([1.0, -2.0, 3.0], labels))


def test_age_check_records_when_age_is_not_in_the_scorecard():
    sc = {"kept_cols": ["EXT_SOURCE_MEAN"], "points_tables": {"EXT_SOURCE_MEAN": pd.Series([1.0])}}
    summary = assert_no_negative_points_for_age_62_plus(sc)
    assert summary["passed"] and not summary["age_feature_in_scorecard"]
    assert summary["bins_covering_62_plus"] == {}


def test_missing_age_bin_is_not_treated_as_62_plus():
    labels = ["(-inf, 50.0]", "(50.0, inf]", "Missing"]
    pts = age_62_plus_points(_age_scorecard([0.0, 2.0, -1.0], labels))
    assert list(pts.index) == ["(50.0, inf]"]
    assert np.isclose(pts.iloc[0], 2.0)
