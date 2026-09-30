"""Prohibited bases must never reach the scorecard's candidate features."""
from baseline_features import BASE_NUMERIC_COLS, CATEGORICAL_COLS, PROHIBITED_BASES


def test_prohibited_bases_are_not_candidates():
    candidates = set(BASE_NUMERIC_COLS) | set(CATEGORICAL_COLS)
    assert not candidates & set(PROHIBITED_BASES)


def test_sex_and_marital_status_are_named_as_prohibited():
    assert {"CODE_GENDER", "NAME_FAMILY_STATUS"} <= set(PROHIBITED_BASES)
