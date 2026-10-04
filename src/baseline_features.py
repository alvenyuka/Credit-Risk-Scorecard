"""
Application-level feature engineering.

Feature choices:
- DAYS_BIRTH and DAYS_EMPLOYED are negative day counts before the application;
  they are converted to positive years.
- DAYS_EMPLOYED holds the placeholder 365243 (about 1,000 years) for pensioners
  and unemployed applicants. The value is set to NaN, so EMPLOYED_YEARS is
  missing for exactly those applicants, and DAYS_EMPLOYED_ANOM records the flag
  for exploration. The scorecard reads the group through the Missing bin of
  EMPLOYED_YEARS only (see EDA_ONLY_COLS in feature_lists.py).
- Ratio features (credit to income, annuity to income, credit to goods price)
  are more comparable across applicants than raw amounts, which are skewed and
  scale with income.
- EXT_SOURCE_1/2/3 are external bureau-style scores supplied with the data.
  Their missingness is informative, so the per-row mean, spread and count of
  present values are kept as features rather than imputed away.
"""
import numpy as np
import pandas as pd

from feature_lists import (  # noqa: F401  (re-exported for existing imports)
    BASE_NUMERIC_COLS,
    CATEGORICAL_COLS,
    EDA_ONLY_COLS,
    ENGINEERED_NUMERIC,
    PROHIBITED_BASES,
)

DAYS_EMPLOYED_ANOMALY = 365243


def engineer_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Application features for every applicant, one row each.

    Converts day counts to years, replaces the DAYS_EMPLOYED placeholder 365243 with
    NaN (flagged in DAYS_EMPLOYED_ANOM), adds income and credit ratios and summarises
    the three external scores. Keeps SK_ID_CURR, TARGET when present, and the columns
    listed in feature_lists.py; sex and marital status are never kept.
    """
    out = df.copy()

    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25

    out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
    days_employed_clean = out["DAYS_EMPLOYED"].where(out["DAYS_EMPLOYED"] != DAYS_EMPLOYED_ANOMALY, np.nan)
    out["EMPLOYED_YEARS"] = -days_employed_clean / 365.25

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"]
    out["INCOME_PER_FAM_MEMBER"] = out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"].replace(0, np.nan)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
    out["EXT_SOURCE_STD"] = out[ext_cols].std(axis=1)
    out["EXT_SOURCE_COUNT"] = out[ext_cols].notna().sum(axis=1)

    engineered = ENGINEERED_NUMERIC + EDA_ONLY_COLS

    keep_cols = ["SK_ID_CURR"] + BASE_NUMERIC_COLS + CATEGORICAL_COLS + engineered
    if "TARGET" in out.columns:
        keep_cols = ["TARGET"] + keep_cols
    return out[keep_cols]


if __name__ == "__main__":
    from io_raw import load_application

    train = load_application("train")
    feats = engineer_baseline(train)
    print(feats.shape)
    print(feats[["AGE_YEARS", "EMPLOYED_YEARS", "DAYS_EMPLOYED_ANOM",
                  "CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN", "EXT_SOURCE_COUNT"]].describe())
