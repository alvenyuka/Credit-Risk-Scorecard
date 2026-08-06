"""
Week 6 — blind rebuild, step 2: baseline feature engineering.

Own design, application-table only (no bureau/previous_application/etc. yet —
that's relational aggregation, out of scope for a first naive baseline).

Feature choices and why (this is the decision log for this step):
- DAYS_BIRTH / DAYS_EMPLOYED are negative day-counts in this dataset (days
  before the application). Converting to positive years reads more naturally
  for a first pass.
- DAYS_EMPLOYED has a well-known placeholder value (365243) used for
  pensioners/unemployed applicants who have no employment history — that's
  ~365,243 days ~= 1000 years, clearly not real. Flagging it as an anomaly
  and setting it to NaN rather than silently averaging it into "years employed"
  matters: leaving it in would make retirees look like they'd been employed
  for a millennium, which would badly distort any ratio built on top of it.
- Ratio features (credit/income, annuity/income, credit/goods price) are a
  standard credit-risk move: raw AMT_* columns are highly skewed and scale
  with income, so ratios are more comparable across applicants than raw
  amounts.
- EXT_SOURCE_1/2/3 are external credit-bureau-style scores already present in
  the data. Missingness on these looks meaningful (not every bureau has
  scored every applicant), so a per-row mean/count of how many EXT_SOURCE
  values are present is included as its own signal, not just imputed away.
"""
import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243

BASE_NUMERIC_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "CNT_CHILDREN",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]


def engineer_baseline(df: pd.DataFrame) -> pd.DataFrame:
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

    engineered = [
        "AGE_YEARS", "DAYS_EMPLOYED_ANOM", "EMPLOYED_YEARS",
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
        "CREDIT_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
        "EXT_SOURCE_MEAN", "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
    ]

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
