"""
Feature engineering on the application table itself: anomaly flags, ratios, EXT_SOURCE aggregates.

Everything here operates on the applicant's own submitted data, before any
join to bureau/prior-application/POS/credit-card/installment history. The
guiding idea is the same one every credit officer applies by hand: raw
numbers (income, credit amount, age) are far less informative than their
ratios to each other. A KES 500,000 loan means something different to an
applicant earning KES 50,000/month than to one earning KES 500,000/month --
CREDIT_INCOME_RATIO captures that, AMT_CREDIT alone doesn't.
"""

import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243  # Kaggle-documented placeholder for "not employed" / pensioner


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Anomaly flags ---
    # DAYS_EMPLOYED should always be negative (days before the application date,
    # same convention as DAYS_BIRTH). Instead, ~18% of rows carry exactly
    # 365243 -- almost certainly a sentinel value Home Credit's own systems use
    # for "not currently employed" (pensioners, unemployed applicants), not a
    # real day-count. Left as-is, it would read as "employed 1,000 years ago,"
    # which would poison EMPLOYED_YEARS and every ratio derived from it. The
    # flag preserves the signal (these applicants really are a distinct group)
    # while the NaN keeps the anomaly out of downstream arithmetic.
    df["ANOM_DAYS_EMPLOYED"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(np.int8)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    # --- Age / tenure conversions (DAYS_* are negative, days-before-application) ---
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = df["EMPLOYED_YEARS"] / df["AGE_YEARS"]

    # --- Financial ratios ---
    # Each of these answers a question a human underwriter would actually ask:
    #   CREDIT_INCOME_RATIO   -- how many years of income does this loan represent?
    #   ANNUITY_INCOME_RATIO  -- what fraction of income does the monthly payment eat?
    #   CREDIT_ANNUITY_RATIO  -- implied loan term in "annuity units"
    #   CREDIT_GOODS_RATIO    -- is the loan larger than the goods it's financing? (overfinancing signal)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    # --- EXT_SOURCE aggregates (external credit bureau scores; the strongest raw signal
    #     in this dataset per every public analysis, including the notebook this project
    #     is modelled on) ---
    # EXT_SOURCE_1/2/3 are three normalized external scores Home Credit buys in from
    # other credit bureaus/scoring services. Kaggle never discloses what they measure,
    # which is realistic: in production, a bureau score is usually a licensed black box
    # too. All three are frequently missing (different bureaus cover different
    # applicants), so MEAN/MIN/MAX/STD and the null count all carry information --
    # an applicant with all three populated is a different population than one with
    # zero, independent of what the scores themselves say.
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_NULL_COUNT"] = df[ext_cols].isnull().sum(axis=1)
    df["EXT_SOURCE_1_x_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_SOURCE_2_x_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN_x_AGE"] = df["EXT_SOURCE_MEAN"] * df["AGE_YEARS"]

    # --- Document / contact-info completeness ---
    # FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21 are binary "did the applicant submit
    # this supporting document" flags. No single one carries much signal, but the
    # count of documents submitted (DOCUMENT_COUNT) has IV 0.0214 in the real ranking
    # (output/feature_iv_ranking.csv, rank 172 of 400) -- real but weak by the
    # Siddiqi bands in woe_iv.py, not a strong predictor on its own. Plausible reading
    # is that it's a diligence proxy (an applicant who submits more paperwork is
    # scrutinized less), but the IV number alone doesn't confirm that mechanism, it
    # only confirms the count isn't nothing.
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    contact_cols = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                                 "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"] if c in df.columns]
    if contact_cols:
        df["CONTACT_FLAG_COUNT"] = df[contact_cols].sum(axis=1)

    return df
