"""Feature engineering on the application table itself: anomaly flags, ratios, EXT_SOURCE aggregates."""

import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243  # Kaggle-documented placeholder for "not employed" / pensioner


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Anomaly flags ---
    df["ANOM_DAYS_EMPLOYED"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(np.int8)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    # --- Age / tenure conversions (DAYS_* are negative, days-before-application) ---
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = df["EMPLOYED_YEARS"] / df["AGE_YEARS"]

    # --- Financial ratios ---
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    # --- EXT_SOURCE aggregates (external credit bureau scores; the strongest raw signal
    #     in this dataset per every public analysis, including the notebook this project
    #     is modelled on) ---
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_NULL_COUNT"] = df[ext_cols].isnull().sum(axis=1)
    df["EXT_SOURCE_1_x_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_SOURCE_2_x_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN_x_AGE"] = df["EXT_SOURCE_MEAN"] * df["AGE_YEARS"]

    # --- Document / contact-info completeness (proxy for applicant diligence) ---
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    contact_cols = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                                 "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"] if c in df.columns]
    if contact_cols:
        df["CONTACT_FLAG_COUNT"] = df[contact_cols].sum(axis=1)

    return df
