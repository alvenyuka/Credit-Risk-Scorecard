"""
Week 8 — relational aggregation, own design, from scratch.

Everything below rolls a table keyed at a finer grain (per bureau record,
per previous application, per monthly balance snapshot) up to one row per
SK_ID_CURR, so it can be joined onto the application table. Only usecols
that actually get aggregated are read, and dtypes are narrowed on read --
these files are large (installments_payments.csv alone is 690MB / 13.6M
rows) and there's no reason to pay for a float64 column of a 0/1 flag.

Feature choices and why (decision log for this step):
- bureau.csv (credit history reported to Home Credit by *other* lenders):
  count of records, share still "Active", overdue-day stats, and credit-sum
  stats. Bureau history is the single biggest information source Week 6/7
  didn't have access to -- this is where the AUC lift, if any, should come
  from.
- bureau_balance.csv (monthly delinquency status per bureau record): STATUS
  in {'1'..'5'} means some degree of days-past-due; {'C','X','0'} mean
  closed/unknown/no-DPD-that-month. Rolled up to "was this bureau record
  ever delinquent" and "how many months of history exist", then joined
  through bureau.csv up to SK_ID_CURR (bureau_balance itself doesn't carry
  SK_ID_CURR).
- previous_application.csv (the applicant's own history with Home Credit):
  count, approval/refusal share, and recency of the last decision.
- POS_CASH_balance.csv / credit_card_balance.csv: days-past-due (SK_DPD)
  stats -- direct behavioral signal of whether this applicant has actually
  missed payments before, which is a stronger signal than anything on the
  application form.
- installments_payments.csv: payment timeliness (days late) and payment
  shortfall (paid less than owed) per installment, aggregated per applicant.
  This is the most direct "did they actually pay on time" signal in the
  whole dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _bureau_balance_agg() -> pd.DataFrame:
    cols = ["SK_ID_BUREAU", "STATUS"]
    df = pd.read_csv(DATA_DIR / "bureau_balance.csv", usecols=cols, dtype={"STATUS": "category"})
    delinquent = df["STATUS"].isin(["1", "2", "3", "4", "5"])
    agg = df.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("STATUS", "count"),
        BB_EVER_DELINQUENT=("STATUS", lambda s: int(delinquent.loc[s.index].any())),
    )
    return agg.reset_index()


def bureau_features() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
             "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
    df = pd.read_csv(DATA_DIR / "bureau.csv", usecols=cols)

    bb = _bureau_balance_agg()
    df = df.merge(bb, on="SK_ID_BUREAU", how="left")

    df["IS_ACTIVE"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_EVER_DELINQUENT_SHARE=("BB_EVER_DELINQUENT", "mean"),
    ).reset_index()
    return agg


def previous_application_features() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
             "AMT_CREDIT", "DAYS_DECISION"]
    df = pd.read_csv(DATA_DIR / "previous_application.csv", usecols=cols)

    df["IS_APPROVED"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["IS_REFUSED"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_SHARE=("IS_APPROVED", "mean"),
        PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),  # least-negative = most recent
    ).reset_index()
    return agg


def pos_cash_features() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT"]
    df = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv", usecols=cols)

    agg = df.groupby("SK_ID_CURR").agg(
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
    ).reset_index()
    return agg


def credit_card_features() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"]
    df = pd.read_csv(DATA_DIR / "credit_card_balance.csv", usecols=cols)

    limit = df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    df["UTILIZATION"] = df["AMT_BALANCE"] / limit

    agg = df.groupby("SK_ID_CURR").agg(
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
    ).reset_index()
    return agg


def installments_features() -> pd.DataFrame:
    cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    dtypes = {c: "float32" for c in cols if c != "SK_ID_CURR"}
    df = pd.read_csv(DATA_DIR / "installments_payments.csv", usecols=cols, dtype=dtypes)

    # positive = paid late (entry payment day comes after the due day, both
    # stored as negative day-offsets from application date)
    df["DAYS_LATE"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["SHORTFALL"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    agg = df.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT=("AMT_INSTALMENT", "count"),
        INSTAL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INSTAL_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INSTAL_SHORTFALL_MEAN=("SHORTFALL", "mean"),
        INSTAL_SHORTFALL_SUM=("SHORTFALL", "sum"),
    ).reset_index()
    return agg


def build_all_relational_features() -> pd.DataFrame:
    parts = [
        bureau_features(),
        previous_application_features(),
        pos_cash_features(),
        credit_card_features(),
        installments_features(),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="SK_ID_CURR", how="outer")
    return out


if __name__ == "__main__":
    feats = build_all_relational_features()
    print("relational feature table:", feats.shape)
    print("columns:", list(feats.columns))
    print("\nmissing-value share (applicants with no history in that table are legitimately NaN):")
    print((feats.isna().mean() * 100).round(1))

    cache_path = Path(__file__).resolve().parent / "_cache_relational_features.parquet"
    feats.to_parquet(cache_path, index=False)
    print(f"\ncached to {cache_path.name} (gitignored, regenerate anytime by re-running this script)")
