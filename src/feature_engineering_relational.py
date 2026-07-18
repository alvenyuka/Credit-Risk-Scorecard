"""
Aggregate the 6 supplementary Home Credit tables to one row per SK_ID_CURR.

The application table alone (feature_engineering_app.py) only sees what the
applicant declared on the form. These functions add what the applicant
*did*: prior credit-bureau behaviour, prior loans with Home Credit itself,
and month-by-month repayment/utilization history. Every function here
follows the same shape -- engineer a few derived flags/ratios on the raw
table, then collapse many rows per applicant down to one row per
SK_ID_CURR via _agg_numeric plus a handful of named counts.
"""

import numpy as np
import pandas as pd


def _agg_numeric(df: pd.DataFrame, group_col: str, prefix: str, stats=("mean", "max", "min", "sum")) -> pd.DataFrame:
    """
    Aggregate every numeric column in df (except the group key) with the given stats.

    This is deliberately generic and a little wasteful -- it aggregates every
    numeric column it finds rather than a hand-picked subset, which is why
    the joined table balloons to 418 columns (see README "Feature
    engineering"). The WoE/IV ranking step is what actually decides which of
    these are worth keeping; this function's job is just to make sure
    nothing potentially useful gets dropped before that ranking happens.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != group_col and not c.endswith("_SK_ID_CURR")]
    agg = df.groupby(group_col)[num_cols].agg(stats)
    agg.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in agg.columns]
    return agg.reset_index()


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Credit bureau records: what other lenders report about this applicant.

    bureau_balance is a separate monthly-status table keyed to bureau.csv's
    SK_ID_BUREAU (not the applicant's SK_ID_CURR directly), so it has to be
    aggregated up to the bureau-record level first, then merged into bureau,
    before either can be aggregated up again to the applicant level.
    """
    bb = bureau_balance.copy()
    # STATUS: 0-5 = days-past-due bucket, C = closed, X = unknown
    bb["BB_IS_DPD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int8)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_DPD_RATE=("BB_IS_DPD", "mean"),
    ).reset_index()

    b = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    b["BUREAU_IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    b["BUREAU_IS_OVERDUE"] = (b["AMT_CREDIT_SUM_OVERDUE"] > 0).astype(np.int8)
    b["BUREAU_CREDIT_DEBT_RATIO"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"].replace(0, np.nan)

    agg = _agg_numeric(b, "SK_ID_CURR", "BUREAU")

    # Named counts, separate from _agg_numeric: these ask "how many" and "what
    # kind" rather than "how much", which mean/max/min/sum don't capture --
    # an applicant with 1 bureau loan and one with 15 look identical to
    # _agg_numeric's MEAN/MAX columns if both have similar balances.
    counts = b.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("BUREAU_IS_ACTIVE", "sum"),
        BUREAU_OVERDUE_COUNT=("BUREAU_IS_OVERDUE", "sum"),
        BUREAU_DISTINCT_CREDIT_TYPES=("CREDIT_TYPE", "nunique"),
    ).reset_index()

    return agg.merge(counts, on="SK_ID_CURR", how="outer")


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Applicant's prior loan applications to Home Credit itself (not other lenders)."""
    p = prev.copy()
    p["PREV_IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    p["PREV_IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    p["PREV_CREDIT_APP_RATIO"] = p["AMT_CREDIT"] / p["AMT_APPLICATION"].replace(0, np.nan)
    p["PREV_ANNUITY_CREDIT_RATIO"] = p["AMT_ANNUITY"] / p["AMT_CREDIT"].replace(0, np.nan)

    agg = _agg_numeric(p, "SK_ID_CURR", "PREV")

    # PREV_REFUSED_RATE (refusals / total applications) is the interesting one
    # here: a raw refusal count conflates "applied once, got refused" with
    # "applied 20 times, got refused twice." The rate normalizes for how much
    # history an applicant even has with Home Credit.
    counts = p.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_COUNT=("PREV_IS_REFUSED", "sum"),
        PREV_APPROVED_COUNT=("PREV_IS_APPROVED", "sum"),
    ).reset_index()
    counts["PREV_REFUSED_RATE"] = counts["PREV_REFUSED_COUNT"] / counts["PREV_APP_COUNT"].replace(0, np.nan)

    return agg.merge(counts, on="SK_ID_CURR", how="outer")


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Point-of-sale and cash loan monthly balance snapshots."""
    p = pos.copy()
    p["POS_IS_DPD"] = (p["SK_DPD"] > 0).astype(np.int8)

    # mean/max only here (not min/sum like the other tables) -- a running
    # balance's minimum is usually just its final near-zero value as the loan
    # winds down, and summing a balance across months double-counts the same
    # money; neither adds information mean/max don't already carry.
    agg = _agg_numeric(p, "SK_ID_CURR", "POS", stats=("mean", "max"))

    counts = p.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("SK_ID_PREV", "count"),
        POS_DPD_RATE=("POS_IS_DPD", "mean"),
    ).reset_index()

    return agg.merge(counts, on="SK_ID_CURR", how="outer")


def aggregate_credit_card_balance(cc: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly revolving credit-card balance snapshots.

    CC_UTILIZATION (balance / credit limit) turned out to matter more than
    the raw balance -- see figs/iv_top20.png, several CC_CC_UTILIZATION_*
    and CC_CNT_DRAWINGS_* aggregates land in the top 20 features by
    Information Value, right behind the EXT_SOURCE cluster. That tracks with
    how revolving credit is actually underwritten elsewhere: utilization
    ratio, not balance size alone, is the standard risk signal because it's
    comparable across applicants with very different credit limits.
    """
    c = cc.copy()
    c["CC_UTILIZATION"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    c["CC_IS_DPD"] = (c["SK_DPD"] > 0).astype(np.int8)

    agg = _agg_numeric(c, "SK_ID_CURR", "CC", stats=("mean", "max"))

    counts = c.groupby("SK_ID_CURR").agg(
        CC_RECORD_COUNT=("SK_ID_PREV", "count"),
        CC_DPD_RATE=("CC_IS_DPD", "mean"),
    ).reset_index()

    return agg.merge(counts, on="SK_ID_CURR", how="outer")


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Actual repayment history against scheduled installments.

    This is the largest table by row count (13.6M rows) and, going in, the
    intuitive guess was that it would carry the strongest signal in the
    dataset -- "did they actually pay on time" sounds more informative than
    an external bureau score. The real IV ranking (output/feature_iv_ranking.csv)
    doesn't support that guess: the best INST_* feature (INST_LATE_FLAG_MEAN)
    lands at rank 57 of 400 with IV ~0.060, "Weak" by the Siddiqi bands in
    woe_iv.py, well behind the EXT_SOURCE cluster that fills ranks 1-9 with
    IV 0.3-0.6. Recorded here rather than quietly dropped, because an
    intuitive-but-wrong assumption about which feature would matter most is
    exactly the kind of claim this project's README insists on checking
    against real output instead of asserting from first principles.
    """
    i = inst.copy()
    i["INST_DAYS_LATE"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["INST_LATE_FLAG"] = (i["INST_DAYS_LATE"] > 0).astype(np.int8)
    i["INST_SHORTFALL"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]
    i["INST_SHORTFALL_FLAG"] = (i["INST_SHORTFALL"] > 0).astype(np.int8)

    agg = _agg_numeric(i, "SK_ID_CURR", "INST", stats=("mean", "max", "sum"))

    counts = i.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "count"),
        INST_LATE_FLAG_MEAN=("INST_LATE_FLAG", "mean"),
        INST_SHORTFALL_FLAG_MEAN=("INST_SHORTFALL_FLAG", "mean"),
    ).reset_index()

    return agg.merge(counts, on="SK_ID_CURR", how="outer")
