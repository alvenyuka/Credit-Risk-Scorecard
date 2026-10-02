"""
Relational aggregation: rolls each linked table up to one row per SK_ID_CURR so
it can be joined onto the application table. Only the aggregated columns are
read, and dtypes are narrowed where it matters (installments_payments.csv alone
is about 690 MB and 13.6 million rows).

Feature choices:
- bureau.csv (credit reported by other lenders): count of records, share still
  active, overdue-day statistics and credit-sum totals. This is the largest
  information source the application form does not have.
- bureau_balance.csv (monthly status per bureau record): STATUS '1' to '5' means
  some days past due; 'C', 'X' and '0' mean closed, unknown or current. Rolled up
  to "was this record ever delinquent" and "months of history", then joined
  through bureau.csv (bureau_balance carries no SK_ID_CURR).
- previous_application.csv (the applicant's history with Home Credit): count,
  approval and refusal shares, amounts, and recency of the last decision.
- POS_CASH_balance.csv and credit_card_balance.csv: days-past-due statistics and
  card utilisation.
- installments_payments.csv: days late and payment shortfall per instalment,
  the most direct record of whether the applicant paid on time.

The full build reads about 2.5 GB of CSV, so load_relational_features caches the
result as a parquet file next to this module, together with a fingerprint of the
aggregation code and the input file sizes. A cache whose fingerprint does not
match is rebuilt, never used silently.
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_raw import DATA_DIR

CACHE_PATH = Path(__file__).resolve().parent / "_cache_relational_features.parquet"
CACHE_META_PATH = CACHE_PATH.with_suffix(".json")
SOURCE_TABLES = [
    "bureau.csv", "bureau_balance.csv", "previous_application.csv",
    "POS_CASH_balance.csv", "credit_card_balance.csv", "installments_payments.csv",
]


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


def cache_fingerprint(data_dir: Path = None) -> str:
    """SHA-256 of the aggregation code plus the name and size of each input table.

    Editing any aggregation function or replacing a CSV changes the fingerprint,
    which forces a rebuild instead of a silent read of stale features.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    funcs = [_bureau_balance_agg, bureau_features, previous_application_features,
             pos_cash_features, credit_card_features, installments_features,
             build_all_relational_features]
    h = hashlib.sha256()
    for f in funcs:
        h.update(inspect.getsource(f).encode("utf-8"))
    for name in SOURCE_TABLES:
        path = data_dir / name
        size = path.stat().st_size if path.exists() else -1
        h.update(f"{name}:{size}".encode("utf-8"))
    return h.hexdigest()


def load_relational_features(verbose: bool = True) -> pd.DataFrame:
    """Return the relational feature table, from a validated cache or rebuilt.

    Prints which path was taken. When a cache exists but cannot be validated,
    the rebuilt table is compared with it and the comparison is printed.
    """
    expected = cache_fingerprint()
    if CACHE_PATH.exists() and CACHE_META_PATH.exists():
        meta = json.loads(CACHE_META_PATH.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == expected:
            if verbose:
                print(f"relational features: cache (fingerprint {expected[:12]} matches)")
            return pd.read_parquet(CACHE_PATH)

    stale = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else None
    feats = build_all_relational_features()
    if verbose:
        print(f"relational features: rebuilt from the CSVs (fingerprint {expected[:12]})")
        if stale is not None:
            same = (stale.shape == feats.shape and list(stale.columns) == list(feats.columns)
                    and stale.sort_values("SK_ID_CURR").reset_index(drop=True)
                    .equals(feats.sort_values("SK_ID_CURR").reset_index(drop=True)))
            print(f"  previous unvalidated cache {'matches' if same else 'DIFFERS FROM'} the rebuilt table")
    feats.to_parquet(CACHE_PATH, index=False)
    CACHE_META_PATH.write_text(json.dumps({"fingerprint": expected}, indent=2), encoding="utf-8")
    return feats


if __name__ == "__main__":
    feats = load_relational_features()
    print("relational feature table:", feats.shape)
    print("columns:", list(feats.columns))
    print("\nmissing-value share (applicants with no history in that table are legitimately NaN):")
    print((feats.isna().mean() * 100).round(1))
