"""
Data loaders for the Home Credit Default Risk tables, with dtype downcasting.

Nothing clever happens here on purpose. This module's only job is getting
8 CSVs off disk and into memory without falling over -- the largest table,
installments_payments.csv, is 723 MB on disk and comfortably several GB
once pandas inflates it to int64/float64 by default. Downcasting to the
smallest dtype that still holds the column's actual range (see _downcast)
is what keeps Phase B of run_pipeline.py from needing a machine with more
RAM than a laptop has.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest dtype that holds their range.

    pandas' default read_csv dtypes (int64, float64) are correct but wasteful:
    a column of 0/1 flags doesn't need 8 bytes per value. pd.to_numeric's
    downcast argument inspects the actual min/max in the column and picks the
    narrowest safe type (int8/int16/int32, float32), so this is lossless for
    anything that started as int64/float64 -- it only ever narrows, never
    changes a value.
    """
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_application(split: str = "train") -> pd.DataFrame:
    """Load application_train.csv or application_test.csv (1 row per applicant)."""
    path = DATA_DIR / f"application_{split}.csv"
    df = pd.read_csv(path)
    return _downcast(df)


def load_bureau() -> pd.DataFrame:
    """Credit history reported by other lenders (1 row per prior credit-bureau record)."""
    return _downcast(pd.read_csv(DATA_DIR / "bureau.csv"))


def load_bureau_balance() -> pd.DataFrame:
    """Monthly delinquency status for each bureau.csv record (27.3M rows -- the second-largest table)."""
    return _downcast(pd.read_csv(DATA_DIR / "bureau_balance.csv"))


def load_previous_application() -> pd.DataFrame:
    """The applicant's own prior loan applications to Home Credit (not other lenders)."""
    return _downcast(pd.read_csv(DATA_DIR / "previous_application.csv"))


def load_pos_cash() -> pd.DataFrame:
    """Monthly point-of-sale / cash loan balance snapshots."""
    return _downcast(pd.read_csv(DATA_DIR / "POS_CASH_balance.csv"))


def load_credit_card_balance() -> pd.DataFrame:
    """Monthly revolving credit-card balance snapshots."""
    return _downcast(pd.read_csv(DATA_DIR / "credit_card_balance.csv"))


def load_installments() -> pd.DataFrame:
    """
    Actual vs. scheduled repayment per installment (13.6M rows, 723 MB on disk).
    The single slowest load in the pipeline -- see SETUP.md's phase timings.
    """
    return _downcast(pd.read_csv(DATA_DIR / "installments_payments.csv"))


def load_columns_description() -> pd.DataFrame:
    # This file is Latin-1 encoded, not UTF-8 -- a documented quirk of Kaggle's
    # export tooling that only affects this one file. Reading it with the
    # default UTF-8 assumption raises UnicodeDecodeError partway through
    # (the failure point depends on where the first non-ASCII byte lands),
    # which is confusing the first time you hit it because every other CSV
    # in this dataset is plain UTF-8.
    return pd.read_csv(DATA_DIR / "HomeCredit_columns_description.csv", encoding="latin-1")
