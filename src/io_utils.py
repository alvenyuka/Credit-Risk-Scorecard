"""Data loaders for the Home Credit Default Risk tables, with dtype downcasting."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that holds their range."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_application(split: str = "train") -> pd.DataFrame:
    path = DATA_DIR / f"application_{split}.csv"
    df = pd.read_csv(path)
    return _downcast(df)


def load_bureau() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "bureau.csv"))


def load_bureau_balance() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "bureau_balance.csv"))


def load_previous_application() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "previous_application.csv"))


def load_pos_cash() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "POS_CASH_balance.csv"))


def load_credit_card_balance() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "credit_card_balance.csv"))


def load_installments() -> pd.DataFrame:
    return _downcast(pd.read_csv(DATA_DIR / "installments_payments.csv"))


def load_columns_description() -> pd.DataFrame:
    # This file is Latin-1 encoded -- a known quirk of the Kaggle export.
    return pd.read_csv(DATA_DIR / "HomeCredit_columns_description.csv", encoding="latin-1")
