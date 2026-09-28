"""
Week 6 blind rebuild, step 1: load the raw application table.

Deliberately written without opening the previous pipeline's loader, which is in
this repo's git history. The only things looked at were the raw CSV header and
Kaggle's own description of the 8 tables (bureau, bureau_balance,
previous_application, POS_CASH_balance, credit_card_balance,
installments_payments, application_train/test).
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_application(split: str = "train") -> pd.DataFrame:
    """Load application_train.csv or application_test.csv.

    No dtype downcasting yet on this first pass. Get something correct and
    running before optimizing memory. application_train.csv is ~166MB, small
    enough to load as-is.
    """
    fname = f"application_{split}.csv"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"expected {path}. See README.md for how to get the data")
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    train = load_application("train")
    test = load_application("test")
    print("train:", train.shape)
    print("test :", test.shape)
    print("target balance:\n", train["TARGET"].value_counts(normalize=True))
