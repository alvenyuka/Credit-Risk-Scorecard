"""
Loads the raw Home Credit application table.

The data directory defaults to data/ at the repo root and can be pointed
elsewhere with the HOME_CREDIT_DATA_DIR environment variable.
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("HOME_CREDIT_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))


def load_application(split: str = "train") -> pd.DataFrame:
    """Load application_train.csv or application_test.csv (about 166 MB, read as-is)."""
    path = DATA_DIR / f"application_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"expected {path}. See README.md for how to get the data")
    return pd.read_csv(path)


if __name__ == "__main__":
    train = load_application("train")
    test = load_application("test")
    print("train:", train.shape)
    print("test :", test.shape)
    print("target balance:\n", train["TARGET"].value_counts(normalize=True))
