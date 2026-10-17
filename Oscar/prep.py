"""Data preparation for Oscar's models: the fixes worked out in 01-eda.ipynb,
plus the ratio features from 02-logistic-regression.ipynb (section 3).

Use from a notebook in Oscar/:

    import prep
    X_train, y_train = prep.load("train")
    X_val, y_val = prep.load("validation")
    X_train, X_val = prep.transform(X_train, X_val)                # fixes 1-4
    X_train, X_val = prep.transform(X_train, X_val, ratios=True)   # fixes 1-4 + ratios
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent     # the repo root
PROCESSED = ROOT / "data" / "processed"
Y = "default payment next month"                  # the label: 1 = defaulted

STATUS = [f"PAY_{i}" for i in range(1, 7)]
STATUS_CODES = [-2, -1, 0, 1, 2, 3]               # 3 means "3 or more months late"
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
AMOUNTS = ["LIMIT_BAL"] + BILLS + PAYMENTS
DEMOGRAPHICS = {"SEX": [1, 2],                    # 1 = male, 2 = female
                "EDUCATION": [1, 2, 3, 4],        # graduate, university, high school, others
                "MARRIAGE": [1, 2, 3]}            # married, single, others


def load(part):
    """Load one part of the shared split ("train", "validation" or "test") as X, y."""
    df = pd.read_csv(PROCESSED / f"taiwan-{part}.csv", index_col="ID")
    return df.drop(columns=Y), df[Y]


def one_hot(df, col, codes, reference):
    """Replace column `col` with one 0/1 column per code, dropping `reference`."""
    values = pd.Categorical(df[col], categories=codes)
    dummies = pd.get_dummies(values, prefix=col, dtype=int)
    dummies.index = df.index
    # A row of all zeros means the reference code; keeping its column too would
    # make the columns add up to 1 on every row, duplicating the intercept.
    dummies = dummies.drop(columns=f"{col}_{reference}")
    return df.drop(columns=col).join(dummies)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))      # log1p(a) = log(1 + a)


# Fix 1: repayment status as categories, 3-8 merged, reference -1 (paid on time)
def fix_status(df):
    df = df.copy()
    for col in STATUS:
        df[col] = df[col].clip(upper=3)
        df = one_hot(df, col, STATUS_CODES, reference=-1)
    return df


# Fix 2: signed log of every money amount
def fix_amounts(df):
    df = df.copy()
    df[AMOUNTS] = signed_log(df[AMOUNTS])
    return df


# Fix 3: sex, education and marriage as categories, reference code 1
def fix_demographics(df):
    for col, codes in DEMOGRAPHICS.items():
        df = one_hot(df, col, codes, reference=1)
    return df


# Ratio features (02-logistic-regression.ipynb, section 3), built from the raw amounts
def add_ratios(df):
    """Utilisation (bill / limit, 6 months) and pay ratio (payment / last month's bill, 5 months)."""
    df = df.copy()
    for k in range(1, 7):
        df[f"UTIL_{k}"] = (df[f"BILL_AMT{k}"] / df["LIMIT_BAL"]).clip(upper=1.5)   # a few go up to ~10
    for k in range(1, 6):
        bill, paid = df[f"BILL_AMT{k + 1}"], df[f"PAY_AMT{k}"]    # PAY_AMTk pays last month's bill
        df[f"PAYRATIO_{k}"] = (paid / bill).where(bill > 0, 1).clip(0, 1)   # nothing owed -> 1
    return df


def prepare(df, ratios=False):
    """Fixes 1-3, plus the ratio features if `ratios`.

    These follow fixed rules and learn nothing from the data, so they can be
    applied to each part of the split separately.
    """
    if ratios:
        df = add_ratios(df)                       # before fix 2 logs the amounts
    return fix_demographics(fix_amounts(fix_status(df)))


def standardise(train, *others):
    """Fix 4: scale to mean 0, sd 1, learning the means and sds from `train` only."""
    scaler = StandardScaler().fit(train)
    scaled = [pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
              for df in (train, *others)]
    return scaled


def transform(train, *others, ratios=False):
    """All fixes (1-4): raw data in, model-ready data out. `ratios` adds the ratio features.

    Pass the training data first, then any other parts of the split; the
    scaling is learned from the training data only. Returns them in the same
    order: X_train, X_val = transform(X_train, X_val)
    """
    return standardise(*[prepare(df, ratios) for df in (train, *others)])
