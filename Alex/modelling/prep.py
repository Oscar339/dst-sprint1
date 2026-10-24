"""Data preparation for Alex's hazard models and clustering (no models here).

Use from a notebook in Alex/modelling/:

    import prep
    wide = prep.load_split()                 # train and validation, one row per customer
    pm = prep.person_months(wide)            # one row per customer per month, model-ready
    feats = prep.customer_features(wide["train"])

Choices (event, months, lags) are explained in ../research/02-model-choices.md.
The person-period layout is the standard one for discrete-time survival models
(Allison, 1982; Singer & Willett, 1993); the code is my own, because the published
helpers (e.g. Wheeler's explode_data) expand durations with fixed covariates,
whereas our covariates change every month.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent      # the repo root
PROCESSED = ROOT / "data" / "processed"
Y = "default payment next month"                           # the label: 1 = defaulted in October
SEED = 2026

# Time runs forwards: t = 1 is April 2005, t = 6 is September, t = 7 is October (the label).
# The raw columns run backwards: PAY_1 is September, PAY_6 is April.
FIRST_T, OCTOBER = 3, 7      # start at June: the payment ratio needs the month before's bill
BEHIND = 2                   # event in t = 3..6: 2 or more months behind (the label is used for t = 7)
STATUS_CODES = [-2, -1, 0, 1, 2, 3]                        # 3 means "3 or more"; -1 (paid on time) is the reference
CATEGORIES = {"status_lag": (STATUS_CODES, -1),            # column: (codes, reference code), as in Oscar/prep.py
              "SEX": ([1, 2], 1),
              "EDUCATION": ([1, 2, 3, 4], 1),
              "MARRIAGE": ([1, 2, 3], 1),
              "t": (list(range(FIRST_T, OCTOBER + 1)), FIRST_T)}
CONTINUOUS = ["util_lag", "payratio_lag", "log_limit", "age"]


def load_split(parts=("train", "validation")):
    """The shared cleaned split from report/01-Data.ipynb, one row per customer.

    The test set is left out by default: nothing is fitted, tuned or chosen on it.
    """
    return {part: pd.read_csv(PROCESSED / f"taiwan-{part}.csv", index_col="ID") for part in parts}


def long_table(df, behind=BEHIND):
    """One row per customer -> one row per customer per month t = 3..7.

    Covariates are lagged one month (month t-1 predicts the event in month t), so a
    month's own status is never used to predict itself. Everyone is at risk every
    month, so customers already behind are kept.
    """
    pieces = []
    for t in range(FIRST_T, OCTOBER + 1):
        now, lag = 7 - t, 8 - t          # raw column numbers of month t and month t-1 (t = 7: lag is September)
        out = pd.DataFrame(index=df.index)
        out["t"] = t
        out["event"] = df[Y] if t == OCTOBER else (df[f"PAY_{now}"] >= behind).astype(int)
        out["status_lag"] = df[f"PAY_{lag}"].clip(upper=3)
        out["util_lag"] = (df[f"BILL_AMT{lag}"] / df["LIMIT_BAL"]).clip(upper=1.5)     # a few go up to ~10
        bill_before = df[f"BILL_AMT{lag + 1}"]                                          # PAY_AMTk pays month k+1's bill
        out["payratio_lag"] = (df[f"PAY_AMT{lag}"] / bill_before).where(bill_before > 0, 1).clip(0, 1)
        out["log_limit"] = np.log(df["LIMIT_BAL"])
        out["age"] = df["AGE"]
        for col in ("SEX", "EDUCATION", "MARRIAGE"):
            out[col] = df[col]
        pieces.append(out)
    return pd.concat(pieces).sort_index(kind="stable")


def person_months(wide, behind=BEHIND):
    """Model-ready person-month tables: t, event, then the features.

    Categories become 0/1 columns (t becomes one dummy per month, so each month has
    its own baseline hazard); continuous columns are standardised with the training
    rows' mean and sd. `wide` must contain "train".
    """
    long = {part: long_table(df, behind) for part, df in wide.items()}
    mean, sd = long["train"][CONTINUOUS].mean(), long["train"][CONTINUOUS].std()
    out = {}
    for part, df in long.items():
        X = pd.DataFrame({"t": df["t"], "event": df["event"]}, index=df.index)
        X[CONTINUOUS] = (df[CONTINUOUS] - mean) / sd
        for col, (codes, ref) in CATEGORIES.items():
            for code in codes:
                if code != ref:
                    X[f"{col}_{code}"] = (df[col] == code).astype(float)
        out[part] = X
    return out


def features(df, months=True):
    """The feature columns; months=False drops the month dummies (October-only and Cox models)."""
    return [c for c in df.columns if c not in ("t", "event") and (months or not c.startswith("t_"))]


def october(df):
    """The t = 7 rows: September covariates, October default. One row per customer."""
    return df[df["t"] == OCTOBER]


def customer_features(df):
    """April-September behaviour summarised per customer, for clustering. Never uses the October label."""
    status = df[[f"PAY_{k}" for k in range(1, 7)]]
    util = pd.DataFrame({k: (df[f"BILL_AMT{k}"] / df["LIMIT_BAL"]).clip(-0.5, 1.5) for k in range(1, 7)})
    paid = pd.DataFrame({k: (df[f"PAY_AMT{k}"] / df[f"BILL_AMT{k + 1}"]).where(df[f"BILL_AMT{k + 1}"] > 0, 1)
                            .clip(0, 1) for k in range(1, 6)})
    f = pd.DataFrame(index=df.index)
    f["mean_util"] = util.mean(axis=1)
    f["util_change"] = util[1] - util[6]                      # September minus April
    f["months_behind"] = (status >= 1).sum(axis=1)            # -2/-1/0 are codes, not months late
    f["worst_status"] = status.clip(lower=0).max(axis=1)
    f["months_inactive"] = (status == -2).sum(axis=1)         # -2 is thought to mean "no consumption"
    f["mean_payratio"] = paid.mean(axis=1)
    f["log_limit"] = np.log(df["LIMIT_BAL"])
    f["age"] = df["AGE"]
    return f
