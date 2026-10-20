# 01 Person-month data

Every hazard model in this folder is fitted on the same **person-month table**: one row per customer per month at
risk, instead of one row per customer. This file builds it from the shared cleaned split
(`data/processed/taiwan-{train,validation,test}.csv`, made by `report/01-Data.ipynb`), so the customers in each part
are exactly the ones Oscar and Louis use. Run this file first; the others load what it saves.

Run the code blocks top to bottom, from any folder inside the repo.

## The timeline

The raw columns count backwards: month 1 is September 2005, month 6 is April 2005, and the label is October.
I number time forwards, `t = 1..7`:

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Month | Apr | May | Jun | Jul | Aug | Sep | **Oct** |
| Status column | `PAY_6` | `PAY_5` | `PAY_4` | `PAY_3` | `PAY_2` | `PAY_1` | (the label) |

Choices made here (all from `../research/02-model-choices.md`; change them in one place and rerun):

| Choice | What I used | Why |
|---|---|---|
| Event, t = 1..6 | Status of 2 or more months behind (`PAY_k >= 2`) | The Sprint 0 proxy; the data has no default date before October |
| Event, t = 7 | The real label `default payment next month` | The only true default |
| Risk set | Everyone, every month (recurrent), previous status as a covariate | Scores every customer, including those already behind |
| Covariates | Lagged by one month (values from t-1 predict the event at t) | A lag of 0 would predict a month from its own data (leakage) |
| First month modelled | t = 3 (June) | The payment ratio needs two months of history (`PAY_AMT` pays the *previous* month's bill) |

So each customer has 5 rows (t = 3..7), and **the t = 7 row is the one that is scored against the shared test set**:
its covariates are September's, and its event is the October default. Nothing from October is used to predict October.

## 1. Load the shared split

```python
from pathlib import Path

import numpy as np
import pandas as pd

# The repo root is the nearest parent folder that has data/processed/
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists())
PROCESSED = ROOT / "data" / "processed"
Y = "default payment next month"

raw = {part: pd.read_csv(PROCESSED / f"taiwan-{part}.csv", index_col="ID")
       for part in ("train", "validation", "test")}
print({part: df.shape for part, df in raw.items()})
```

## 2. Reshape: one row per customer per month

`PAY_AMTk` is what was paid in month k, and it pays the bill of month k+1 (the older one), so the payment ratio for
lagged month k is `PAY_AMTk / BILL_AMT(k+1)`. Money amounts are not logged here: the ratios already scale them.

```python
FIRST_T, LAST_T = 3, 7           # t = 7 is October
BEHIND = 2                       # event proxy: this many months behind or more
STATUS_CODES = [-2, -1, 0, 1, 2, 3]    # 3 means "3 or more months behind"; -1 (paid on time) is the reference


def person_month(df):
    """Wide table (one row per customer) -> long table (one row per customer per month t = 3..7)."""
    pieces = []
    for t in range(FIRST_T, LAST_T + 1):
        lag = 8 - t              # the column number of month t-1: t=7 -> 1 (September), t=3 -> 5 (May)
        now = 7 - t              # the column number of month t:   t=6 -> 1 (September), t=3 -> 4 (June)
        out = pd.DataFrame(index=df.index)
        out["t"] = t
        out["event"] = df[Y] if t == LAST_T else (df[f"PAY_{now}"] >= BEHIND).astype(int)
        out["status_lag"] = df[f"PAY_{lag}"].clip(upper=3)
        out["util_lag"] = (df[f"BILL_AMT{lag}"] / df["LIMIT_BAL"]).clip(upper=1.5)    # a few go up to ~10
        bill_before = df[f"BILL_AMT{lag + 1}"]
        out["payratio_lag"] = (df[f"PAY_AMT{lag}"] / bill_before).where(bill_before > 0, 1).clip(0, 1)
        out["log_limit"] = np.log(df["LIMIT_BAL"])
        out["age"] = df["AGE"]
        for col in ("SEX", "EDUCATION", "MARRIAGE"):
            out[col] = df[col]
        pieces.append(out)
    return pd.concat(pieces).sort_index(kind="stable")


long = {part: person_month(df) for part, df in raw.items()}
print(long["train"].head(10))
```

## 3. Checks

```python
train = long["train"]

# 5 rows per customer, and the October row's event is exactly the label
assert (train.groupby(level=0).size() == LAST_T - FIRST_T + 1).all()
october = train[train["t"] == LAST_T]
assert (october["event"] == raw["train"].loc[october.index, Y]).all()

# Event rate by month. October is the real default rate (22%); the other months are the proxy.
print(train.groupby("t")["event"].agg(["mean", "sum"]).round(3))
```

## 4. Model-ready features

Status and the demographics become 0/1 columns (reference codes `-1` for status and `1` for the others, the same
references as `Oscar/prep.py`), month becomes a dummy per month (so each month has its own baseline hazard
$\alpha_t$), and the four continuous columns are standardised using the **training** rows only.

```python
CONTINUOUS = ["util_lag", "payratio_lag", "log_limit", "age"]
mean, sd = train[CONTINUOUS].mean(), train[CONTINUOUS].std()


def design(df):
    """Long table -> numeric feature table (no event column)."""
    X = pd.DataFrame(index=df.index)
    X[CONTINUOUS] = (df[CONTINUOUS] - mean) / sd
    for col, codes, ref in [("status_lag", STATUS_CODES, -1),
                            ("SEX", [1, 2], 1), ("EDUCATION", [1, 2, 3, 4], 1), ("MARRIAGE", [1, 2, 3], 1),
                            ("t", list(range(FIRST_T, LAST_T + 1)), FIRST_T)]:
        for code in codes:
            if code != ref:
                X[f"{col}_{code}"] = (df[col] == code).astype(int)
    return X


for part, df in long.items():
    out = design(df)
    # The ID repeats (5 rows per customer), so assign by position, never join on the index
    out.insert(0, "t", df["t"].to_numpy())          # keep t as a plain number as well, for the Cox model
    out["event"] = df["event"].to_numpy()
    out.to_csv(PROCESSED / f"hazard-{part}.csv")    # ID is the first column
    print(f"hazard-{part}.csv", out.shape)

print(list(out.columns))
```

The three `hazard-*.csv` files are derived data in `data/processed/`, which is not committed (see `data/README.md`),
so regenerate them by running this file.

## What this table can and cannot do

* The **month dummies carry a lot of the signal**: the proxy event is much more common in some months than others,
  and October (the real label) has its own level. The `t_7` coefficient is the October baseline, estimated from the
  October rows only.
* Rows from the same customer are correlated. The point estimates are fine; the standard errors need clustering by
  customer (done in `02`), and any tuning needs to keep each customer's rows together.
* t = 1 and 2 are dropped. If we want them back, drop the payment ratio and start at t = 2.
