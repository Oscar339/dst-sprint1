# Data

Following the unit's practice:

* `data/raw/` — datasets exactly as published: downloaded, never edited.
  **Not committed**; `report/01-Data.ipynb` re-creates it.
* `data/processed/` — anything derived, made by code (the cleaned data and the
  shared train / validation / test split). **Not committed**; regenerate it.
* `data/output/` — final outputs: figures, and each model's test-set
  predictions as `predictions-<name>.csv`.

Commit the code that makes these files, not the raw or processed files
themselves (GitHub size limits, and binary data bloats the history).

## Taiwan credit card default

**Load `data/processed/taiwan-clean.csv`, not the raw `.xls`.** Make it by
running `report/01-Data.ipynb` top to bottom. Same source and cleaning as
Sprint 0 ([its data notes](https://github.com/Oscar339/dst-sprint0/blob/577d547/data/README.md));
the output is identical. Changes from the raw file (29,995 rows):

* 5 duplicate customers removed (IDs 14295, 20876, 21882, 27352, 29266).
* `PAY_0` renamed `PAY_1`, so month 1 (September 2005) matches `BILL_AMT1` and `PAY_AMT1`.
* `EDUCATION` 0, 5, 6 merged into 4 (others); `MARRIAGE` 0 merged into 3 (others).
* `PAY_*` values -2 and 0 are **left as they are**. They are undocumented
  status codes, not amounts of delay, so treat -2/-1/0 as categories when modelling.

| File | Source |
|---|---|
| `raw/taiwan-credit-default/default of credit card clients.xls` | [UCI 350 — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) ([zip](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)) |

Cite as: Yeh, I-C. (2009) *Default of Credit Card Clients* [dataset], UCI,
doi:10.24432/C55S3H.
