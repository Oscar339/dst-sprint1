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

Same source and cleaning as Sprint 0 — see
[its data notes](https://github.com/Oscar339/dst-sprint0/blob/main/data/README.md)
for every change made and why.

| File | Source |
|---|---|
| `raw/taiwan-credit-default/default of credit card clients.xls` | [UCI 350 — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) ([zip](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)) |

Cite as: Yeh, I-C. (2009) *Default of Credit Card Clients* [dataset], UCI,
doi:10.24432/C55S3H.
