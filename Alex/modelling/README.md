# Modelling (Alex)

Hazard models and clustering for the Sprint 1 credit default task. Background and choices are in `../research/`;
this folder is the code.

All models use the **shared cleaned split** made by `report/01-Data.ipynb`
(`data/processed/taiwan-{train,validation,test}.csv`, seed 2026), so the customers in each part are the ones Oscar and
Louis use. Nothing here is fitted, tuned or selected on the test set.

## Code (run these)

| File | What it does |
|---|---|
| `prep.py` | Data only: loads the split, builds the person-month table, and the per-customer features for clustering |
| `01-hazard-models.ipynb` | **Every hazard model in one file**: logit, cloglog, L1 / L2 penalised and time-varying Cox, each one function (`fit_glm(link=...)`, `fit_penalised(penalty=..., C=...)`, `fit_cox`). Then the October-only comparison, C tuning, clustered vs naive standard errors, pseudo R², calibration by month and in October, and the test-prediction writer (commented out). A table at the top lists where each piece of code comes from |
| `02-clustering.ipynb` | k-means customer types (train only), hazard AUC within each cluster with bootstrap intervals, leave-one-cluster-out |

Run `report/01-Data.ipynb` first, then the notebooks from this folder (they `import prep`). The person-month table
is built in memory, so `data/processed/hazard-*.csv` is no longer needed.

**Reused code:** only Python is lifted, from documented, citable sources: library routines (statsmodels,
scikit-learn, lifelines, SciPy) wherever one does the job. Sources in another language (Botha & Muller's R code) or
with no licence (Wheeler's blog code) are cited for the approach only; that code here is my own.

## Earlier drafts (markdown, same models)

| File | What it does | Status |
|---|---|---|
| `01-person-month-data.md` | Reshapes the clean split to one row per customer per month (t = 3..7) and saves `data/processed/hazard-*.csv`. **Run first.** | Done |
| `02-discrete-time-logit-cloglog.md` | Pooled discrete-time hazard, logit and cloglog, customer-clustered standard errors, October probability, comparison with an October-only model | Done |
| `03-penalised-discrete-time.md` | L1 / L2 penalised version, C tuned on validation (like Oscar's) | Done |
| `04-cox-time-varying.md` | Cox with time-varying covariates, the continuous-time comparison | Done |
| `05-clustering.md` | k-means customer types (train only), AUC within each cluster, leave-one-cluster-out, and whether clusters help prediction | Done |

## Decision: which models

Two approaches, not a long list. **Submitted model: the pooled discrete-time logit hazard (`02`).** **Second approach:
clustering (`05`)**, used for per-cluster metrics, leave-one-cluster-out, and as a prediction test. The cloglog,
penalised and Cox files (`02` to `04`) stay as sensitivity rows in one table, not separate models. Frailty and the
mixture cure model are **not planned**: the cure model has no Python package with covariates (only R: `smcure`,
`mixcure`, `EMGCR`), so it would be our own EM, which is a lot for the gain.

## Setup

From the repo root, with the packages in `requirements.txt`, plus two that are not in it yet:

```sh
pip install statsmodels lifelines
```

(`xlrd` is needed once, by `report/01-Data.ipynb`.) Then run `report/01-Data.ipynb` to make the split, then
`01-person-month-data.md` here.

## What was found (validation set, probability of default in October)

| Model | AUC | Brier | Log loss |
|---|---|---|---|
| Hazard, logit (pooled months) | 0.744 | 0.138 | 0.446 |
| Hazard, cloglog | 0.744 | 0.139 | 0.450 |
| Logit, October rows only | 0.750 | 0.137 | 0.439 |
| Penalised logit, best C | 0.744 | n/a | 0.446 |
| Cox, time-varying | 0.745 | 0.141 | 0.460 |

Clustering (k = 5, validation): the hazard model's AUC is 0.74 in the high-risk cluster (51% default) and only
0.64 to 0.65 in the other four; leaving a cluster out costs up to 0.02 AUC. A separate hazard model per cluster scores
0.760 against 0.744 for one model, which needs checking before it is believed (see the notes in `05`).

Pooling the earlier months does **not** beat fitting on October alone. The earlier months use a proxy event
(2+ months behind) and October is the real label, so the shared coefficients blend two questions. That is a result to
report, not hide. These numbers are single runs on the validation set; re-run the files rather than quote this table.

## Sources of the code

Each file lists its own sources at the top. In short: the person-period + binomial GLM recipe follows Allison (1982) and
[Andrew Wheeler's Python write-up](https://andrewpwheeler.com/2020/09/26/discrete-time-survival-models-in-python/);
the Cox code follows the [lifelines time-varying regression docs](https://lifelines.readthedocs.io/en/stable/Time%20varying%20survival%20regression.html);
penalised fits use scikit-learn and statsmodels as documented. The person-month construction and the Cox October
prediction are my own.
