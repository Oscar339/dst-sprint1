# Modelling (Alex)

This folder contains the hazard models and clustering for Sprint 1, in the order the work was done. We start with
the plan, then the further research behind it (in `../research/`), and then the models.

| Order | File | What it is |
|---|---|---|
| 1 | `01-plan.md` | What we decided to do and why, and how we test the different kinds of generalisation |
| 2 | `../research/` | The further research: types of hazard model, what we can vary, clustering and references |
| 3 | `02-data.ipynb` | Builds the person-month table: the timeline, the event, the lags and the checks |
| 4 | `03-hazard-models.ipynb` | Every hazard model (logit, cloglog, L1 / L2 penalised and time-varying Cox) plus an October-only logistic regression. Covers cross-validation by customer, AIC / BIC, choosing the penalty, ROC and precision-recall curves, a bootstrap, calibration and a next-month test. The test-set predictions are written here at the end (commented out for now) |
| 5 | `04-clustering.ipynb` | Groups customers with k-means, tries a range of K, compares with a Gaussian mixture, and checks the hazard model within each cluster and with each cluster left out |
| | `prep.py` | The data preparation used by notebooks 02 to 04 |

## Running

The project is Python only. From the repo root, install `requirements.txt` (this includes `statsmodels` and
`lifelines`), run `report/01-Data.ipynb` to make the shared split, and then run the notebooks here in order. They
import `prep.py`, so they need to be run from this folder. Notebook 03 takes about two minutes and 04 about one. The
saved outputs are from a full run.

## Rules followed

* We use the same shared split as Oscar and Louis (seed 2026). Everything is chosen on the training customers using
  cross-validation, the validation set is used as a single check, and the test set is only used once, for the final
  model.
* Code is only lifted from documented Python sources, such as library documentation and examples. Other sources are
  cited for the approach, and the code is our own.
