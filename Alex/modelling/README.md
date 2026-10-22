# Modelling (Alex)

Hazard models and clustering for Sprint 1, in the order the work was done: what the lectures teach, what I decided,
the further research (in `../research/`), then the models.

| Order | File | What it is |
|---|---|---|
| 1 | `01-lecture-notes.ipynb` | What blocks 01 to 05 of the Data Science Toolbox lectures teach that bears on Sprint 1, and what each means for us |
| 2 | `02-plan.ipynb` | The decisions (models, event, metrics, validation, clustering) and where each comes from; the four generalisation tests from lecture 02.2 |
| 3 | `../research/` | Further research: hazard model families, choices inside the model, clustering, references |
| 4 | `03-data.ipynb` | The person-month table: timeline, event, lags, checks, design matrix |
| 5 | `04-hazard-models.ipynb` | **Every hazard model**: logit, cloglog, L1 / L2 penalised, time-varying Cox, and the October-only logit. Grouped 10-fold cross-validation, nested models with AIC / BIC, penalty chosen by cross-validation, validation check with ROC / PR curves and a paired bootstrap, calibration, next-month test, test-prediction writer (commented out) |
| 6 | `05-clustering.ipynb` | k-means customer types, sensitivity analysis over K, Gaussian mixture with BIC, PCA, the hazard model's AUC within each cluster, leave-one-cluster-out |
| | `prep.py` | Data preparation shared by 03 to 05 (no models) |

## Running

Python only. From the repo root, install `requirements.txt` (it includes `statsmodels` and `lifelines`), run
`report/01-Data.ipynb` to make the shared split, then the notebooks here in order (they `import prep`, so run them
from this folder). `04` takes about two minutes, `05` about one. The saved outputs are from a full run.

## Rules followed

* Same shared split as Oscar and Louis (seed 2026). Everything is chosen on the training customers by
  cross-validation; the validation set is a single check; the test set is used once, for the final model.
* Code is lifted only from documented, citable Python sources (library documentation and examples). Other sources
  are cited for the approach, and the code is my own. Each notebook lists its sources at the top.
