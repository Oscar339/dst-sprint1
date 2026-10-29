# Data Science Toolbox — Sprint 1: Supervised Prediction (Credit Default)

Assessed Sprint 1 for [Data Science Toolbox](https://dsbristol.github.io/dst/)
(MATHM0029, University of Bristol, 2026-27), worth 30%. Brief:
[Assessment1.pdf](https://dsbristol.github.io/dst/assets/assessments/Assessment1.pdf).
Laid out after the unit's
[example project](https://github.com/dsbristol/dst_example_project), and
continuing the credit default domain of
[Sprint 0](https://github.com/Oscar339/dst-sprint0).

**Deadline: Wednesday 4 November 2026, 12:00 (week 7).**
Group content is final by **Monday 2 November, 12:00** (48 hours before), so
each of us can write the individual reflection.

## The task, in one paragraph

A binary classification problem (default / no default) in which **each member
submits one model**, all evaluated on the **same left-out test data**, using a
**performance metric we agree and test together**, then compared. Half the
effort goes on the metric itself: when does it say something about real-world
generalisation, and in what sense is our "best" model truly best? Show it with
data or simulation.

## Project Group

* Oscar Butler — Data handling and regression
* Alex Robinson — Hazard and clustering
* Louis Mulryan — Markov and forests

Equity: _to be agreed before the work gets underway_ (the brief asks for this).

## Ground rules for the shared test set

Every model is scored on the same split, or the comparison means nothing.

* `report/01-Data.ipynb` makes the train / validation / test split **once**,
  with a fixed random seed, and saves it to `data/processed/`.
* Nobody fits, tunes or selects anything on the test set. Tune on validation.
* Each model writes its test-set predictions (probabilities, not just labels)
  to `data/output/predictions-<name>.csv`, so the metric notebook can score
  them all the same way.

## Reading order

All report content is in `report/`, numbered `NN-Name.ext` in reading order.
_To be filled in as the report is written._

1. `01-Data.ipynb` — downloads and cleans the Taiwan data, using the same
   steps as our Sprint 0 project, and makes the shared split: 60% train,
   20% validation, 20% test, stratified by default, seed 2026.
   **Run this first**: it makes `data/processed/taiwan-train.csv`,
   `taiwan-validation.csv` and `taiwan-test.csv`, which every model loads:

   ```python
   train      = pd.read_csv(ROOT / "data" / "processed" / "taiwan-train.csv", index_col="ID")
   validation = pd.read_csv(ROOT / "data" / "processed" / "taiwan-validation.csv", index_col="ID")
   test       = pd.read_csv(ROOT / "data" / "processed" / "taiwan-test.csv", index_col="ID")
   ```

   `PAY_0` is renamed `PAY_1` in the clean file, and the `PAY_*` codes -2, -1
   and 0 are categories, not months of delay.

2. `02-LogisticRegression.ipynb` (Oscar) — the baseline: regularised logistic
   regression, ratio and trend features, calibration.
3. `03-SVM.ipynb` (Oscar) — support vector machine, tuning C and $\gamma$,
   compared with the linear model.
4. `04-RandomForest.ipynb` (Louis) — random forest and gradient boosting.
5. `05-MarkovChain.ipynb` (Louis) — Markov chain models of the monthly
   payment status.
6. `06-HazardAndClusteringResearch.md` (Alex) — the research behind the hazard
   models and clustering, and what we decided to build.
7. `07-HazardAndClusteringModels.ipynb` (Alex) — the hazard models, the clustering checks on
   the performance metric, and a conclusion. Needs `statsmodels` and
   `lifelines` (in `requirements.txt`); takes about three minutes. The longer
   working is in `Alex/`.
8. `08-FinalModels.ipynb` — every model scored on the shared test set.

### Requirements

Python 3.13. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data is downloaded by `report/01-Data.ipynb` into `data/raw/` (not committed).
Any manual step is stated in the notebook where it happens.

## Evidence

Each member's working, including dead ends, is in their own folder:

* `Oscar/`
* `Alex/`
* `Louis/`

## References

### Data

* Yeh, I-C. (2009). *Default of Credit Card Clients* [dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55S3H

### Papers

* Yeh, I-C. & Lien, C-H. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default of credit card clients.
  *Expert Systems with Applications* 36(2), 2473–2480.
  https://doi.org/10.1016/j.eswa.2007.12.020

_Add as we go. The unit's example project scored lowest on literature and
referencing, so cite widely._
