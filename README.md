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

* Oscar Butler — _to be agreed_
* Alex Robinson — _to be agreed_
* Louis Mulryan — _to be agreed_

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

1. `01-Data.ipynb` — getting and cleaning the data, and the shared split.
   **Run this first**: every later notebook loads its output.

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

_Add as we go. The unit's example project scored lowest on literature and
referencing, so cite widely._
